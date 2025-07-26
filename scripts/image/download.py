import os 
from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download
from modelscope import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import re
import torch
import sys
import shutil
import psutil
import gc
import json
import time
from safetensors.torch import save_file
from flask import Flask, request

app = Flask(__name__)

def get_current_memory_gb():
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    return info.uss / 1024. / 1024. / 1024.

def get_size(sz):
    mul = 1
    for dim in sz:
        mul = mul * dim
    return mul

def getType(key : str, _tied_weights_keys: list[str]):
    """
    Get download type for a layer name to ensure equally layer partition
    0: embed tokens
    1: decoder layer 0
    2: lm_head, etc.
    3: none
    """
    if "lm_head" in key:
        # final lm head
        if _tied_weights_keys is not None:
            for tied_key in _tied_weights_keys:
                if "lm_head" in tied_key:
                    return 3
        return 2
    if "rotary" in key:
        return 3
    if "emb" in key or "wte" in key:
        # embedding
        # Check if lm_head is a copy of embed_token
        if _tied_weights_keys is not None:
            for tied_key in _tied_weights_keys:
                if "lm_head" in tied_key:
                    return [0, 2]
        return 0
    pattern = re.compile(r'\d+')
    substrs = pattern.findall(key)
    if len(substrs) > 0:
        # decoding layers
        id = int(substrs[0])    # get layer id
        if id == 0:
            return 1
        return 3
    else:
        # final layer norm
        return 2

def get_layer_distribution(tot_layer, pp_size, state_dict, _tied_weights_keys):
    layers = []
    embed_weights = 0
    decoder_weights = 0
    final_weights = 0
    for key, value in state_dict.items():
        tensor_type_list = getType(key, _tied_weights_keys)
        tensor_size = value.nelement() * value.element_size()
        if isinstance(tensor_type_list, int):
            tensor_type_list = [tensor_type_list]
        for tensor_type in tensor_type_list:
            if tensor_type == 0:
                embed_weights += tensor_size
            elif tensor_type == 1:
                decoder_weights += tensor_size
            elif tensor_type == 2:
                final_weights += tensor_size
    if embed_weights == 0 and final_weights == 0:
        # equally distribution
        num_layers_per_node = int(tot_layer / pp_size)
        sum = 0
        for i in range(pp_size):
            sum += num_layers_per_node 
            layers.append(sum)
        if sum < tot_layer:
            for i in range(0, tot_layer - sum):
                layers[-i-1] += 1
    else:
        # consider embedding and lm_head
        layers = [0] * pp_size
        weights = [0] * pp_size
        weights[0] = embed_weights
        weights[-1] = final_weights
        for i in range(tot_layer):
            # find the stage that holds minimum weights
            mn_id = 0
            mn_value = weights[0]
            for j in range(1, pp_size):
                if weights[j] <= mn_value:
                    mn_value = weights[j]
                    mn_id = j
            weights[mn_id] += decoder_weights
            layers[mn_id] += 1
        for i in range(1, pp_size):
            layers[i] += layers[i-1]
    return layers

def transform(key : str, layers : list[int], _tied_weights_keys: list[str]):
    """
    Get host id for a layer name.
    """
    num_hosts = len(layers)
    if "lm_head" in key:
        # final lm head
        if _tied_weights_keys is not None:
            for tied_key in _tied_weights_keys:
                if "lm_head" in tied_key:
                    return [], key
        return num_hosts - 1, key
    if "rotary" in key:
        return range(0, num_hosts), key
    if "emb" in key or "wte" in key:
        # embedding
        # Check if lm_head is a copy of embed_token
        if _tied_weights_keys is not None:
            for tied_key in _tied_weights_keys:
                if "lm_head" in tied_key:
                    return [0, num_hosts - 1], key
        return 0, key
    pattern = re.compile(r'\d+')
    substrs = pattern.findall(key)
    if len(substrs) > 0:
        # decoding layers
        id = int(substrs[0])    # get layer id
        last_sum = 0
        for i, sum in enumerate(layers):
            if sum > id:
                key = key.replace(str(id), str(int(id - last_sum)), 1)
                return i, key
            last_sum = sum
        raise ValueError("model layer id is too large")
    else:
        # final layer norm
        return num_hosts - 1, key

    raise ValueError("do not recognize key")

def partition(model_path: str, pp_size: int, dest_pp_size: int, model_part: int):
    if model_path.endswith("/"):
        model_path = model_path[:-1]

    print(f"partition: model_path = {model_path}")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.torch_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype = config.torch_dtype,
        trust_remote_code = True
    )
    print(f"torch dtype = {model.config.torch_dtype}")

    if pp_size == 1:
        config.save_pretrained(model_path)
        # delete large files
        items = os.listdir(model_path)
        for item in items:
            if item.endswith(".bin") or item.endswith(".pth") or item.endswith(".pt") or item.endswith(".gguf") or item.endswith(".safetensors"):
                os.remove(model_path + "/" + item)
        # save state_dict as a single file
        save_file(model.state_dict(), model_path + "/model.safetensors", metadata = {'format': 'pt'})
        return

    if hasattr(model.config, "num_hidden_layers"):
        tot_layer = model.config.num_hidden_layers
    elif hasattr(model.config, "num_layers"):
        tot_layer = model.config.num_layers
    else:
        raise ValueError("Cannot obtain layer num.")

    # Obtain layer count for each pp stage
    state_dict = model.state_dict()
    _tied_weights_keys = None
    if config.tie_word_embeddings and hasattr(model, "_tied_weights_keys"):
        _tied_weights_keys = model._tied_weights_keys
    layers = get_layer_distribution(tot_layer, pp_size, state_dict, _tied_weights_keys)       # prefix sum of number of decoder layers of each pp stage
    dest_layers = get_layer_distribution(tot_layer, dest_pp_size, state_dict, _tied_weights_keys)       # prefix sum of number of decoder layers of each pp stage
    state_dict = None
    _tied_weights_keys = None

    print(f"Layer number = {tot_layer}, layers: ")
    for num in layers:
        print(num)

    items = os.listdir(model_path)
    items_needed = []
    for item in items:
        if item.endswith(".safetensors"):
            continue
        if item.endswith(".bin") or item.endswith(".pth") or item.endswith(".pt") or item.endswith(".gguf"):
            continue
        if os.path.isfile(model_path + "/" + item):
            items_needed.append(item)

    # save related items and config
    if model_part == 0:
        start_pp_rank = 0
        if pp_size <= 4:
            end_pp_rank = pp_size - 1
        else:
            end_pp_rank = 3
    else:
        start_pp_rank = 4
        end_pp_rank = pp_size - 1
    for id in range(start_pp_rank, end_pp_rank + 1):
        model_path_ = model_path.replace("-0/", "-" + str(id) + "/") + "-" + str(id) + "-" + str(pp_size)
        # copy all needed files
        os.makedirs(model_path_, exist_ok=True)
        for file_name in items_needed:
            shutil.copy(model_path + "/" + file_name, model_path_)
        print(f"file copied to {model_path_}!")
        # save config
        num_layers = layers[id] - (0 if id == 0 else layers[id-1])
        config_ = config
        if hasattr(config_, "num_hidden_layers"):
            config_.num_hidden_layers = num_layers
        elif hasattr(config_, "num_layers"):
            config_.num_layers = num_layers
        config_.save_pretrained(model_path_)
        # save dest info
        if dest_pp_size < pp_size:
            for dest_pp_rank in range(dest_pp_size):
                pp_tail = "-" + str(dest_pp_rank) + "-" + str(dest_pp_size)
                dest_num_hidden_layers = dest_layers[dest_pp_rank] - (0 if dest_pp_rank == 0 else dest_layers[dest_pp_rank-1])
                with open(model_path_+ "/dest_info" + pp_tail, "w") as f:
                    f.write(str(dest_num_hidden_layers))


    # save parameters
    for id in range(start_pp_rank, end_pp_rank + 1):
        model_path_ = model_path.replace("-0/", "-" + str(id) + "/") + "-" + str(id) + "-" + str(pp_size)
        # obtain split state dict
        _tied_weights_keys = None
        if config.tie_word_embeddings and hasattr(model, "_tied_weights_keys"):
            _tied_weights_keys = model._tied_weights_keys
        split_state_dict = {}
        state_dict = model.state_dict()
        duplicate_state_dict = [{} for _ in range(dest_pp_size)]
        dest_split_state_dict = [{} for _ in range(dest_pp_size)]
        for key, value in state_dict.items():
            # get normal split state dict
            new_id, new_key = transform(key, layers, _tied_weights_keys)
            if isinstance(new_id, int):
                new_id = [new_id]
            split_state_dict_key = None
            for id_ in new_id:
                if id_ == id:
                    print("{}: {}, use new key = {}".format(new_id, key, new_key))
                    split_state_dict[new_key] = value
                    split_state_dict_key = new_key
                    break
            if dest_pp_size < pp_size:
                # get dest split state dict for each possible dest_pp_rank
                dest_new_id, dest_new_key = transform(key, dest_layers, _tied_weights_keys)
                for dest_pp_rank in range(dest_pp_size):
                    if isinstance(dest_new_id, int):
                        dest_new_id = [dest_new_id]
                    for id_ in dest_new_id:
                        if id_ == dest_pp_rank:
                            print("For dest_pp_rank [{}] - {}: {}, use new key = {}".format(dest_pp_rank, dest_new_id, key, dest_new_key))
                            if split_state_dict_key is not None:
                                duplicate_state_dict[dest_pp_rank][split_state_dict_key] = dest_new_key
                            else:
                                dest_split_state_dict[dest_pp_rank][dest_new_key] = value
                            break

        # delete all other state dicts so that we have enough memory for current state_dict
        model = None
        state_dict = None
        _tied_weights_keys = None
        gc.collect()

        # save split_state_dict
        save_file(split_state_dict, model_path_ + "/model.safetensors")
        print(f"model saved to {model_path_}!")
        # save dest
        if dest_pp_size < pp_size:
            for dest_pp_rank in range(dest_pp_size):
                pp_tail = "-" + str(dest_pp_rank) + "-" + str(dest_pp_size)
                save_file(dest_split_state_dict[dest_pp_rank], model_path_ + "/dest_model" + pp_tail + ".safetensors", metadata = {'format': 'pt'})
                with open(model_path_ + "/duplicate_state_dict" + pp_tail + ".json", "w") as f:
                    json.dump(duplicate_state_dict[dest_pp_rank], f)
                print(f"dest model [{dest_pp_rank}] saved to {model_path_}!")
        if id < end_pp_rank:
            # load model for next id
            split_state_dict = None
            dest_split_state_dict = None
            gc.collect()

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype = config.torch_dtype,
                trust_remote_code = True
            )
            print(f"next iteration model loaded!")
    
    # delete original files
    # shutil.rmtree(model_path)

@app.route('/invoke', methods=['POST', 'GET'])
def download_model():
    model_id = os.getenv('MODEL_ID', '')
    revision = os.getenv('MODEL_VERSION', '')
    cache_dir = os.getenv('MODELSCOPE_CACHE', '')
    sdk_token = os.getenv('MODELSCOPE_TOKEN', '')
    pp_size = int(os.getenv('PP_SIZE', '1'))
    dest_pp_size = int(os.getenv('DEST_PP_SIZE', '1'))
    model_part = int(os.getenv('MODEL_PART', ''))

    if pp_size <= 4 and model_part == 1:
        return "success"

    if revision == "latest":
        revision = ""

    # check if file exists
    max_pp_size = pp_size
    if model_part == 0 and pp_size > 4:
        max_pp_size = 4
    model_path = cache_dir + "-" + str(max_pp_size - 1) + "/" + model_id.replace('.', '___')
    print(f"model_path = {model_path}")
    pp_tail = ""
    if max_pp_size > 1:
        pp_tail = "-" + str(max_pp_size - 1) + "-" + str(pp_size)
    flag1 = False       # check the last stage
    flag2 = False       # check the first dest stage
    if os.path.exists(model_path + pp_tail):
        items = os.listdir(model_path + pp_tail)
        for item in items:
            if item == 'model.safetensors':
                flag1 = True
                break
    if pp_size == dest_pp_size or dest_pp_size == 1 and model_part == 1:
        flag2 = True
    else:
        if model_part == 0:
            # check that dest model is in the first stage
            last_pp_rank = 0
            last_dest_pp_rank = 0
        else:
            # check that dest model is in the last stage
            last_pp_rank = pp_size - 1
            last_dest_pp_rank = dest_pp_size - 1
        model_path = cache_dir + "-" + str(last_pp_rank) + "/" + model_id.replace('.', '___')
        pp_tail = "-" + str(last_pp_rank) + "-" + str(pp_size)
        if os.path.exists(model_path + pp_tail):
            items = os.listdir(model_path + pp_tail)
            flag2_1 = False
            flag2_2 = False
            dest_pp_tail = '-' + str(last_dest_pp_rank) + '-' + str(dest_pp_size)
            for item in items:
                if item == 'dest_model' + dest_pp_tail + '.safetensors':
                    flag2_1 = True
                elif item == 'dest_info' + dest_pp_tail:
                    with open(model_path + pp_tail + "/dest_info" + dest_pp_tail, "r") as f:
                        res = f.read()
                        res = res.split()
                        if len(res) == 1:
                            flag2_2 = True
            if flag2_1 and flag2_2:
                flag2 = True
    if flag1 and flag2:
        print("partition completed, exit.")
        return "success"
    
    # login first.
    HubApi().login(sdk_token)
    cache_dir = cache_dir + "-0"
    print(f"cache_dir = {cache_dir}")
    if len(revision) > 0:
        model_dir = snapshot_download (model_id =model_id, 
                           revision =revision,
                           cache_dir = cache_dir)
    else:
        model_dir = snapshot_download (model_id =model_id, 
                            cache_dir = cache_dir)
    print("download model scuccess, execute partition...")
    partition(model_dir, pp_size, dest_pp_size, model_part)

    return "success"

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=9000)