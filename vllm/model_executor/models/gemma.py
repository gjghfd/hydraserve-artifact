# coding=utf-8
# Copyright 2023 The vLLM team.
# Copyright (c) Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Gemma model compatible with HuggingFace weights."""
from functools import lru_cache
from typing import Iterable, List, Optional, Tuple
import time
import os

import torch
from torch import nn
from transformers import GemmaConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import LoRAConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import send_states, recv_states, get_duplicate_state_dict, extract_dest_layer
from vllm.sequence import SamplerOutput

logger = init_logger(__name__)


@lru_cache(maxsize=None)
def _get_gemma_act_fn(
    hidden_act: Optional[str],
    hidden_activation: Optional[str],
) -> nn.Module:
    if hidden_activation is None:
        if hidden_act is not None:
            logger.warning(
                "Gemma's activation function was incorrectly set to exact GeLU "
                "in the config JSON file when it was initially released. "
                "Changing the activation function to approximate GeLU "
                "(`gelu_pytorch_tanh`). If you want to use the legacy "
                "`%s`, edit the config JSON to set "
                "`hidden_activation=%s` instead of `hidden_act`. "
                "See https://github.com/huggingface/transformers/pull/29402 "
                "for more details.", hidden_act, hidden_act)
        return GeluAndMul(approximate="tanh")
    elif hidden_activation == "gelu_pytorch_tanh":
        return GeluAndMul(approximate="tanh")
    elif hidden_activation == "gelu":
        return GeluAndMul(approximate="none")
    else:
        raise ValueError(f"Activation function {hidden_act} is not "
                         "supported for Gemma models.")


class GemmaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: Optional[str] = None,
        hidden_activation: Optional[str] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config)
        self.act_fn = _get_gemma_act_fn(hidden_act, hidden_activation)

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class GemmaAttention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 head_dim: int,
                 max_position_embeddings: int = 8192,
                 rope_theta: float = 10000,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=self.rope_theta,
            is_neox_style=True,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class GemmaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: GemmaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            quant_config=quant_config,
        )
        self.mlp = GemmaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            hidden_activation=getattr(config, "hidden_activation", None),
            quant_config=quant_config,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class GemmaModel(nn.Module):

    def __init__(
        self,
        config: GemmaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.pp_rank = int(os.getenv('PP_RANK', '0'))
        self.pp_size = int(os.getenv('PP_SIZE', '1'))

        if self.pp_rank == 0 or self.pp_rank == self.pp_size - 1:
            if GemmaForCausalLM.duplicate_embed_tokens is not None:
                self.embed_tokens = GemmaForCausalLM.duplicate_embed_tokens
            else:
                self.embed_tokens = VocabParallelEmbedding(
                    config.vocab_size,
                    config.hidden_size,
                )
        if len(GemmaForCausalLM.duplicate_decoder_layer) > 0:
            layer_list = []
            for i in range(config.num_hidden_layers):
                if i in GemmaForCausalLM.duplicate_decoder_layer:
                    layer_list.append(GemmaForCausalLM.duplicate_decoder_layer[i])
                else:
                    layer_list.append(GemmaDecoderLayer(config, quant_config))
            self.layers = nn.ModuleList(layer_list)
        else:
            self.layers = nn.ModuleList([
                GemmaDecoderLayer(config, quant_config)
                for _ in range(config.num_hidden_layers)
            ])
        if self.pp_rank == self.pp_size - 1:
            if GemmaForCausalLM.duplicate_norm is not None:
                self.norm = GemmaForCausalLM.duplicate_norm
            else:
                self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.pp_rank == 0:
            # Normalize the embedding by sqrt(hidden_size)
            # The normalizer's data type should be downcasted to the model's
            # data type such as bfloat16, not float32.
            # See https://github.com/huggingface/transformers/pull/29402
            normalizer = self.config.hidden_size**0.5
            self.register_buffer("normalizer", torch.tensor(normalizer))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
            hidden_states *= self.normalizer
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
            )
        if self.pp_rank == self.pp_size - 1:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            # reserve residual to next stage
            hidden_states = hidden_states + residual
        return hidden_states


class GemmaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    # Gemma does not apply LoRA to the embedding layer.
    embedding_modules = {}
    embedding_padding_modules = []

    # For duplicate state dict
    prev_model = None
    duplicate_embed_tokens = None
    duplicate_norm = None
    duplicate_decoder_layer = {}

    def __init__(
        self,
        config: GemmaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        del lora_config  # Unused.
        super().__init__()
        self.check_duplicate()
        self.config = config
        self.quant_config = quant_config
        self.model = GemmaModel(config, quant_config)
        self.pp_rank = self.model.pp_rank
        self.pp_size = self.model.pp_size
        if self.pp_rank == self.pp_size - 1:
            self.logits_processor = LogitsProcessor(config.vocab_size)
            self.sampler = Sampler()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if self.pp_rank > 0:
            stime = time.time()

            device = input_ids.device
            shape = [input_ids.shape[0], self.config.hidden_size]
            hidden_states = recv_states(self.read_socket, self.config.torch_dtype, device, shape)

            hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, inputs_embeds=hidden_states)
        else:
            hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata)
        if self.pp_rank < self.pp_size - 1:
            send_states(self.write_socket, hidden_states)

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        if self.pp_rank < self.pp_size - 1:
            return hidden_states
        logits = self.logits_processor(self.model.embed_tokens.weight,
                                       hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        if self.pp_rank < self.pp_size - 1:
            return None
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        for name, loaded_weight in weights:
            if "normalizer" in name:
                continue
            for (param_name, shard_name, shard_id) in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # lm_head is not used in vllm as it is tied with embed_token.
                # To prevent errors, skip loading lm_head.weight.
                if "lm_head.weight" in name:
                    continue
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # GemmaRMSNorm is different from Llama's in that it multiplies
                # (1 + weight) to the output, instead of just weight.
                if "norm.weight" in name:
                    # For remote storage, we do not need to modify norm.weight
                    if int(os.getenv("CPU_ENV", "0")) == 0:
                        loaded_weight += 1.0
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            print(f"Warning: Some weights are not initialized from checkpoints: {unloaded_params}")

    def check_duplicate(self):
        if GemmaForCausalLM.prev_model is None:
            # it is the first time to load model, so store a ptr to current model
            GemmaForCausalLM.prev_model = self
        else:
            # we are loading dest model
            duplicate_state_dict_json = get_duplicate_state_dict()
            if len(duplicate_state_dict_json) > 0:
                prev_model = GemmaForCausalLM.prev_model
                for prev_name, after_name in duplicate_state_dict_json.items():
                    prev_layer_num, dest_layer_num = extract_dest_layer(prev_name, after_name)
                    if prev_layer_num == -1:
                        # not decoder layer
                        # may be embed, lm_head, final_norm, etc.
                        if prev_name != after_name:
                            print(f"prev = {prev_name}, after = {after_name}")
                            raise ValueError("Cannot extract param mapping.")
                        if 'embed' in prev_name:
                            GemmaForCausalLM.duplicate_embed_tokens = prev_model.model.embed_tokens
                        elif 'norm' in prev_name:
                            GemmaForCausalLM.duplicate_norm = prev_model.model.norm
                        else:
                            print(f"param name = {prev_name}")
                            raise ValueError(f"Error: unrecognized parameter.")
                    else:
                        # decoder layer
                        GemmaForCausalLM.duplicate_decoder_layer[dest_layer_num] = prev_model.model.layers[prev_layer_num]

    def set_socket(self, read_socket, write_socket):
        self.read_socket = read_socket
        self.write_socket = write_socket