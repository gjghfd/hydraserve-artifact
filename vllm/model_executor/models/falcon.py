# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/a5cc30d72ae2dc19af534e4b35c986cc28db1275/src/transformers/models/falcon/modeling_falcon.py
# Copyright 2023 The vLLM team.
# Copyright 2023 the Falcon authors and HuggingFace Inc. team.  All rights
# reserved.
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
"""PyTorch Falcon model."""

import os
import math
from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import LayerNorm
from transformers import FalconConfig as HF_FalconConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
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
from vllm.transformers_utils.configs import RWConfig

FalconConfig = Union[HF_FalconConfig, RWConfig]


def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
    closest_power_of_2 = 2**math.floor(math.log2(total_num_heads))
    base = torch.tensor(2**(-(2**-(math.log2(closest_power_of_2) - 3))),
                        dtype=torch.float32)
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != total_num_heads:
        extra_base = torch.tensor(
            2**(-(2**-(math.log2(2 * closest_power_of_2) - 3))),
            dtype=torch.float32)
        num_remaining_heads = min(closest_power_of_2,
                                  total_num_heads - closest_power_of_2)
        extra_powers = torch.arange(1,
                                    1 + 2 * num_remaining_heads,
                                    2,
                                    dtype=torch.int32)
        slopes = torch.cat(
            [slopes, torch.pow(extra_base, extra_powers)], dim=0)

    return slopes


class FalconAttention(nn.Module):

    def __init__(
        self,
        config: FalconConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.head_dim = self.hidden_size // self.total_num_heads
        assert self.head_dim * self.total_num_heads == self.hidden_size

        self.new_decoder_architecture = config.new_decoder_architecture
        self.multi_query = config.multi_query

        if self.new_decoder_architecture:
            self.total_num_kv_heads = config.num_kv_heads
        elif self.multi_query:
            self.total_num_kv_heads = 1
        else:
            self.total_num_kv_heads = self.total_num_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.bias,
            skip_bias_add=True,
            quant_config=quant_config,
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.reduce_row_parallel_results = not (config.new_decoder_architecture
                                                or config.parallel_attn)
        self.dense = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=config.bias,
            skip_bias_add=True,
            quant_config=quant_config,
            reduce_results=self.reduce_row_parallel_results)

        self.use_rotary = config.rotary
        self.use_alibi = config.alibi
        assert not (self.use_rotary and self.use_alibi), (
            "Rotary and alibi are mutually exclusive.")

        if self.use_rotary:
            rope_theta = getattr(config, "rope_theta", 10000)
            max_position_embeddings = getattr(config,
                                              "max_position_embeddings", 8192)
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
            )
            self.attn = Attention(self.num_heads,
                                  self.head_dim,
                                  self.inv_norm_factor,
                                  num_kv_heads=self.num_kv_heads)
        elif self.use_alibi:
            tp_rank = get_tensor_model_parallel_rank()
            head_start = tp_rank * self.num_heads
            head_end = (tp_rank + 1) * self.num_heads
            alibi_slopes = (_get_alibi_slopes(self.total_num_heads) *
                            self.inv_norm_factor)
            alibi_slopes = alibi_slopes[head_start:head_end].tolist()
            self.attn = Attention(self.num_heads,
                                  self.head_dim,
                                  self.inv_norm_factor,
                                  num_kv_heads=self.num_kv_heads,
                                  alibi_slopes=alibi_slopes)
        else:
            self.attn = Attention(self.num_heads,
                                  self.head_dim,
                                  scale=self.inv_norm_factor,
                                  num_kv_heads=self.num_kv_heads)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, bias = self.query_key_value(hidden_states)
        if bias is not None:
            qkv += bias
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.use_rotary:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        attn_output, bias = self.dense(attn_output)
        return attn_output, bias


class FalconMLP(nn.Module):

    def __init__(
        self,
        config: FalconConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        hidden_size = config.hidden_size

        self.dense_h_to_4h = ColumnParallelLinear(hidden_size,
                                                  4 * hidden_size,
                                                  bias=config.bias,
                                                  skip_bias_add=True,
                                                  quant_config=quant_config)
        self.act = get_act_fn("gelu", quant_config, 4 * hidden_size)
        self.reduce_row_parallel_results = not (config.new_decoder_architecture
                                                or config.parallel_attn)
        self.dense_4h_to_h = RowParallelLinear(
            4 * hidden_size,
            hidden_size,
            bias=config.bias,
            skip_bias_add=True,
            reduce_results=self.reduce_row_parallel_results,
            quant_config=quant_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE(zhuohan): Following huggingface, we do not fuse bias add here.
        x, bias = self.dense_h_to_4h(x)
        if bias is not None:
            x += bias
        x = self.act(x)
        x, bias = self.dense_4h_to_h(x)
        return x, bias


class FalconDecoderLayer(nn.Module):

    def __init__(
        self,
        config: FalconConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.self_attention = FalconAttention(config, quant_config)
        self.mlp = FalconMLP(config, quant_config)
        self.config = config

        if config.new_decoder_architecture:
            # The layer norm before self-attention
            self.ln_attn = LayerNorm(hidden_size,
                                     eps=config.layer_norm_epsilon)
            # The layer norm before the MLP
            self.ln_mlp = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        else:
            self.input_layernorm = LayerNorm(hidden_size,
                                             eps=config.layer_norm_epsilon)
            if not config.parallel_attn:
                self.post_attention_layernorm = LayerNorm(
                    hidden_size, eps=config.layer_norm_epsilon)

        self.reduce_row_parallel_results = not (config.new_decoder_architecture
                                                or config.parallel_attn)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        residual = hidden_states

        if self.config.new_decoder_architecture:
            attention_layernorm_out = self.ln_attn(hidden_states)
            mlp_layernorm_out = self.ln_mlp(hidden_states)
        else:
            attention_layernorm_out = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output, attention_bias = self.self_attention(
            positions=positions,
            hidden_states=attention_layernorm_out,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        if self.reduce_row_parallel_results and attention_bias is not None:
            attention_output += attention_bias

        if not self.config.new_decoder_architecture:
            if self.config.parallel_attn:
                mlp_layernorm_out = attention_layernorm_out
            else:
                residual += attention_output
                mlp_layernorm_out = self.post_attention_layernorm(residual)

        # MLP.
        mlp_output, mlp_bias = self.mlp(mlp_layernorm_out)
        if self.reduce_row_parallel_results and mlp_bias is not None:
            mlp_output += mlp_bias

        if not self.reduce_row_parallel_results:
            # When MLP and Attention layers are parallel, we can use
            # only one all-reduce operator to reduce the results from
            # both MLP and Attention layers.
            mlp_output += attention_output
            mlp_output = tensor_model_parallel_all_reduce(mlp_output)
            if attention_bias is not None:
                mlp_output += attention_bias
            if mlp_bias is not None:
                mlp_output += mlp_bias

        output = mlp_output + residual
        return output


class FalconModel(nn.Module):

    def __init__(
        self,
        config: FalconConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_alibi = config.alibi
        self.pp_rank = int(os.getenv('PP_RANK', '0'))
        self.pp_size = int(os.getenv('PP_SIZE', '1'))

        # Embedding + LN Embedding
        if self.pp_rank == 0 or self.pp_rank == self.pp_size - 1:
            if FalconForCausalLM.duplicate_embed_tokens is not None:
                self.word_embeddings = FalconForCausalLM.duplicate_embed_tokens
            else:
                self.word_embeddings = VocabParallelEmbedding(
                    config.vocab_size,
                    self.embed_dim,
                )

        # Transformer blocks
        if len(FalconForCausalLM.duplicate_decoder_layer) > 0:
            layer_list = []
            for i in range(config.num_hidden_layers):
                if i in FalconForCausalLM.duplicate_decoder_layer:
                    layer_list.append(FalconForCausalLM.duplicate_decoder_layer[i])
                else:
                    layer_list.append(FalconDecoderLayer(config, quant_config))
            self.h = nn.ModuleList(layer_list)
        else:
            self.h = nn.ModuleList([
                FalconDecoderLayer(config, quant_config)
                for _ in range(config.num_hidden_layers)
            ])

        # Final Layer Norm
        if self.pp_rank == self.pp_size - 1:
            if FalconForCausalLM.duplicate_norm is not None:
                self.ln_f = FalconForCausalLM.duplicate_norm
            else:
                self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.word_embeddings(input_ids)
        for i in range(len(self.h)):
            layer = self.h[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
            )
        if self.pp_rank == self.pp_size - 1:
            hidden_states = self.ln_f(hidden_states)
        return hidden_states


class FalconForCausalLM(nn.Module):

    # For duplicate state dict
    prev_model = None
    duplicate_embed_tokens = None
    duplicate_norm = None
    duplicate_decoder_layer = {}

    def __init__(
        self,
        config: FalconConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.transformer = FalconModel(config, quant_config)
        self.pp_rank = self.transformer.pp_rank
        self.pp_size = self.transformer.pp_size
        if self.pp_rank == self.pp_size - 1:
            self.lm_head_weight = self.transformer.word_embeddings.weight
            self.logits_processor = LogitsProcessor(config.vocab_size)
            self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if self.pp_rank > 0:
            device = input_ids.device
            shape = [input_ids.shape[0], self.config.hidden_size]
            hidden_states = recv_states(self.read_socket, self.config.torch_dtype, device, shape)

            hidden_states = self.transformer(input_ids, positions, kv_caches,
                                   attn_metadata, inputs_embeds=hidden_states)
        else:
            hidden_states = self.transformer(input_ids, positions, kv_caches,
                                   attn_metadata)
        if self.pp_rank < self.pp_size - 1:
            send_states(self.write_socket, hidden_states)

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        if self.pp_rank < self.pp_size - 1:
            return hidden_states
        logits = self.logits_processor(self.lm_head_weight, hidden_states,
                                       sampling_metadata)
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
        total_num_heads = self.config.num_attention_heads
        if self.config.new_decoder_architecture:
            total_num_kv_heads = self.config.num_kv_heads
        elif self.config.multi_query:
            total_num_kv_heads = 1
        else:
            total_num_kv_heads = total_num_heads
        num_query_heads_per_kv_head = total_num_heads // total_num_kv_heads
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if name == "lm_head.weight":
                # Falcon uses tied embeddings.
                continue
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict[name]
            if "query_key_value" in name:
                output_dim = getattr(param, "output_dim", None)
                loaded_weight_shape = loaded_weight.shape
                if output_dim is not None:
                    loaded_weight = loaded_weight.view(
                        loaded_weight_shape[:output_dim] +
                        (total_num_kv_heads, num_query_heads_per_kv_head + 2,
                         -1) + loaded_weight_shape[output_dim + 1:])
                    wq = loaded_weight.narrow(
                        output_dim + 1, 0,
                        num_query_heads_per_kv_head).reshape(
                            *loaded_weight_shape[:output_dim], -1,
                            *loaded_weight_shape[output_dim + 1:])
                    wk = loaded_weight.narrow(
                        output_dim + 1, num_query_heads_per_kv_head,
                        1).reshape(*loaded_weight_shape[:output_dim], -1,
                                   *loaded_weight_shape[output_dim + 1:])
                    wv = loaded_weight.narrow(
                        output_dim + 1, num_query_heads_per_kv_head + 1,
                        1).reshape(*loaded_weight_shape[:output_dim], -1,
                                   *loaded_weight_shape[output_dim + 1:])
                    loaded_weight = torch.cat([wq, wk, wv], dim=output_dim)

            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
    
    def check_duplicate(self):
        if FalconForCausalLM.prev_model is None:
            # it is the first time to load model, so store a ptr to current model
            FalconForCausalLM.prev_model = self
        else:
            # we are loading dest model
            duplicate_state_dict_json = get_duplicate_state_dict()
            if len(duplicate_state_dict_json) > 0:
                prev_model = FalconForCausalLM.prev_model
                for prev_name, after_name in duplicate_state_dict_json.items():
                    prev_layer_num, dest_layer_num = extract_dest_layer(prev_name, after_name)
                    if prev_layer_num == -1:
                        # not decoder layer
                        # may be embed, lm_head, final_norm, etc.
                        if prev_name != after_name:
                            print(f"prev = {prev_name}, after = {after_name}")
                            raise ValueError("Cannot extract param mapping.")
                        if 'embed' in prev_name:
                            FalconForCausalLM.duplicate_embed_tokens = prev_model.transformer.word_embeddings
                        elif 'ln_f' in prev_name:
                            FalconForCausalLM.duplicate_norm = prev_model.transformer.ln_f
                        else:
                            print(f"param name = {prev_name}")
                            raise ValueError(f"Error: unrecognized parameter.")
                    else:
                        # decoder layer
                        FalconForCausalLM.duplicate_decoder_layer[dest_layer_num] = prev_model.transformer.h[prev_layer_num]

    def set_socket(self, read_socket, write_socket):
        self.read_socket = read_socket
        self.write_socket = write_socket
