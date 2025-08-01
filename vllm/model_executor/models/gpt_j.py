# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/gptj/modeling_gptj.py
# Copyright 2023 The vLLM team.
# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
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
"""Inference-only GPT-J model compatible with HuggingFace weights."""
from typing import Iterable, List, Optional, Tuple

import os
import torch
from torch import nn
from transformers import GPTJConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.distributed import get_tensor_model_parallel_world_size
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
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import send_states, recv_states, get_duplicate_state_dict, extract_dest_layer
from vllm.sequence import SamplerOutput


class GPTJAttention(nn.Module):

    def __init__(
        self,
        config: GPTJConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.total_num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.total_num_heads

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_size,
            self.total_num_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.out_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        scaling = self.head_size**-0.5
        assert getattr(config, "rotary", True)
        assert config.rotary_dim % 2 == 0
        rope_theta = getattr(config, "rope_theta", 10000)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.rotary_emb = get_rope(
            self.head_size,
            rotary_dim=config.rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            is_neox_style=False,
        )
        self.attn = Attention(self.num_heads, self.head_size, scaling)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q, k = self.rotary_emb(position_ids, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        attn_output, _ = self.out_proj(attn_output)
        return attn_output


class GPTJMLP(nn.Module):

    def __init__(
        self,
        intermediate_size: int,
        config: GPTJConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        hidden_size = config.n_embd
        self.fc_in = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            quant_config=quant_config,
        )
        self.fc_out = RowParallelLinear(
            intermediate_size,
            hidden_size,
            quant_config=quant_config,
        )
        self.act = get_act_fn(config.activation_function, quant_config,
                              intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.fc_out(hidden_states)
        return hidden_states


class GPTJBlock(nn.Module):

    def __init__(
        self,
        config: GPTJConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        inner_dim = (4 * config.n_embd
                     if config.n_inner is None else config.n_inner)
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPTJAttention(config, quant_config)
        self.mlp = GPTJMLP(inner_dim, config, quant_config)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        mlp_output = self.mlp(hidden_states)
        hidden_states = attn_output + mlp_output + residual
        return hidden_states


class GPTJModel(nn.Module):

    def __init__(
        self,
        config: GPTJConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.n_embd
        self.pp_rank = int(os.getenv('PP_RANK', '0'))
        self.pp_size = int(os.getenv('PP_SIZE', '1'))

        if self.pp_rank == 0:
            if GPTJForCausalLM.duplicate_embed_tokens is not None:
                self.wte = GPTJForCausalLM.duplicate_embed_tokens
            else:
                self.wte = VocabParallelEmbedding(
                    config.vocab_size,
                    self.embed_dim,
                )
        if len(GPTJForCausalLM.duplicate_decoder_layer) > 0:
            layer_list = []
            for i in range(config.num_hidden_layers):
                if i in GPTJForCausalLM.duplicate_decoder_layer:
                    layer_list.append(GPTJForCausalLM.duplicate_decoder_layer[i])
                else:
                    layer_list.append(GPTJBlock(config, quant_config))
            self.h = nn.ModuleList(layer_list)
        else:
            self.h = nn.ModuleList(
                [GPTJBlock(config, quant_config) for _ in range(config.n_layer)])
        if self.pp_rank == self.pp_size - 1:
            if GPTJForCausalLM.duplicate_norm is not None:
                self.ln_f = GPTJForCausalLM.duplicate_norm
            else:
                self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.wte(input_ids)
        for i in range(len(self.h)):
            layer = self.h[i]
            hidden_states = layer(
                position_ids,
                hidden_states,
                kv_caches[i],
                attn_metadata,
            )
        if self.pp_rank == self.pp_size - 1:
            hidden_states = self.ln_f(hidden_states)
        return hidden_states


class GPTJForCausalLM(nn.Module):

    # For duplicate state dict
    prev_model = None
    duplicate_embed_tokens = None
    duplicate_norm = None
    duplicate_lm_head = None
    duplicate_decoder_layer = {}

    def __init__(
        self,
        config: GPTJConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        assert not config.tie_word_embeddings
        self.transformer = GPTJModel(config, quant_config)
        self.pp_rank = self.transformer.pp_rank
        self.pp_size = self.transformer.pp_size
        if self.pp_rank == self.pp_size - 1:
            if GPTJForCausalLM.duplicate_lm_head is not None:
                self.lm_head = GPTJForCausalLM.duplicate_lm_head
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.n_embd,
                    bias=True,
                )
            self.logits_processor = LogitsProcessor(config.vocab_size)
            self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
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
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata, self.lm_head.bias)
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
        for name, loaded_weight in weights:
            if "attn.bias" in name or "attn.masked_bias" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
    
    def check_duplicate(self):
        if GPTJForCausalLM.prev_model is None:
            # it is the first time to load model, so store a ptr to current model
            GPTJForCausalLM.prev_model = self
        else:
            # we are loading dest model
            duplicate_state_dict_json = get_duplicate_state_dict()
            if len(duplicate_state_dict_json) > 0:
                prev_model = GPTJForCausalLM.prev_model
                for prev_name, after_name in duplicate_state_dict_json.items():
                    prev_layer_num, dest_layer_num = extract_dest_layer(prev_name, after_name)
                    if prev_layer_num == -1:
                        # not decoder layer
                        # may be embed, lm_head, final_norm, etc.
                        if prev_name != after_name:
                            print(f"prev = {prev_name}, after = {after_name}")
                            raise ValueError("Cannot extract param mapping.")
                        if 'lm_head' in prev_name:
                            GPTJForCausalLM.duplicate_lm_head = prev_model.lm_head
                        elif 'wte' in prev_name:
                            GPTJForCausalLM.duplicate_embed_tokens = prev_model.transformer.wte
                        elif 'ln_f' in prev_name:
                            GPTJForCausalLM.duplicate_norm = prev_model.transformer.ln_f
                        else:
                            print(f"param name = {prev_name}")
                            raise ValueError(f"Error: unrecognized parameter.")
                    else:
                        # decoder layer
                        GPTJForCausalLM.duplicate_decoder_layer[dest_layer_num] = prev_model.transformer.h[prev_layer_num]
    
    def set_socket(self, read_socket, write_socket):
        self.read_socket = read_socket
        self.write_socket = write_socket
