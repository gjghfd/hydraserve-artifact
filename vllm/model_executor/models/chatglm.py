# coding=utf-8
# Adapted from
# https://github.com/THUDM/ChatGLM2-6B
"""Inference-only ChatGLM model compatible with THUDM weights."""
from typing import Iterable, List, Optional, Tuple
import time
import os

import torch
from torch import nn
from torch.nn import LayerNorm

from vllm.attention import Attention, AttentionMetadata
from vllm.config import LoRAConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
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
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import send_states, recv_states, get_duplicate_state_dict, extract_dest_layer
from vllm.sequence import SamplerOutput
from vllm.transformers_utils.configs import ChatGLMConfig


class GLMAttention(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.multi_query_attention = config.multi_query_attention
        self.total_num_kv_heads = (config.multi_query_group_num
                                   if config.multi_query_attention else
                                   config.num_attention_heads)
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.add_bias_linear or config.add_qkv_bias,
            quant_config=quant_config,
        )
        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.add_bias_linear,
            quant_config=quant_config,
        )

        # https://huggingface.co/THUDM/chatglm3-6b-32k/blob/e210410255278dd9d74463cf396ba559c0ef801c/modeling_chatglm.py#L141
        rope_ratio = getattr(config, "rope_ratio", 1.0)
        max_positions = getattr(config, "seq_length", 8192)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim // 2,
            max_position=max_positions,
            base=10000 * rope_ratio,
            is_neox_style=False,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(position_ids, q, k)
        context_layer = self.attn(
            q,
            k,
            v,
            kv_cache,
            attn_metadata,
        )
        attn_output, _ = self.dense(context_layer)
        return attn_output


class GLMMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        self.add_bias = config.add_bias_linear

        # Project to 4h.
        self.dense_h_to_4h = MergedColumnParallelLinear(
            config.hidden_size,
            [config.ffn_hidden_size] * 2,
            bias=config.add_bias_linear,
            quant_config=quant_config,
        )

        self.activation_func = SiluAndMul()

        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=config.add_bias_linear,
            quant_config=quant_config,
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel, _ = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output, _ = self.dense_4h_to_h(intermediate_parallel)
        return output


class GLMBlock(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm)

        self.fp32_residual_connection = config.fp32_residual_connection

        layer_norm_func = RMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = layer_norm_func(config.hidden_size,
                                               eps=config.layernorm_epsilon)

        # Self attention.
        self.self_attention = GLMAttention(config, quant_config)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = layer_norm_func(
            config.hidden_size, eps=config.layernorm_epsilon)

        # MLP
        self.mlp = GLMMLP(config, quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # hidden_states: [num_tokens, h]
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output = self.self_attention(
            hidden_states=layernorm_output,
            position_ids=position_ids,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = residual + attention_output

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = self.mlp(layernorm_output) + residual

        return output


class GLMTransformer(nn.Module):
    """Transformer class."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.pp_rank = int(os.getenv('PP_RANK', '0'))
        self.pp_size = int(os.getenv('PP_SIZE', '1'))

        self.post_layer_norm = config.post_layer_norm
        if self.pp_rank < self.pp_size - 1:
            self.post_layer_norm = False

        # Number of layers.
        self.num_layers = config.num_layers

        # Transformer layers.
        if len(ChatGLMForCausalLM.duplicate_decoder_layer) > 0:
            layer_list = []
            for i in range(config.num_hidden_layers):
                if i in ChatGLMForCausalLM.duplicate_decoder_layer:
                    layer_list.append(ChatGLMForCausalLM.duplicate_decoder_layer[i])
                else:
                    layer_list.append(GLMBlock(config, quant_config))
            self.layers = nn.ModuleList(layer_list)
        else:
            self.layers = nn.ModuleList(
                [GLMBlock(config, quant_config) for i in range(self.num_layers)])

        if self.post_layer_norm:
            layer_norm_func = RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            if ChatGLMForCausalLM.duplicate_norm is not None:
                self.final_layernorm = ChatGLMForCausalLM.duplicate_norm
            else:
                self.final_layernorm = layer_norm_func(
                    config.hidden_size, eps=config.layernorm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        for i in range(self.num_layers):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                kv_cache=kv_caches[i],
                attn_metadata=attn_metadata,
            )
        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class ChatGLMModel(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.pp_rank = int(os.getenv('PP_RANK', '0'))
        self.pp_size = int(os.getenv('PP_SIZE', '1'))

        if self.pp_rank == 0:
            if ChatGLMForCausalLM.duplicate_embed_tokens is not None:
                self.embedding = ChatGLMForCausalLM.duplicate_embed_tokens
            else:
                self.embedding = VocabParallelEmbedding(config.padded_vocab_size,
                                                        config.hidden_size)

        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels
        self.encoder = GLMTransformer(config, quant_config)

        if self.pp_rank == self.pp_size - 1:
            if ChatGLMForCausalLM.duplicate_lm_head is not None:
                self.output_layer = ChatGLMForCausalLM.duplicate_lm_head
            else:
                self.output_layer = ParallelLMHead(config.padded_vocab_size,
                                            config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        # Run encoder.
        hidden_states = self.encoder(
            hidden_states=inputs_embeds,
            position_ids=position_ids,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )
        return hidden_states


class ChatGLMForCausalLM(nn.Module):
    packed_modules_mapping = {
        "query_key_value": ["query_key_value"],
        "dense_h_to_4h": ["dense_h_to_4h"]
    }
    # LoRA specific attributes
    supported_lora_modules = [
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    # For duplicate state dict
    prev_model = None
    duplicate_embed_tokens = None
    duplicate_norm = None
    duplicate_lm_head = None
    duplicate_decoder_layer = {}

    def __init__(
        self,
        config: ChatGLMConfig,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ):
        super().__init__()
        self.check_duplicate()
        self.config: ChatGLMConfig = config
        self.quant_config = quant_config
        self.transformer = ChatGLMModel(config, quant_config)

        self.pp_rank = self.transformer.pp_rank
        self.pp_size = self.transformer.pp_size
        if self.pp_rank == self.pp_size - 1:
            self.lm_head_weight = self.transformer.output_layer.weight
            self.logits_processor = LogitsProcessor(config.padded_vocab_size)
            self.sampler = Sampler()

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
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_pos_emb.inv_freq" in name:
                continue
            if "word_embeddings" in name:
                name = name.replace(".word_embeddings", "")
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)

    def check_duplicate(self):
        if ChatGLMForCausalLM.prev_model is None:
            # it is the first time to load model, so store a ptr to current model
            ChatGLMForCausalLM.prev_model = self
        else:
            # we are loading dest model
            duplicate_state_dict_json = get_duplicate_state_dict()
            if len(duplicate_state_dict_json) > 0:
                prev_model = ChatGLMForCausalLM.prev_model
                for prev_name, after_name in duplicate_state_dict_json.items():
                    if "rotary_pos_emb.inv_freq" in prev_name:
                        continue
                    prev_layer_num, dest_layer_num = extract_dest_layer(prev_name, after_name)
                    if prev_layer_num == -1:
                        # not decoder layer
                        # may be embed, lm_head, final_norm, etc.
                        if prev_name != after_name:
                            print(f"prev = {prev_name}, after = {after_name}")
                            raise ValueError("Cannot extract param mapping.")
                        if 'output_layer' in prev_name:
                            ChatGLMForCausalLM.duplicate_lm_head = prev_model.transformer.output_layer
                        elif 'embed' in prev_name:
                            ChatGLMForCausalLM.duplicate_embed_tokens = prev_model.transformer.embedding
                        elif 'final_layernorm' in prev_name:
                            ChatGLMForCausalLM.duplicate_norm = prev_model.transformer.encoder.final_layernorm
                        else:
                            print(f"param name = {prev_name}")
                            raise ValueError(f"Error: unrecognized parameter.")
                    else:
                        # decoder layer
                        ChatGLMForCausalLM.duplicate_decoder_layer[dest_layer_num] = prev_model.transformer.encoder.layers[prev_layer_num]

    def set_socket(self, read_socket, write_socket):
        self.read_socket = read_socket
        self.write_socket = write_socket
