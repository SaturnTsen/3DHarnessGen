# Adapted from Transformers 5.17.0 modeling_qwen3_5.py.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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

"""Explicit Qwen3.5 subclasses for GPU-only long-context LoRA training.

Parameter paths match Transformers 5.17.0. No methods or upstream classes are
modified at runtime. The parent implementation remains the inference path.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3_5Config
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5DecoderLayer,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MLP,
    Qwen3_5Model,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5PreTrainedModel,
    Qwen3_5RMSNorm,
    Qwen3_5TextModel,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5VisionModel,
)
from transformers.utils.generic import can_return_tuple

from harness_training.config import LongContextConfig
from .chunked_loss import chunked_cross_entropy


class ActivationDtypeEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None,
                 activation_dtype: torch.dtype):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.activation_dtype = activation_dtype

    def forward(self, input):
        # PEFT keeps non-quantized weights in FP32. Keep the residual stream in
        # the configured compute dtype instead of saving FP32 layer boundaries.
        return super().forward(input).to(self.activation_dtype)


class ChunkedQwenMLP(Qwen3_5MLP):
    def __init__(self, config, chunk_size: int):
        super().__init__(config, config.intermediate_size)
        self.chunk_size = chunk_size

    def forward(self, x):
        hidden_states = x
        if hidden_states.shape[-2] <= self.chunk_size:
            return super().forward(hidden_states)
        parts = []
        for chunk in hidden_states.split(self.chunk_size, dim=-2):
            if self.training and torch.is_grad_enabled():
                part = checkpoint(super().forward, chunk, use_reentrant=False)
            else:
                part = super().forward(chunk)
            parts.append(part)
        return torch.cat(parts, dim=-2)


class LongContextQwenDecoderLayer(Qwen3_5DecoderLayer):
    def __init__(self, config, layer_idx: int, options: LongContextConfig):
        super().__init__(config, layer_idx)
        self.mlp = ChunkedQwenMLP(config, options.mlp_chunk_size)


class LongContextQwenTextModel(Qwen3_5TextModel):
    _no_split_modules = ["LongContextQwenDecoderLayer"]

    def __init__(self, config, options: LongContextConfig):
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.options = options
        self.embed_tokens = ActivationDtypeEmbedding(
            config.vocab_size, config.hidden_size, config.pad_token_id,
            getattr(torch, options.activation_dtype),
        )
        self.layers = nn.ModuleList([
            LongContextQwenDecoderLayer(config, index, options) for index in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.layer_types = cast(list[str], config.layer_types)
        self.post_init()

    def _layer_group(self, hidden_states, *, start, stop, position_embeddings, masks, position_ids, kwargs):
        # Calling modules normally preserves FSDP pre/post-forward hooks and the
        # inner per-layer checkpoints enabled through gradient_checkpointing_enable.
        for index in range(start, stop):
            hidden_states = self.layers[index](
                hidden_states, position_embeddings=position_embeddings,
                attention_mask=masks[self.layer_types[index]],
                position_ids=position_ids, past_key_values=None, use_cache=False, **kwargs,
            )
        return hidden_states

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, **kwargs: Any):
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(getattr(torch, self.options.activation_dtype))
        group_size = self.options.checkpoint_group_size
        if not (self.training and self.gradient_checkpointing and torch.is_grad_enabled() and group_size is not None):
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=cast(torch.FloatTensor | None, inputs_embeds),
                use_cache=use_cache, **kwargs,
            )
        if use_cache or past_key_values is not None:
            raise ValueError("Grouped training checkpoints cannot update a KV cache")
        if kwargs.pop("output_hidden_states", False) or kwargs.pop("output_attentions", False):
            raise ValueError("Grouped training does not retain intermediate hidden states or attentions")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
        text_positions = None
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_positions, position_ids = position_ids[0], position_ids[1:]
        masks = attention_mask
        if not isinstance(masks, dict):
            masks = {
                "full_attention": create_causal_mask(
                    config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                    past_key_values=None, position_ids=text_positions,
                ),
                "linear_attention": create_recurrent_attention_mask(
                    config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                    past_key_values=None, position_ids=text_positions,
                ),
            }
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for start in range(0, len(self.layers), group_size):
            # Bind loop values now: checkpoint invokes this callable again in backward.
            run_group = partial(
                self._layer_group, start=start, stop=min(start + group_size, len(self.layers)),
                position_embeddings=position_embeddings, masks=masks, position_ids=text_positions, kwargs=kwargs,
            )
            hidden_states = checkpoint(run_group, hidden_states, use_reentrant=False)
        return Qwen3_5ModelOutputWithPast(last_hidden_state=self.norm(hidden_states), past_key_values=None)


class LongContextQwenModel(Qwen3_5Model):
    # Vision is owned by the root FSDP unit. Microbatch counts may differ by rank;
    # wrapping vision blocks would introduce mismatched collectives and deadlock.
    _no_split_modules = ["LongContextQwenDecoderLayer"]

    def __init__(self, config, options: LongContextConfig):
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.options = options
        self.visual = Qwen3_5VisionModel(config.vision_config)
        self.language_model = LongContextQwenTextModel(config.text_config, options)
        self.rope_deltas = None
        self.post_init()

    @can_return_tuple
    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        if image_grid_thw is None:
            raise ValueError("Image microbatching requires image_grid_thw")
        # A video/one image is never split internally; its patch ordering is unchanged.
        patch_counts = image_grid_thw.prod(-1).tolist()
        if sum(patch_counts) != pixel_values.shape[0]:
            raise ValueError("Image grids do not match the pixel tensor")
        image_embeds = []
        offset = 0
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("output_attentions", None)
        with torch.no_grad():
            for start in range(0, len(patch_counts), self.options.vision_batch_size):
                stop = min(start + self.options.vision_batch_size, len(patch_counts))
                count = sum(patch_counts[start:stop])
                output = self.visual(
                    pixel_values[offset:offset + count].to(self.visual.dtype),
                    grid_thw=image_grid_thw[start:stop], return_dict=True, **kwargs,
                )
                sizes = [n // self.visual.spatial_merge_size**2 for n in patch_counts[start:stop]]
                image_embeds.extend(output.pooler_output.split(sizes))
                del output
                offset += count
        # Only the merged features are needed for language-model training.
        # Upstream also uses a tuple here, although its ModelOutput annotation
        # declares a single FloatTensor.
        return BaseModelOutputWithPooling(pooler_output=cast(torch.FloatTensor, tuple(image_embeds)))


@dataclass
class LongContextCausalLMOutput(Qwen3_5CausalLMOutputWithPast):
    num_correct_tokens: torch.Tensor | None = None
    entropy_sum: torch.Tensor | None = None
    num_valid_tokens: torch.Tensor | None = None


class LongContextQwenForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    _no_split_modules = ["LongContextQwenDecoderLayer"]
    accepts_loss_kwargs = True

    def __init__(self, config: Qwen3_5Config):
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.options = LongContextConfig.model_validate(config.harness_long_context)
        self.model = LongContextQwenModel(config, self.options)
        text_config = cast(Qwen3_5TextConfig, config.text_config)
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, pixel_values=None, pixel_values_videos=None,
                image_grid_thw=None, video_grid_thw=None, mm_token_type_ids=None,
                logits_to_keep: int | torch.Tensor = 0, num_items_in_batch=None, **kwargs: Any):
        if labels is None:
            return super().forward(
                input_ids=cast(torch.LongTensor, input_ids), attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds, pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos, image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw, mm_token_type_ids=mm_token_type_ids,
                logits_to_keep=logits_to_keep, **kwargs,
            )
        if not isinstance(logits_to_keep, int) or logits_to_keep != 0:
            raise ValueError("Target-only chunked training requires logits_to_keep: 0")
        kwargs.pop("use_cache", None)
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos, image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw, mm_token_type_ids=mm_token_type_ids, use_cache=False, **kwargs,
        )
        weight = self.lm_head.weight
        if isinstance(weight, DTensor):
            weight = weight.full_tensor()
        loss, correct, entropy, count = chunked_cross_entropy(
            outputs.last_hidden_state, weight, labels, self.options.loss_chunk_size, num_items_in_batch,
        )
        return LongContextCausalLMOutput(
            loss=cast(torch.FloatTensor, loss), logits=None, past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas, num_correct_tokens=correct,
            entropy_sum=entropy, num_valid_tokens=count,
        )
