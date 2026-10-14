from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, cast
from harness_training.types import JsonValue, Processor

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedModel

from harness_training.config import ModelConfig, TemplateConfig
from .base import BaseModel


class QwenModel(BaseModel):
    def __init__(self, config: ModelConfig, template_config: TemplateConfig):
        self.config, self.template_config = config, template_config
        self._processor: Processor | None = None
        self.model: PreTrainedModel | PeftModel | None = None

    @property
    def processor(self) -> Processor:
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = cast(Processor, AutoProcessor.from_pretrained(
                str(self.config.path), local_files_only=self.config.local_files_only,
                trust_remote_code=self.config.trust_remote_code))
            if not hasattr(self._processor, "tokenizer"):
                raise ValueError("QwenModel requires a multimodal Processor with a tokenizer")
            self._processor.tokenizer.padding_side = self.template_config.padding_side
            image_processor = self._processor.image_processor
            if hasattr(image_processor, "size"):
                image_processor.size = {"shortest_edge": self.template_config.min_image_pixels,
                                        "longest_edge": self.template_config.max_image_pixels}
        return self._processor

    def load(self) -> PreTrainedModel | PeftModel:
        if self.model is not None:
            return self.model
        import torch
        from transformers import AutoConfig, AutoModelForImageTextToText

        model_config = AutoConfig.from_pretrained(
            str(self.config.path), local_files_only=self.config.local_files_only,
            trust_remote_code=self.config.trust_remote_code)
        if not model_config.model_type.startswith("qwen"):
            raise ValueError(f"Expected Qwen architecture, got {model_config.model_type}")
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(self.config.path), local_files_only=self.config.local_files_only,
            trust_remote_code=self.config.trust_remote_code, dtype=getattr(torch, self.config.dtype),
            attn_implementation=self.config.attention_implementation,
        )
        self.model.config.use_cache = self.config.use_cache
        if self.config.tuning == "lora":
            from peft import LoraConfig, LoraRuntimeConfig, PeftModel, get_peft_model

            lora = self.config.lora
            options: dict[str, JsonValue | LoraRuntimeConfig] = lora.backend.model_dump(mode="json")
            options.update(r=lora.rank, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
                           target_modules=lora.target_modules, task_type=lora.task_type, bias=lora.bias)
            options["runtime_config"] = LoraRuntimeConfig(**lora.backend.runtime_config.model_dump())
            required = {field.name for field in fields(LoraConfig) if field.init}
            if options.keys() != required:
                raise ValueError(f"LoraConfig fields changed: missing={sorted(required - options.keys())}, "
                                 f"unknown={sorted(options.keys() - required)}")
            adapter = self.config.adapter
            self.model = cast(PeftModel, get_peft_model(
                self.model, LoraConfig(**options), adapter_name=adapter.adapter_name, mixed=adapter.mixed,
                autocast_adapter_dtype=adapter.autocast_adapter_dtype, revision=adapter.revision,
                low_cpu_mem_usage=adapter.low_cpu_mem_usage))
        if self.config.freeze_vision:
            for name, parameter in self.model.named_parameters():
                if any(part in {"visual", "vision_tower", "vision_model"} for part in name.split(".")):
                    parameter.requires_grad_(False)
        if not any(parameter.requires_grad for parameter in self.model.parameters()):
            raise ValueError("No trainable model parameters")
        return self.model

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Model weights have not been loaded")
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
