from __future__ import annotations

from pathlib import Path
from typing import Any

from harness_training.config import ModelConfig, TemplateConfig
from .base import BaseModel


class QwenModel(BaseModel):
    def __init__(self, config: ModelConfig, template_config: TemplateConfig):
        self.config, self.template_config = config, template_config
        self._processor = None
        self.model = None

    @property
    def processor(self) -> Any:
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(str(self.config.path), local_files_only=True)
            if not hasattr(self._processor, "tokenizer"):
                raise ValueError("QwenModel requires a multimodal Processor with a tokenizer")
            self._processor.tokenizer.padding_side = "right"
            image_processor = self._processor.image_processor
            if hasattr(image_processor, "size"):
                image_processor.size = {"shortest_edge": self.template_config.min_image_pixels,
                                        "longest_edge": self.template_config.max_image_pixels}
        return self._processor

    def load(self) -> Any:
        if self.model is not None:
            return self.model
        import torch
        from transformers import AutoConfig, AutoModelForImageTextToText

        model_config = AutoConfig.from_pretrained(str(self.config.path), local_files_only=True)
        if not model_config.model_type.startswith("qwen"):
            raise ValueError(f"Expected Qwen architecture, got {model_config.model_type}")
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(self.config.path), local_files_only=True, dtype=getattr(torch, self.config.dtype),
            attn_implementation=self.config.attention_implementation,
        )
        self.model.config.use_cache = False
        if self.config.tuning == "lora":
            from peft import LoraConfig, get_peft_model

            lora = self.config.lora
            self.model = get_peft_model(self.model, LoraConfig(
                r=lora.rank, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
                target_modules=lora.target_modules, task_type="CAUSAL_LM", bias="none",
            ))
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
