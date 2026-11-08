"""Translate the explicit project LoRA settings into PEFT arguments."""

from dataclasses import fields
from typing import TYPE_CHECKING

from harness_training.config import LoRAConfig

if TYPE_CHECKING:
    from peft import LoraConfig


def build_lora_config(config: LoRAConfig) -> "LoraConfig":
    from peft import LoraConfig, LoraRuntimeConfig

    options = config.backend.model_dump(mode="json")
    options.update(r=config.rank, lora_alpha=config.alpha, lora_dropout=config.dropout,
                   target_modules=config.target_modules, task_type=config.task_type, bias=config.bias)
    options["runtime_config"] = LoraRuntimeConfig(**config.backend.runtime_config.model_dump())
    required = {field.name for field in fields(LoraConfig) if field.init}
    if options.keys() != required:
        raise ValueError(f"LoraConfig fields changed: missing={sorted(required - options.keys())}, "
                         f"unknown={sorted(options.keys() - required)}")
    return LoraConfig(**options)
