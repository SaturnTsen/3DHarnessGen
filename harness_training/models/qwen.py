from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, cast
from harness_training.types import Processor

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedModel

from harness_training.config import LongContextConfig, ModelConfig, TemplateConfig
from .base import BaseModel


class QwenModel(BaseModel):
    def __init__(self, config: ModelConfig, template_config: TemplateConfig,
                 long_context: LongContextConfig | None = None):
        self.config, self.template_config = config, template_config
        self.long_context = long_context
        self._processor: Processor | None = None
        self.model: PreTrainedModel | PeftModel | None = None

    def validate_runtime(self) -> None:
        if self.long_context is None:
            return
        from importlib.metadata import version

        if version("transformers") != "5.17.0":
            raise ValueError("Long-context Qwen classes require transformers 5.17.0")
        if self.long_context.require_optimized_kernels:
            from .kernels import validate_long_context_kernels

            validate_long_context_kernels()

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
        self.validate_runtime()
        import torch
        from transformers import AutoConfig, AutoModelForImageTextToText

        model_config = AutoConfig.from_pretrained(
            str(self.config.path), local_files_only=self.config.local_files_only,
            trust_remote_code=self.config.trust_remote_code)
        if not model_config.model_type.startswith("qwen"):
            raise ValueError(f"Expected Qwen architecture, got {model_config.model_type}")
        quantization = self.config.quantization
        loading = {}
        if quantization is not None:
            from transformers import BitsAndBytesConfig

            if not torch.cuda.is_available():
                raise ValueError("4-bit training requires a CUDA GPU")
            loading["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quantization.load_in_4bit,
                bnb_4bit_quant_type=quantization.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=quantization.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=getattr(torch, quantization.bnb_4bit_compute_dtype),
                bnb_4bit_quant_storage=getattr(torch, quantization.bnb_4bit_quant_storage),
            )
        model_class = AutoModelForImageTextToText
        if self.long_context is not None:
            from .qwen_long_context import LongContextQwenForConditionalGeneration

            if model_config.model_type != "qwen3_5":
                raise ValueError("long_context requires the qwen3_5 multimodal architecture")
            model_config.harness_long_context = self.long_context.model_dump(mode="json")
            model_class = LongContextQwenForConditionalGeneration
        self.model = model_class.from_pretrained(
            str(self.config.path), local_files_only=self.config.local_files_only,
            config=model_config,
            trust_remote_code=self.config.trust_remote_code, dtype=getattr(torch, self.config.dtype),
            attn_implementation=self.config.attention_implementation,
            device_map={"": torch.cuda.current_device()} if quantization is not None else self.config.device_map,
            **loading,
        )
        self.model.config.use_cache = self.config.use_cache
        if self.long_context is not None:
            self.model.config.get_text_config().update({"use_cache": False})
        if quantization is not None:
            from peft import prepare_model_for_kbit_training

            self.model = cast("PreTrainedModel", prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=quantization.use_gradient_checkpointing,
                gradient_checkpointing_kwargs=quantization.gradient_checkpointing_kwargs.model_dump(),
                auto_clear_cache=quantization.auto_clear_cache,
            ))
        if self.config.tuning == "lora":
            from peft import LoraConfig, LoraRuntimeConfig, PeftModel, get_peft_model

            lora = self.config.lora
            options = lora.backend.model_dump(mode="json")
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
        self.model.save_pretrained(str(path))
        self.processor.save_pretrained(path)
