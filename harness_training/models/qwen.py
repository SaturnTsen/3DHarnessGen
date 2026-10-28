from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, cast
from harness_training.types import Processor

if TYPE_CHECKING:
    from peft import LoraConfig, PeftModel
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
        self._validate_initialization()
        if self.long_context is None:
            return
        from importlib.metadata import version

        if version("transformers") != "5.17.0":
            raise ValueError("Long-context Qwen classes require transformers 5.17.0")
        if self.long_context.require_optimized_kernels:
            from .kernels import validate_long_context_kernels

            validate_long_context_kernels()

    def _lora_config(self) -> LoraConfig:
        from peft import LoraConfig, LoraRuntimeConfig

        lora = self.config.lora
        options = lora.backend.model_dump(mode="json")
        options.update(r=lora.rank, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
                       target_modules=lora.target_modules, task_type=lora.task_type, bias=lora.bias)
        options["runtime_config"] = LoraRuntimeConfig(**lora.backend.runtime_config.model_dump())
        required = {field.name for field in fields(LoraConfig) if field.init}
        if options.keys() != required:
            raise ValueError(f"LoraConfig fields changed: missing={sorted(required - options.keys())}, "
                             f"unknown={sorted(options.keys() - required)}")
        return LoraConfig(**options)

    def _validate_initialization(self) -> None:
        source = self.config.init_from
        if source is None:
            return
        if not source.is_dir():
            raise ValueError("model.init_from must be an existing checkpoint or adapter directory")
        if (source / "pytorch_model_fsdp_0/.metadata").is_file():
            from pydantic import RootModel
            from pydantic_yaml import parse_yaml_raw_as
            from harness_training.types import JsonObject

            snapshot = source.parent.parent / "resolved_config.yaml"
            if not snapshot.is_file():
                raise ValueError("FSDP model.init_from requires the source run's resolved_config.yaml")
            prior = cast(JsonObject, parse_yaml_raw_as(RootModel[JsonObject], snapshot.read_text()).root["model"])
            current = self.config.model_dump(mode="json")
            if any(prior[key] != current[key] for key in ("name", "path", "tuning", "lora", "adapter")):
                raise ValueError("model.init_from base model, LoRA and adapter settings must match the source run")
        elif (source / "adapter_config.json").is_file():
            from peft import LoraConfig

            prior = LoraConfig.from_pretrained(str(source)).to_dict()
            base_path = prior["base_model_name_or_path"]
            if not base_path or Path(base_path).expanduser().resolve() != self.config.path.resolve():
                raise ValueError("model.init_from base model must match model.path")
            current = self._lora_config().to_dict()
            ignored = {"base_model_name_or_path", "revision", "inference_mode", "peft_version", "auto_mapping"}
            if any(prior[key] != value for key, value in current.items() if key not in ignored):
                raise ValueError("model.init_from LoRA settings must match the saved adapter")
            if not any((source / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
                raise ValueError("model.init_from has no saved adapter weights")
        else:
            raise ValueError("model.init_from requires adapter weights or an FSDP sharded checkpoint")

    def _load_initial_weights(self) -> None:
        source = self.config.init_from
        if source is None:
            return
        from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict

        model = cast(PeftModel, self.model)
        adapter = self.config.adapter
        if (source / "pytorch_model_fsdp_0/.metadata").is_file():
            from torch.distributed.checkpoint.filesystem import FileSystemReader
            from torch.distributed.checkpoint.metadata import TensorStorageMetadata
            from torch.distributed.checkpoint.state_dict_loader import load

            # Read only trainable tensors; frozen 27B/quantized base weights stay on disk.
            state = get_peft_model_state_dict(model, adapter_name=adapter.adapter_name)
            weights = {name: value.detach().cpu().clone() for name, value in state.items()}
            metadata = FileSystemReader(source / "pytorch_model_fsdp_0").read_metadata()
            for name, value in weights.items():
                saved = metadata.state_dict_metadata.get(f"model.{name}")
                if not isinstance(saved, TensorStorageMetadata) or saved.size != value.shape:
                    raise ValueError(f"model.init_from missing or incompatible trainable tensor: {name}")
            load({"model": weights}, checkpoint_id=source / "pytorch_model_fsdp_0", no_dist=True)
            set_peft_model_state_dict(model, weights, adapter_name=adapter.adapter_name,
                                      low_cpu_mem_usage=adapter.low_cpu_mem_usage)
        else:
            from peft.utils.save_and_load import load_peft_weights

            weights = load_peft_weights(str(source), device="cpu", local_files_only=True)
            expected = get_peft_model_state_dict(model, adapter_name=adapter.adapter_name)
            if weights.keys() != expected.keys():
                raise ValueError("model.init_from adapter weight keys do not match the configured model")
            set_peft_model_state_dict(model, weights, adapter_name=adapter.adapter_name,
                                      low_cpu_mem_usage=adapter.low_cpu_mem_usage)

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
            from peft import PeftModel, get_peft_model

            adapter = self.config.adapter
            self.model = cast(PeftModel, get_peft_model(
                self.model, self._lora_config(), adapter_name=adapter.adapter_name, mixed=adapter.mixed,
                autocast_adapter_dtype=adapter.autocast_adapter_dtype, revision=adapter.revision,
                low_cpu_mem_usage=adapter.low_cpu_mem_usage))
        if self.config.freeze_vision:
            for name, parameter in self.model.named_parameters():
                if any(part in {"visual", "vision_tower", "vision_model"} for part in name.split(".")):
                    parameter.requires_grad_(False)
        if not any(parameter.requires_grad for parameter in self.model.parameters()):
            raise ValueError("No trainable model parameters")
        self._load_initial_weights()
        return self.model

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Model weights have not been loaded")
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.processor.save_pretrained(path)
