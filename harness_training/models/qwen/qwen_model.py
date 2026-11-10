from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, cast
from harness_training.types import Processor

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedModel

from harness_training.config import LongContextConfig, ModelConfig, TemplateConfig, read_yaml
from ..base import BaseModel


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
        if not self.long_context.require_optimized_kernels:
            return
        import torch
        from causal_conv1d import causal_conv1d_fn
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        from transformers.models.qwen3_5 import modeling_qwen3_5

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("Optimized long-context training requires a BF16-capable CUDA GPU")
        if not torch.backends.cuda.is_flash_attention_available():
            raise ValueError("This PyTorch build does not include CUDA Flash SDPA support")
        # Transformers captures the selected implementation when its module imports.
        # Checking imports alone would miss an earlier, cached fallback selection.
        for wrapper, expected in [
            (modeling_qwen3_5.causal_conv1d_fn, causal_conv1d_fn),
            (modeling_qwen3_5.torch_chunk_gated_delta_rule, chunk_gated_delta_rule),
        ]:
            current = wrapper
            while inspect.isfunction(current):
                selected = inspect.getclosurevars(current).nonlocals.get("implementation")
                if selected is not None:
                    if selected is not expected:
                        raise ValueError(f"{wrapper.__name__} selected a fallback instead of its native kernel; "
                                         "restart the process after installing dependencies")
                    break
                current = getattr(current, "__wrapped__", None)
            else:
                raise ValueError(f"Cannot verify the optimized dispatch for {wrapper.__name__}")

    def _validate_initialization(self) -> None:
        source = self.config.init_from
        if source is None:
            return
        if not source.is_dir():
            raise ValueError("model.init_from must be an existing checkpoint or adapter directory")
        if (source / "pytorch_model_fsdp_0/.metadata").is_file():
            run_dir = source.parent.parent if source.parent.name == "checkpoints" else source.parent
            snapshot = run_dir / "resolved_config.yaml"
            if not snapshot.is_file():
                raise ValueError("FSDP model.init_from requires the source run's resolved_config.yaml")
            prior = read_yaml(snapshot)["model"]
            current = self.config.model_dump(mode="json")
            if any(prior[key] != current[key] for key in ("name", "path", "tuning", "lora", "adapter")):
                raise ValueError("model.init_from base model, LoRA and adapter settings must match the source run")
        elif (source / "adapter_config.json").is_file():
            from peft import LoraConfig

            prior = LoraConfig.from_pretrained(str(source)).to_dict()
            base_path = prior["base_model_name_or_path"]
            if not base_path or Path(base_path).expanduser().resolve() != self.config.path.resolve():
                raise ValueError("model.init_from base model must match model.path")
            current = self.config.lora.to_peft().to_dict()
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
                self.model, self.config.lora.to_peft(), adapter_name=adapter.adapter_name, mixed=adapter.mixed,
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
