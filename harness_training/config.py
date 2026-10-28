from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_yaml import parse_yaml_raw_as, to_yaml_str

from harness_training.types import JsonObject

Scene = Literal["Single-view", "Multi-view", "ActiveVisual", "Full3DInteraction"]


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(StrictConfig):
    name: str
    scene: Scene | Annotated[list[Scene], Field(min_length=1)]
    seed: int
    output_dir: Path

    @model_validator(mode="after")
    def unique_scenes(self):
        if isinstance(self.scene, list) and len(set(self.scene)) != len(self.scene):
            raise ValueError("scene list must contain unique scenes")
        return self


class WandbSettingsConfig(StrictConfig):
    disable_code: Literal[True]
    disable_git: Literal[True]
    console: Literal["off"]
    quiet: bool
    x_disable_stats: bool
    init_timeout: float = Field(gt=0)


class TrackingConfig(StrictConfig):
    mode: Literal["disabled", "offline", "online"]
    project: str = Field(min_length=1)
    entity: str | None
    name: str = Field(min_length=1)
    group: str | None
    tags: list[str]
    job_type: str
    run_id: str | None = Field(pattern=r"^[A-Za-z0-9_-]+$")
    new_run_resume: Literal["never"]
    checkpoint_resume: Literal["must"]
    log_model: Literal["false"]
    watch: Literal["false"]
    settings: WandbSettingsConfig


class DataConfig(StrictConfig):
    schema_version: Literal[3]
    files: list[Path] | None
    image_root: Path | None
    runs_root: Path
    source_model: str
    instances: list[str] | None
    validation_fraction: float = Field(ge=0, lt=1)
    test_fraction: float = Field(ge=0, lt=1)
    missing_images: Literal["skip_sample", "error"]
    verify_images: bool
    accepted_statuses: list[Literal["OK", "ERROR", "UNKNOWN"]]
    include_final_reply: bool

    @model_validator(mode="after")
    def valid_split(self):
        if self.files is not None:
            if not self.files or len(set(self.files)) != len(self.files):
                raise ValueError("files must be nonempty and unique when specified")
            if self.instances is not None:
                raise ValueError("Explicit files select objects from metadata; instances must be null")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation_fraction + test_fraction must be less than 1")
        if self.instances is not None and (not self.instances or len(set(self.instances)) != len(self.instances)):
            raise ValueError("instances must be nonempty and unique when specified")
        for name in [self.source_model, *(self.instances or [])]:
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError("source_model and instances must be directory names")
        return self


class TemplateConfig(StrictConfig):
    name: Literal["qwen"]
    enable_thinking: bool
    custom_tools: dict[Annotated[str, Field(min_length=1)], Annotated[str, Field(min_length=1)]]
    max_length: int = Field(ge=16)
    min_image_pixels: int = Field(ge=1024)
    max_image_pixels: int = Field(ge=1024)
    padding_side: Literal["right"]
    tokenize: Literal[False]
    prompt_add_generation_prompt: Literal[True]
    target_add_generation_prompt: Literal[False]
    image_mode: Literal["RGB"]
    return_tensors: Literal["pt"]
    add_special_tokens: Literal[False]
    padding: Literal[False]
    truncation: Literal[False]
    label_ignore_index: Literal[-100]
    attention_padding_value: Literal[0]
    token_type_padding_value: Literal[0]

    @model_validator(mode="after")
    def image_budget(self):
        if self.min_image_pixels > self.max_image_pixels:
            raise ValueError("min_image_pixels cannot exceed max_image_pixels")
        return self


class LoRARuntimeConfig(StrictConfig):
    ephemeral_gpu_offload: bool


class LoRABackendConfig(StrictConfig):
    peft_type: Literal["LORA"]
    auto_mapping: JsonObject | None
    peft_version: str | None
    base_model_name_or_path: str | None
    revision: str | None
    inference_mode: Literal[False]
    exclude_modules: list[str] | str | None
    fan_in_fan_out: bool
    use_rslora: bool
    modules_to_save: list[str] | None
    init_lora_weights: bool | str
    layers_to_transform: list[int] | int | None
    layers_pattern: list[str] | str | None
    rank_pattern: dict[str, int]
    alpha_pattern: dict[str, int]
    megatron_config: JsonObject | None
    megatron_core: str | None
    trainable_token_indices: list[int] | dict[str, list[int]] | None
    loftq_config: JsonObject
    eva_config: Literal[None]
    corda_config: Literal[None]
    lora_ga_config: Literal[None]
    use_dora: bool
    velora_config: Literal[None]
    alora_invocation_tokens: list[int] | None
    use_qalora: bool
    qalora_group_size: int
    monteclora_config: Literal[None]
    layer_replication: list[tuple[int, int]] | None
    runtime_config: LoRARuntimeConfig
    lora_bias: bool
    target_parameters: list[str] | None
    use_bdlora: Literal[None]
    arrow_config: Literal[None]
    kasa_config: Literal[None]
    ensure_weight_tying: bool


class AdapterConfig(StrictConfig):
    adapter_name: str
    mixed: Literal[False]
    autocast_adapter_dtype: bool
    revision: str | None
    low_cpu_mem_usage: bool


class LoRAConfig(StrictConfig):
    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)
    target_modules: list[str]
    task_type: Literal["CAUSAL_LM"]
    bias: Literal["none", "all", "lora_only"]
    backend: LoRABackendConfig


class LongContextConfig(StrictConfig):
    require_optimized_kernels: bool
    activation_dtype: Literal["bfloat16", "float16", "float32"]
    checkpoint_group_size: int | None = Field(ge=2)
    mlp_chunk_size: int = Field(gt=0)
    vision_batch_size: int = Field(gt=0)
    loss_chunk_size: int = Field(gt=0)


class ModelConfig(StrictConfig):
    name: str = Field(min_length=1)
    path: Path
    max_length: int = Field(ge=16)
    dtype: Literal["bfloat16", "float16", "float32"]
    tuning: Literal["lora", "full"]
    freeze_vision: bool
    attention_implementation: Literal["sdpa", "eager", "flash_attention_2"]
    device_map: Literal["auto", "balanced", "balanced_low_0", "sequential"] | None
    lora: LoRAConfig
    adapter: AdapterConfig
    local_files_only: bool
    trust_remote_code: bool
    use_cache: Literal[False]
    quantization: QuantizationConfig | None


class QuantizationConfig(StrictConfig):
    load_in_4bit: Literal[True]
    bnb_4bit_quant_type: Literal["nf4", "fp4"]
    bnb_4bit_use_double_quant: bool
    bnb_4bit_compute_dtype: Literal["bfloat16", "float16", "float32"]
    bnb_4bit_quant_storage: Literal["bfloat16", "float16", "float32"]
    use_gradient_checkpointing: bool
    gradient_checkpointing_kwargs: CheckpointingConfig
    auto_clear_cache: bool


class DistributedConfig(StrictConfig):
    strategy: Literal["single", "ddp", "fsdp"]


class FSDPConfig(StrictConfig):
    version: Literal[2]
    auto_wrap_policy: Literal["TRANSFORMER_BASED_WRAP"]
    transformer_layer_cls_to_wrap: list[str] | None
    min_num_params: Literal[0]
    state_dict_type: Literal["SHARDED_STATE_DICT", "FULL_STATE_DICT"]
    cpu_ram_efficient_loading: bool
    cpu_offload: bool
    activation_checkpointing: Literal[False]
    reshard_after_forward: bool


class CheckpointingConfig(StrictConfig):
    use_reentrant: bool


class DatasetPreparationConfig(StrictConfig):
    skip_prepare_dataset: Literal[True]


class SFTBackendConfig(StrictConfig):
    lr_scheduler_type: str
    lr_scheduler_kwargs: JsonObject | str | None
    optim: str
    optim_args: str | None
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    optim_target_modules: None | str | list[str]
    average_tokens_across_devices: bool
    max_grad_norm: float
    label_smoothing_factor: float
    bf16: bool
    fp16: bool
    bf16_full_eval: bool
    fp16_full_eval: bool
    tf32: bool
    gradient_checkpointing_kwargs: CheckpointingConfig
    torch_compile: bool
    torch_compile_backend: str | None
    torch_compile_mode: str | None
    use_liger_kernel: bool
    liger_kernel_config: dict[str, bool] | None
    use_cache: Literal[False]
    neftune_noise_alpha: float | None
    torch_empty_cache_steps: int | None
    auto_find_batch_size: bool
    logging_strategy: str
    logging_first_step: bool
    log_on_each_node: bool
    logging_nan_inf_filter: bool
    include_num_input_tokens_seen: str | bool
    log_level: str
    log_level_replica: str
    disable_tqdm: bool
    report_to: None | str | list[str]
    run_name: str | None
    project: str
    trackio_space_id: str | None
    trackio_bucket_id: str | None
    trackio_static_space_id: str | None | Literal[False]
    eval_strategy: Literal["no", "steps", "epoch"]
    eval_delay: float
    per_device_eval_batch_size: int
    prediction_loss_only: bool
    eval_on_start: bool
    eval_do_concat_batches: bool
    eval_use_gather_object: bool
    eval_accumulation_steps: int | None
    include_for_metrics: list[str]
    batch_eval_metrics: bool
    save_only_model: bool
    save_strategy: str
    save_on_each_node: bool
    enable_jit_checkpoint: bool
    push_to_hub: bool
    hub_token: str | None
    hub_private_repo: bool | None
    hub_model_id: str | None
    hub_strategy: str
    hub_always_push: bool
    hub_revision: str | None
    load_best_model_at_end: bool
    metric_for_best_model: str | None
    greater_is_better: bool | None
    ignore_data_skip: bool
    restore_callback_states_from_checkpoint: bool
    full_determinism: bool
    data_seed: int
    accelerator_config: JsonObject | str | None
    parallelism_config: Literal[None]
    dataloader_drop_last: bool
    dataloader_pin_memory: bool
    dataloader_persistent_workers: bool
    dataloader_prefetch_factor: int | None
    dataloader_multiprocessing_context: str | None
    dataloader_in_order: bool
    remove_unused_columns: Literal[False]
    label_names: list[str] | None
    train_sampling_strategy: str
    length_column_name: str
    ddp_find_unused_parameters: bool | None
    ddp_bucket_cap_mb: int | None
    ddp_broadcast_buffers: bool | None
    ddp_static_graph: bool | None
    ddp_backend: str | None
    ddp_timeout: int
    fsdp: Literal[True] | None
    fsdp_config: FSDPConfig | None
    deepspeed: JsonObject | str | None
    debug: str | list[str]
    skip_memory_metrics: bool
    do_train: Literal[True]
    do_eval: bool
    do_predict: Literal[False]
    local_rank: int
    model_init_kwargs: Literal[None]
    trust_remote_code: bool
    router_aux_loss_coef: float
    chat_template_path: Literal[None]
    dataset_text_field: str
    dataset_kwargs: DatasetPreparationConfig
    dataset_num_proc: int | None
    eos_token: str | None
    max_length: Literal[None]
    truncation_mode: str
    shuffle_dataset: bool
    packing: Literal[False]
    packing_strategy: str
    padding_free: Literal[False]
    pad_to_multiple_of: Literal[None]
    eval_packing: Literal[False]
    completion_only_loss: Literal[False]
    assistant_only_loss: Literal[False]
    loss_type: Literal["nll", "chunked_nll", "dft"]
    activation_offloading: bool
    pad_token: Literal[None]


class SFTTrainerConfig(StrictConfig):
    compute_loss_func: Literal[None]
    compute_metrics: Literal[None]
    callbacks: Literal[None]
    optimizers: tuple[None, None]
    optimizer_cls_and_kwargs: Literal[None]
    preprocess_logits_for_metrics: Literal[None]
    quantization_config: Literal[None]
    peft_config: Literal[None]
    formatting_func: Literal[None]


class MethodConfig(StrictConfig):
    long_context: LongContextConfig | None
    name: str = Field(min_length=1)
    resume_from: Path | None
    backend: SFTBackendConfig
    trainer: SFTTrainerConfig
    learning_rate: float = Field(gt=0)
    epochs: float = Field(gt=0)
    max_steps: int = Field(ge=-1)
    batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    gradient_checkpointing: bool
    warmup_ratio: float = Field(ge=0, lt=1)
    weight_decay: float = Field(ge=0)
    logging_steps: int = Field(gt=0)
    save_steps: int = Field(gt=0)
    eval_steps: int = Field(gt=0)
    save_total_limit: int = Field(gt=0)
    dataloader_workers: int = Field(ge=0)
    use_cpu: bool

    @model_validator(mode="after")
    def steps(self):
        if self.max_steps == 0:
            raise ValueError("max_steps must be -1 or positive")
        return self


class TrainingConfig(StrictConfig):
    schema_version: Literal[1]
    experiment: ExperimentConfig
    data: DataConfig
    template: TemplateConfig
    model: ModelConfig
    method: MethodConfig
    distributed: DistributedConfig
    tracking: TrackingConfig

    @model_validator(mode="after")
    def compatible_training(self):
        backend = self.method.backend
        long_context = self.method.long_context
        if long_context is not None:
            if long_context.activation_dtype != self.model.dtype:
                raise ValueError("long_context.activation_dtype must match model.dtype")
            if backend.loss_type != "nll" or backend.use_liger_kernel:
                raise ValueError("long_context implements chunked NLL in the model; "
                                 "use backend.loss_type: nll and use_liger_kernel: false")
            if not backend.prediction_loss_only or backend.label_smoothing_factor != 0:
                raise ValueError("long_context requires prediction_loss_only: true and label_smoothing_factor: 0")
            if not self.model.freeze_vision:
                raise ValueError("long_context image microbatches require freeze_vision: true")
            if (not set(self.model.lora.target_modules) <= {"q_proj", "k_proj", "v_proj", "o_proj"}
                    or self.model.lora.backend.modules_to_save is not None
                    or self.model.lora.backend.target_parameters is not None):
                raise ValueError("long_context supports q/k/v/o LoRA only, without modules_to_save")
            if self.model.tuning != "lora":
                raise ValueError("long_context requires tuning: lora")
            if long_context.checkpoint_group_size is not None and not self.method.gradient_checkpointing:
                raise ValueError("Grouped checkpointing requires gradient_checkpointing: true")
            if backend.gradient_checkpointing_kwargs.use_reentrant:
                raise ValueError("long_context requires use_reentrant: false")
            if backend.activation_offloading or (backend.fsdp_config is not None and backend.fsdp_config.cpu_offload):
                raise ValueError("long_context does not permit CPU offloading")
            if long_context.require_optimized_kernels and self.model.attention_implementation != "sdpa":
                raise ValueError("Optimized long_context requires attention_implementation: sdpa")
            if backend.fsdp_config is not None and backend.fsdp_config.transformer_layer_cls_to_wrap not in (
                None, ["LongContextQwenDecoderLayer"],
            ):
                raise ValueError("long_context FSDP may wrap only LongContextQwenDecoderLayer; "
                                 "vision microbatches must not introduce per-image collectives")
        report_to = backend.report_to
        if self.tracking.mode == "disabled":
            if report_to not in ([], ["none"], "none"):
                raise ValueError("Disabled tracking requires report_to: [none] or []")
        elif report_to not in (["wandb"], "wandb"):
            raise ValueError("Enabled W&B tracking requires report_to: [wandb]")
        fsdp = self.distributed.strategy == "fsdp"
        if fsdp != (backend.fsdp is True) or fsdp != (backend.fsdp_config is not None):
            raise ValueError("distributed.strategy and backend.fsdp/fsdp_config must agree")
        if self.distributed.strategy != "single" and self.model.device_map is not None:
            raise ValueError("DDP/FSDP require model.device_map: null")
        quantization = self.model.quantization
        if quantization is not None:
            if self.model.tuning != "lora" or self.method.use_cpu:
                raise ValueError("4-bit training requires LoRA and GPU execution")
            if self.model.device_map is not None:
                raise ValueError("4-bit training requires model.device_map: null; each process selects its GPU")
            if quantization.use_gradient_checkpointing != self.method.gradient_checkpointing:
                raise ValueError("Quantization and method gradient_checkpointing must agree")
            if quantization.gradient_checkpointing_kwargs != backend.gradient_checkpointing_kwargs:
                raise ValueError("Quantization and backend gradient_checkpointing_kwargs must agree")
            if fsdp and quantization.bnb_4bit_quant_storage != self.model.dtype:
                raise ValueError("FSDP quantization storage must match model.dtype")
        return self


def load_config(path: str | Path, overrides: list[str] | None = None) -> TrainingConfig:
    from hydra import compose, initialize_config_dir

    path = Path(path).expanduser().resolve()
    with initialize_config_dir(version_base="1.3", config_dir=str(path.parent)):
        composed = compose(config_name=path.name, overrides=overrides or [])
    return parse_config(composed, path.parent)


def parse_config(composed, directory: Path) -> TrainingConfig:
    from omegaconf import OmegaConf

    config = parse_yaml_raw_as(TrainingConfig, OmegaConf.to_yaml(composed, resolve=True))
    for section, field in [(config.experiment, "output_dir"), (config.data, "runs_root"),
                           (config.data, "image_root"),
                           (config.model, "path"),
                           (config.method, "resume_from")]:
        value = getattr(section, field)
        if value is not None:
            expanded = value.expanduser()
            setattr(section, field, (expanded if expanded.is_absolute() else directory / expanded).resolve())
    if config.data.files is not None:
        expanded_files = [path.expanduser() for path in config.data.files]
        config.data.files = [(path if path.is_absolute() else directory / path).resolve()
                             for path in expanded_files]
    return config


def save_config(config: TrainingConfig, path: Path) -> None:
    from ruamel.yaml import YAML

    # Hydra uses YAML 1.1: quote strings such as "no" to preserve them on reload.
    writer = YAML()
    writer.version = (1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_yaml_str(config, custom_yaml_writer=writer), encoding="utf-8")
