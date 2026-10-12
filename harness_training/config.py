from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_yaml import parse_yaml_raw_as, to_yaml_str

Scene = Literal["Single-view", "Multi-view", "ActiveVisual", "Full3DInteraction"]


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(StrictConfig):
    name: str
    scene: Scene
    seed: int = 42
    output_dir: Path


class DataConfig(StrictConfig):
    runs_root: Path
    source_model: str = "gpt-6-astra"
    instances: list[str] | None = None
    validation_fraction: float = Field(default=0.1, ge=0, lt=1)
    test_fraction: float = Field(default=0.1, ge=0, lt=1)
    missing_images: Literal["skip_sample", "error"] = "skip_sample"
    verify_images: bool = True
    accepted_statuses: list[str] = Field(default_factory=lambda: ["OK"])
    strip_plugin_recommendations: bool = True
    include_final_reply: bool = True

    @model_validator(mode="after")
    def valid_split(self):
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation_fraction + test_fraction must be less than 1")
        if self.instances is not None and (not self.instances or len(set(self.instances)) != len(self.instances)):
            raise ValueError("instances must be nonempty and unique when specified")
        for name in [self.source_model, *(self.instances or [])]:
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError("source_model and instances must be directory names")
        return self


class TemplateConfig(StrictConfig):
    name: Literal["qwen"] = "qwen"
    enable_thinking: Literal[False] = False
    tools_path: Path | None = None
    max_length: int = Field(default=32768, ge=16)
    min_image_pixels: int = Field(default=65536, ge=1024)
    max_image_pixels: int = Field(default=262144, ge=1024)

    @model_validator(mode="after")
    def image_budget(self):
        if self.min_image_pixels > self.max_image_pixels:
            raise ValueError("min_image_pixels cannot exceed max_image_pixels")
        return self


class LoRAConfig(StrictConfig):
    rank: int = Field(default=16, gt=0)
    alpha: int = Field(default=32, gt=0)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])


class ModelConfig(StrictConfig):
    name: Literal["qwen"] = "qwen"
    path: Path
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    tuning: Literal["lora", "full"] = "lora"
    freeze_vision: bool = True
    attention_implementation: Literal["sdpa", "eager", "flash_attention_2"] = "sdpa"
    lora: LoRAConfig = Field(default_factory=LoRAConfig)


class MethodConfig(StrictConfig):
    name: Literal["sft"] = "sft"
    learning_rate: float = Field(default=2e-5, gt=0)
    epochs: float = Field(default=1, gt=0)
    max_steps: int = Field(default=-1, ge=-1)
    batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=8, gt=0)
    gradient_checkpointing: bool = True
    warmup_ratio: float = Field(default=0.03, ge=0, lt=1)
    weight_decay: float = Field(default=0.01, ge=0)
    logging_steps: int = Field(default=10, gt=0)
    save_steps: int = Field(default=100, gt=0)
    eval_steps: int = Field(default=100, gt=0)
    save_total_limit: int = Field(default=2, gt=0)
    dataloader_workers: int = Field(default=0, ge=0)
    use_cpu: bool = False

    @model_validator(mode="after")
    def steps(self):
        if self.max_steps == 0:
            raise ValueError("max_steps must be -1 or positive")
        return self


class TrainingConfig(StrictConfig):
    schema_version: Literal[1] = 1
    experiment: ExperimentConfig
    data: DataConfig
    template: TemplateConfig = Field(default_factory=TemplateConfig)
    model: ModelConfig
    method: MethodConfig = Field(default_factory=MethodConfig)


def load_config(path: str | Path) -> TrainingConfig:
    path = Path(path).expanduser().resolve()
    config = parse_yaml_raw_as(TrainingConfig, path.read_text(encoding="utf-8"))
    for section, field in [(config.experiment, "output_dir"), (config.data, "runs_root"),
                           (config.model, "path"), (config.template, "tools_path")]:
        value = getattr(section, field)
        if value is not None:
            expanded = value.expanduser()
            setattr(section, field, (expanded if expanded.is_absolute() else path.parent / expanded).resolve())
    return config


def save_config(config: TrainingConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_yaml_str(config), encoding="utf-8")
