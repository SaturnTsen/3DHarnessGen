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
    seed: int
    output_dir: Path


class DataConfig(StrictConfig):
    runs_root: Path
    source_model: str
    instances: list[str] | None
    validation_fraction: float = Field(ge=0, lt=1)
    test_fraction: float = Field(ge=0, lt=1)
    missing_images: Literal["skip_sample", "error"]
    verify_images: bool
    accepted_statuses: list[str]
    strip_plugin_recommendations: bool
    include_final_reply: bool

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
    name: Literal["qwen"]
    enable_thinking: Literal[False]
    tools_path: Path | None
    max_length: int = Field(ge=16)
    min_image_pixels: int = Field(ge=1024)
    max_image_pixels: int = Field(ge=1024)

    @model_validator(mode="after")
    def image_budget(self):
        if self.min_image_pixels > self.max_image_pixels:
            raise ValueError("min_image_pixels cannot exceed max_image_pixels")
        return self


class LoRAConfig(StrictConfig):
    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)
    target_modules: list[str]


class ModelConfig(StrictConfig):
    name: Literal["qwen"]
    path: Path
    dtype: Literal["bfloat16", "float16", "float32"]
    tuning: Literal["lora", "full"]
    freeze_vision: bool
    attention_implementation: Literal["sdpa", "eager", "flash_attention_2"]
    lora: LoRAConfig


class MethodConfig(StrictConfig):
    name: Literal["sft"]
    resume_from: Path | None
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


def load_config(path: str | Path) -> TrainingConfig:
    path = Path(path).expanduser().resolve()
    config = parse_yaml_raw_as(TrainingConfig, path.read_text(encoding="utf-8"))
    for section, field in [(config.experiment, "output_dir"), (config.data, "runs_root"),
                           (config.model, "path"), (config.template, "tools_path"),
                           (config.method, "resume_from")]:
        value = getattr(section, field)
        if value is not None:
            expanded = value.expanduser()
            setattr(section, field, (expanded if expanded.is_absolute() else path.parent / expanded).resolve())
    return config


def save_config(config: TrainingConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_yaml_str(config), encoding="utf-8")
