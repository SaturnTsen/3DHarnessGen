"""Shared checkpoint layout and metadata; tensor I/O belongs to its consumer."""

from dataclasses import dataclass
import json
from pathlib import Path

from harness_training.config import read_yaml


@dataclass(frozen=True)
class Checkpoint:
    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", self.path.expanduser().resolve())

    @property
    def run_dir(self) -> Path:
        return self.path.parent.parent if self.path.parent.name == "checkpoints" else self.path.parent

    @property
    def snapshot_path(self) -> Path:
        return self.run_dir / "resolved_config.yaml"

    def snapshot(self) -> dict:
        return read_yaml(self.snapshot_path) if self.snapshot_path.is_file() else {}

    @property
    def model_shards(self) -> Path:
        return self.path / "pytorch_model_fsdp_0"

    @property
    def optimizer_shards(self) -> Path:
        return self.path / "optimizer_0"

    @property
    def is_fsdp(self) -> bool:
        return (self.model_shards / ".metadata").is_file()

    @property
    def manifest_path(self) -> Path:
        saved = self.path / "resume_manifest.json"
        return saved if saved.is_file() else self.run_dir / "manifest.json"

    def adapter_config(self) -> dict:
        source = self.path / "adapter_config.json"
        if source.is_file():
            return json.loads(source.read_text())
        from harness_training.config import LoRAConfig
        from harness_training.models.adapters import build_lora_config

        model = self.snapshot()["model"]
        if model["tuning"] != "lora":
            raise ValueError("Checkpoint does not contain a LoRA adapter")
        config = build_lora_config(LoRAConfig.model_validate(model["lora"])).to_dict()
        config["base_model_name_or_path"] = model["path"]
        return config
