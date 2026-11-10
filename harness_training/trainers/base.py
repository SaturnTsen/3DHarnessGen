from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from harness_training.config import TrainingConfig, save_config
from harness_training.data.base import BaseDataset, DatasetBundle
from harness_training.data.trajectory import TrajectoryDataset
from harness_training.utils.manifest import build_manifest
from harness_training.models.base import BaseModel
from harness_training.models.qwen import QwenModel
from harness_training.chat_templates.base import BaseTemplate, SampleTooLong, TemplateError
from harness_training.chat_templates.qwen import QwenTemplate
from harness_training.types import AuditReport, DataReport, JsonValue, Manifest, Metrics, PreviewItem


class BaseTrainer(ABC):
    """Shared data inspection, preparation and persistence for concrete trainers."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.dataset: BaseDataset = TrajectoryDataset(
            config.data, config.experiment.scene, config.experiment.seed)
        self.model: BaseModel
        match config.model.name:
            case "qwen":
                self.model = QwenModel(config.model, config.template, config.method.long_context)
            case _:
                raise ValueError(f"Unsupported model: {config.model.name}")
        self._template: BaseTemplate | None = None

    @property
    def template(self) -> BaseTemplate:
        if self._template is None:
            self._template = QwenTemplate(self.config.template, self.model.processor)
        return self._template

    def prepare(self, *, encode: bool = False) -> DatasetBundle:
        data = self.dataset.prepare()
        if encode:
            for split in ("train", "validation", "test"):
                accepted = []
                for example in data.examples(split):
                    try:
                        self.template.encode(example)
                    except (TemplateError, OSError, ValueError) as exc:
                        data.rejected.append({"id": example.id, "path": str(example.record.path),
                                              "reason": str(exc), "stage": "encoding",
                                              "too_long": isinstance(exc, SampleTooLong)})
                    else:
                        accepted.append(example)
                setattr(data, split, accepted)
        return data

    def audit(self) -> AuditReport:
        data = self.prepare()
        return {**data.report(), "rejections": data.rejected}

    def preview(self, limit: int = 1, *, split: Literal["train", "validation", "test"] = "train", encode: bool = True) -> list[PreviewItem]:
        data = self.prepare()
        results: list[PreviewItem] = []
        for example in data.examples(split)[:limit]:
            try:
                rendered = self.template.render(example)
                item: PreviewItem = {"id": example.id, "source": str(example.record.path), "line": example.record.line,
                        "prompt": rendered.prompt, "target": rendered.text[len(rendered.prompt):],
                        "images": [str(p) for p in rendered.images]}
                if encode:
                    encoded = self.template.encode(example)
                    item.update(tokens=encoded["input_ids"].shape[1],
                                target_tokens=int((encoded["labels"] != self.config.template.label_ignore_index).sum()))
                results.append(item)
            except (TemplateError, OSError, ValueError) as exc:
                results.append({"id": example.id, "error": str(exc)})
        return results

    @abstractmethod
    def run(self) -> Metrics:
        """Run a concrete training workflow."""

    def _prepare_run(self) -> DatasetBundle:
        # Revalidate programmatic edits before loading the Processor or weights.
        TrainingConfig.model_validate(self.config.model_dump())
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        strategy = self.config.distributed.strategy
        if (world_size > 1) != (strategy != "single"):
            raise ValueError("single requires one process; ddp/fsdp require torchrun with at least two processes")
        if strategy == "single" and not self.config.method.use_cpu and self.config.model.device_map is None:
            import torch

            if torch.cuda.device_count() > 1:
                raise ValueError("single requires one visible GPU: set CUDA_VISIBLE_DEVICES, or select ddp/fsdp")
        output = self.config.experiment.output_dir
        if (output / "final").exists() or list((output / "checkpoints").glob("checkpoint-*")):
            raise ValueError("Output already contains a training run; choose another output_dir")
        self.model.validate_runtime()
        # Validate and measure inputs before allocating model weights.
        data = self.prepare(encode=True)
        if not data.train:
            self.save_report(data)
            reasons = Counter(item["reason"] for item in data.rejected)
            summary = "; ".join(f"{count} 条：{reason[:500]}" for reason, count in reasons.most_common(3))
            raise ValueError(
                f"No usable training samples after encoding checks. Rejected: {len(data.rejected)}. "
                f"Most common reasons: {summary}. "
                f"Details: {output / 'rejected_samples.jsonl'}"
            )
        manifest = build_manifest(self.config, data)
        if int(os.environ.get("RANK", "0")) == 0:
            self.save_report(data)
            save_config(self.config, output / "resolved_config.yaml")
            self._write_json(output / "manifest.json", manifest)
        return data

    def save_report(self, data: DatasetBundle) -> None:
        if int(os.environ.get("RANK", "0")) != 0:
            return
        output = self.config.experiment.output_dir
        self._write_json(output / "data_report.json", data.report())
        self._write_json(output / "object_splits.json", data.assignments)
        (output / "rejected_samples.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in data.rejected), encoding="utf-8")

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, JsonValue] | DataReport | Manifest | Metrics) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
