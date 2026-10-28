from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast, TYPE_CHECKING

from harness_training.types import (
    AuditReport, DataReport, JsonObject, JsonValue, Manifest, ManifestPayload, Metrics, PreviewItem,
)

from harness_training.config import TrainingConfig, save_config
from harness_training.data.base import BaseDataset, DatasetBundle
from harness_training.data.trajectory import TrajectoryDataset
from harness_training.models.base import BaseModel
from harness_training.models.qwen import QwenModel
from harness_training.templates.base import BaseTemplate, SampleTooLong, TemplateError
from harness_training.templates.qwen import QwenTemplate
from harness_training.training.methods.base import BaseTrainingMethod
from harness_training.training.methods.sft import SFTMethod
from harness_training.training.tracking import tracking_session

if TYPE_CHECKING:
    from harness_training.data.base import TrainingExample


def write_json(path: Path, value: Mapping[str, JsonValue] | DataReport | Manifest | Metrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class Trainer:
    """Unified entrypoint. Concrete components are initialized directly."""

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
        self.method: BaseTrainingMethod
        match config.method.name:
            case "sft":
                self.method = SFTMethod(config.method, config.experiment.seed)
            case _:
                raise ValueError(f"Unsupported training method: {config.method.name}")

    @property
    def template(self) -> BaseTemplate:
        if self._template is None:
            self._template = QwenTemplate(self.config.template, self.model.processor)
        return self._template

    def prepare(self, *, encode: bool = False) -> DatasetBundle:
        data = self.dataset.prepare()
        if encode:
            for split in ("train", "validation", "test"):
                accepted: list[TrainingExample] = []
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

    def _manifest(self, data: DatasetBundle) -> Manifest:
        identity: JsonObject = self.config.model_dump(mode="json")
        cast(JsonObject, identity["experiment"]).pop("output_dir")
        for key in ("max_steps", "epochs", "logging_steps", "save_steps", "eval_steps", "save_total_limit", "dataloader_workers"):
            cast(JsonObject, identity["method"]).pop(key)
        cast(JsonObject, identity["method"]).pop("resume_from")
        sources = dict(data.source_hashes)
        images = {path for split in (data.train, data.validation, data.test) for example in split
                  for index in example.indices for path in example.record.images_by_message[index]}
        for path in sorted(images):
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(self.config.model.path.glob("*.json")) + sorted(self.config.model.path.glob("*.jinja")):
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "trl", "pydantic-yaml")}
        code_root = Path(__file__).resolve().parents[1]
        code = {str(path.relative_to(code_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(code_root.rglob("*.py"))}
        payload: ManifestPayload = {
            "config": identity, "sources": sources, "versions": versions, "code": code,
            "targets": {split: [e.id for e in data.examples(split)] for split in ("train", "validation", "test")},
        }
        return {**payload, "fingerprint": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}

    def run(self) -> Metrics:
        import os

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
        resume_from = self.config.method.resume_from
        output = self.config.experiment.output_dir
        if resume_from is None and ((output / "final").exists() or list((output / "checkpoints").glob("checkpoint-*"))):
            raise ValueError("Output already contains a training run; choose another output_dir or resume explicitly")
        if resume_from is not None:
            resume_from = resume_from.expanduser().resolve()
            if not resume_from.is_dir() or not (resume_from / "trainer_state.json").is_file():
                raise ValueError("resume_from must be a saved training checkpoint")
            if resume_from.parent != (output / "checkpoints").resolve():
                raise ValueError("Resume checkpoint must belong to this output directory")
        self.model.validate_runtime()
        # Validate and measure inputs before allocating model weights.
        data = self.prepare(encode=True)
        manifest = self._manifest(data)
        if resume_from is not None:
            prior = json.loads((output / "manifest.json").read_text())
            if prior["fingerprint"] != manifest["fingerprint"]:
                raise ValueError("Data, model, template, or training settings changed since this checkpoint")
        if not data.train:
            raise ValueError(f"No usable training samples after preflight: {data.report()}")
        if int(os.environ.get("RANK", "0")) == 0:
            self.save_report(data)
            save_config(self.config, output / "resolved_config.yaml")
            write_json(output / "manifest.json", manifest)
        with tracking_session(self.config, resume_from, is_main_process=int(os.environ.get("RANK", "0")) == 0):
            metrics = self.method.run(self.model, self.template, data, output, resume_from)
            # Distributed training has a single writer for the final metrics.
            if int(os.environ.get("RANK", "0")) == 0:
                write_json(output / "metrics.json", metrics)
        return metrics

    def save_report(self, data: DatasetBundle) -> None:
        output = self.config.experiment.output_dir
        write_json(output / "data_report.json", data.report())
        write_json(output / "object_splits.json", data.assignments)
        output.mkdir(parents=True, exist_ok=True)
        (output / "rejected_samples.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in data.rejected), encoding="utf-8")
