from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

from harness_training.types import (
    AuditReport, DataReport, JsonObject, JsonValue, Manifest, ManifestPayload, Metrics, PreviewItem,
)

from harness_training.config import TrainingConfig, save_config
from harness_training.data.base import BaseDataset, DatasetBundle
from harness_training.data.trajectory import TrajectoryDataset
from harness_training.models.base import BaseModel
from harness_training.models.qwen import QwenModel
from harness_training.templates.base import BaseTemplate, TemplateError
from harness_training.templates.qwen import QwenTemplate
from harness_training.training.methods.base import BaseTrainingMethod
from harness_training.training.methods.sft import SFTMethod
from harness_training.training.tracking import tracking_session, record_resume_transition
from .manifest import backend_versions, code_identity, config_identity, hash_file, model_sources, hash_source
from .resume import check_resume, verify_encoded_manifest
from .preflight import preflight_log, validate_encoding


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

    def prepare(self, *, encode: bool = False, resume_targets: dict | None = None) -> DatasetBundle:
        if encode:
            preflight_log("开始读取轨迹、校验格式和检查图片；尚未加载模型权重。")
        data = self.dataset.prepare()
        if resume_targets is not None:
            from .manifest import ResumeMismatch, differences

            for split in ("train", "validation", "test"):
                selected = set(resume_targets[split])
                examples = [example for example in data.examples(split) if example.id in selected]
                details = differences(resume_targets[split], [example.id for example in examples], f"targets.{split}")
                if details:
                    raise ResumeMismatch(details)
                excluded = [example for example in data.examples(split) if example.id not in selected]
                for example in excluded:
                    data.rejected.append({"id": example.id, "path": str(example.record.path),
                                          "reason": "Not selected by the source checkpoint; not added during resume",
                                          "stage": "resume_selection"})
                if encode:
                    preflight_log(f"固定 checkpoint 的 {split} 样本集合：保留 {len(examples)}，"
                                  f"未插入 {len(excluded)} 条其他候选。")
                setattr(data, split, examples)
        if encode:
            preflight_log(f"数据检查完成：{data.report()['samples']}，拒绝记录 {len(data.rejected)} 条。")
            validate_encoding(self.config, data, lambda: self.template)
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
        identity = config_identity(self.config)
        sources = dict(data.source_hashes)
        images = {path for split in (data.train, data.validation, data.test) for example in split
                  for index in example.indices for path in example.record.images_by_message[index]}
        for path in sorted(images):
            if str(path) not in sources:
                sources[str(path)] = hash_file(path)
        for path in model_sources(self.config):
            if str(path) not in sources:
                sources[str(path)] = hash_source(path)
        payload: ManifestPayload = {
            "config": identity, "sources": sources, "versions": backend_versions(), "code": code_identity(),
            "targets": {split: [e.id for e in data.examples(split)] for split in ("train", "validation", "test")},
        }
        return {**payload, "fingerprint": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}

    def check_resume(self) -> dict:
        return check_resume(self.config, int(os.environ.get("WORLD_SIZE", "1")))[2]

    def run(self) -> Metrics:
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
        prior = None
        resume_report = None
        if resume_from is not None:
            if int(os.environ.get("RANK", "0")) == 0:
                print("[resume] 提前检查 checkpoint、配置、版本、源文件和 W&B 身份；不加载 Processor。", flush=True)
            prior, progress, resume_report = check_resume(self.config, world_size)
            cast(SFTMethod, self.method).resume_progress = progress
        preflight_log("检查训练运行环境和优化内核。")
        self.model.validate_runtime()
        # Validate and measure inputs before allocating model weights.
        data = self.prepare(encode=True, resume_targets=prior["targets"] if prior is not None else None)
        if not data.train:
            if int(os.environ.get("RANK", "0")) == 0:
                self.save_report(data)
            reasons = Counter(item["reason"] for item in data.rejected)
            summary = "; ".join(f"{count} 条：{reason[:500]}" for reason, count in reasons.most_common(3))
            raise ValueError(
                f"No usable training samples after preflight. Rejected: {len(data.rejected)}. "
                f"Most common reasons: {summary}. "
                f"Details: {output / 'rejected_samples.jsonl'}"
            )
        preflight_log("编码完成，计算数据指纹；模型权重尚未加载。")
        manifest = self._manifest(data)
        if prior is not None:
            verify_encoded_manifest(prior, cast(dict, manifest), self.config, cast(dict, resume_report))
            if self.config.method.resume_policy == "elastic":
                from .elastic import DataProgress

                progress = cast(DataProgress, cast(SFTMethod, self.method).resume_progress)
                preflight_log(f"恢复数据位置：第 {progress.epoch} 轮，"
                              f"下一样本位置 {progress.next_sample}；"
                              f"进程数 {progress.world_size} → {world_size}。")
        cast(SFTMethod, self.method).resume_manifest = manifest
        if int(os.environ.get("RANK", "0")) == 0:
            self.save_report(data)
            # Keep legacy provenance available when retrying an older checkpoint.
            if resume_from is None or self.config.method.resume_policy == "strict":
                save_config(self.config, output / "resolved_config.yaml")
                write_json(output / "manifest.json", manifest)
        preflight_log("预检查完成，进入训练后端并加载模型权重。")
        with tracking_session(self.config, resume_from, is_main_process=int(os.environ.get("RANK", "0")) == 0):
            progress = cast(SFTMethod, self.method).resume_progress
            if progress is not None:
                record_resume_transition(self.config, progress, world_size)
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
