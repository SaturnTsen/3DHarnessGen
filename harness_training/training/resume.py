"""Early resume checks: no Processor, image decoding, model tensors or W&B connection."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from harness_training.config import TrainingConfig
from harness_training.data.trajectory import TrajectoryDataset
from .checkpoint import checkpoint_problem
from .elastic import DataProgress, load_progress, validate_manifest
from .manifest import (
    ResumeMismatch, backend_versions, code_identity, config_identity, differences, saved_source_inventory,
)
from .tracking import WandbRunState
from .preflight import PreflightWorkers


def write_resume_report(report: dict) -> Path:
    path = Path(__file__).resolve().parents[2] / "temp/resume_check.json"
    if int(os.environ.get("RANK", "0")) == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)
    return path


def resume_stage(report: dict, stage: str, message: str, workers: PreflightWorkers | None) -> None:
    report["stage"] = stage
    write_resume_report(report)
    if workers is not None:
        workers.log(message)


def check_resume(config: TrainingConfig, world_size: int, *,
                 workers: PreflightWorkers | None = None) -> tuple[dict, DataProgress | None, dict]:
    checkpoint = config.method.resume_from
    report = {"status": "checking", "checkpoint": str(checkpoint), "world_size": world_size,
              "processor_loaded": False, "weights_loaded": False, "wandb_connected": False}
    try:
        resume_stage(report, "checkpoint_config", "校验 checkpoint、配置、后端版本及代码。", workers)
        if checkpoint is None:
            raise ValueError("check-resume requires method.resume_from (a checkpoint path or auto)")
        if (world_size == 1) != (config.distributed.strategy == "single"):
            raise ValueError("single requires one process; ddp/fsdp require torchrun")
        if checkpoint.parent != (config.experiment.output_dir / "checkpoints").resolve():
            raise ValueError("Resume checkpoint must belong to this output directory")
        problem = checkpoint_problem(checkpoint, config, world_size)
        if problem is not None:
            raise ValueError(f"{checkpoint.name}: {problem}")
        manifest = checkpoint / "resume_manifest.json"
        if not manifest.is_file():
            manifest = config.experiment.output_dir / "manifest.json"
        prior = json.loads(manifest.read_text())
        current = {"config": config_identity(config), "versions": backend_versions(), "code": code_identity(),
                   "targets": prior["targets"], "sources": prior["sources"]}
        # Catch cheap version/configuration/code errors before hashing any large input files.
        validate_compatibility(prior, current, config)
        resume_stage(report, "training_state", "校验全局 batch、数据位置及本地 W&B 身份。", workers)
        progress = load_progress(checkpoint, prior, config, world_size) if config.method.resume_policy == "elastic" else None
        if config.tracking.mode != "disabled":
            saved = WandbRunState.model_validate_json((config.experiment.output_dir / "wandb_run.json").read_text())
            tracking = config.tracking
            selected = {"project": tracking.project, "entity": tracking.entity, "mode": tracking.mode}
            details = differences({key: getattr(saved, key) for key in selected}, selected, "tracking")
            if tracking.run_id is not None and tracking.run_id != saved.id:
                details.extend(differences(saved.id, tracking.run_id, "tracking.run_id"))
            if details:
                raise ResumeMismatch(details)
            report["wandb_run_id"] = saved.id
        resume_stage(report, "source_files", "静态条件通过，开始核对源文件；尚未加载 Processor 和模型权重。", workers)
        dataset = TrajectoryDataset(config.data, config.experiment.scene, config.experiment.seed)
        current["sources"] = saved_source_inventory(prior, dataset.input_paths(), config, workers=workers)
        validate_compatibility(prior, current, config)
        report.update({"status": "passed", "checked_sources": len(current["sources"]),
                       "sample_counts": {split: len(ids) for split, ids in prior["targets"].items()},
                       "sample_selection": "checkpoint targets, original order",
                       "length_limits": {section: {"previous": prior["config"][section]["max_length"],
                                                    "current": current["config"][section]["max_length"]}
                                         for section in ("model", "template")},
                       "encoding_verification": "pending", "tensor_load_verification": "pending"})
        if progress is not None:
            report["data_position"] = {"global_step": progress.global_step, "epoch": progress.epoch,
                                       "next_sample": progress.next_sample, "source_world_size": progress.world_size,
                                       "global_batch_size": progress.global_batch_size}
        resume_stage(report, "complete", f"恢复检查通过：{len(current['sources'])} 个源文件已核对。", workers)
    except (ValueError, OSError) as exc:
        report.update({"status": "failed", "error": str(exc)})
        if isinstance(exc, ResumeMismatch):
            report["differences"] = exc.details
        path = write_resume_report(report)
        raise ValueError(f"{exc}\nResume report: {path}") from exc
    path = write_resume_report(report)
    report["report_path"] = str(path)
    return prior, progress, report


def validate_compatibility(prior: dict, current: dict, config: TrainingConfig) -> None:
    if config.method.resume_policy == "elastic":
        validate_manifest(prior, current)
    else:
        payload = {key: value for key, value in current.items() if key != "fingerprint"}
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if fingerprint != prior["fingerprint"]:
            details = differences({key: prior[key] for key in payload}, payload, "manifest")
            raise ResumeMismatch(details)


def verify_encoded_manifest(prior: dict, current: dict, config: TrainingConfig, report: dict) -> None:
    try:
        validate_compatibility(prior, current, config)
    except ResumeMismatch as exc:
        report.update({"status": "failed", "encoding_verification": "failed",
                       "error": str(exc), "differences": exc.details})
        path = write_resume_report(report)
        raise ValueError(f"{exc}\nResume report: {path}") from exc
    report["encoding_verification"] = "passed"
    write_resume_report(report)
