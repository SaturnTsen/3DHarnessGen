"""Bounded source/Processor checks. Test suites and training are separate actions."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import mkdtemp

from harness_training.config import TrainingConfig, save_config
from harness_training.data.trajectory import TrajectoryDataset, validate_image
from harness_training.training.trainer import Trainer


def run_smoke(config: TrainingConfig, *, limit: int, encode: bool = False) -> dict:
    if limit <= 0:
        raise ValueError("limit must be positive")
    root = Path(__file__).resolve().parents[2] / "temp"
    root.mkdir(exist_ok=True)
    output = Path(mkdtemp(prefix="smoke-post-training-", dir=root))
    save_config(config, output / "resolved_config.yaml")
    dataset = TrajectoryDataset(config.data, config.experiment.scene, config.experiment.seed)
    trainer = Trainer(config) if encode else None
    items = []
    checked_records = 0
    for path in dataset.input_paths():
        with path.open(encoding="utf-8") as stream:
            for line, raw in enumerate(stream, 1):
                checked_records += 1
                try:
                    record = dataset.read_record(path, line, raw)
                    for example in dataset.select(record)[:limit - len(items)]:
                        images = [image for index in example.indices for image in record.images_by_message[index]]
                        errors = [error for image in dict.fromkeys(images)
                                  if (error := validate_image(image, config.data.verify_images)) is not None]
                        if errors:
                            raise ValueError("; ".join(errors))
                        item = {"id": example.id, "source": str(path), "line": line, "images": len(images)}
                        if trainer is not None:
                            encoded = trainer.template.encode(example)
                            item["tokens"] = encoded["input_ids"].shape[1]
                            item["target_tokens"] = int((encoded["labels"] != config.template.label_ignore_index).sum())
                        items.append(item)
                except (ValueError, OSError) as exc:
                    items.append({"source": str(path), "line": line, "error": str(exc)})
                if checked_records >= limit or len(items) >= limit:
                    break
        if checked_records >= limit or len(items) >= limit:
            break
    result = {"passed": bool(items) and all("error" not in item for item in items),
              "checked_records": checked_records, "samples": items, "processor_checked": encode,
              "scope": "first configured records and selected targets; no split changes",
              "training_run": False, "benchmark_run": False, "report_dir": str(output)}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result
