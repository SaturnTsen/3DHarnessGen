from __future__ import annotations

import hashlib
from pathlib import Path
from random import Random
from typing import Literal

from harness_training.config import DataConfig, Scene
from .contract import ImageBlock, Message, Trajectory, validate_record
from .base import BaseDataset, DatasetBundle, TrainingExample, TrajectoryRecord


class DataError(ValueError):
    pass


def validate_image(path: Path, decode: bool) -> str | None:
    if not path.is_file():
        return f"missing image: {path}"
    if decode:
        from PIL import Image

        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            return f"unreadable image: {path}: {exc}"
    return None


class TrajectoryDataset(BaseDataset):
    def __init__(self, config: DataConfig, scene: Scene, seed: int):
        self.config, self.scene, self.seed = config, scene, seed
        self._image_errors: dict[Path, str | None] = {}

    def prepare(self) -> DatasetBundle:
        self._image_errors.clear()
        bundle = DatasetBundle()
        root = self.config.runs_root / self.scene / self.config.source_model
        paths = sorted(root.glob("*/post_train_trajectory/trajectory.jsonl"))
        if self.config.instances is not None:
            paths = [p for p in paths if p.parent.parent.name in self.config.instances]
            missing = set(self.config.instances) - {p.parent.parent.name for p in paths}
            if missing:
                raise DataError(f"Instances not found under {root}: {sorted(missing)}")
        if not paths:
            raise DataError(f"No trajectory.jsonl files found under {root}")
        assignments = self._split_objects(paths)
        for path in paths:
            sessions: set[str] = set()
            instance = path.parent.parent.name
            split = assignments[instance]
            bundle.assignments[instance] = split
            payload = path.read_bytes()
            bundle.source_hashes[str(path)] = hashlib.sha256(payload).hexdigest()
            if not payload:
                bundle.rejected.append({"path": str(path), "reason": "Empty trajectory file"})
            for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
                try:
                    record = self._read_record(path, line_number, instance, validate_record(line))
                    if record.row.metadata.session_id in sessions:
                        raise DataError("Duplicate session_id within object")
                    sessions.add(record.row.metadata.session_id)
                    examples = self._select(record)
                except ValueError as exc:
                    bundle.rejected.append({"path": str(path), "line": line_number, "reason": str(exc)})
                    continue
                for example in examples:
                    errors = self._example_errors(example)
                    if errors:
                        if self.config.missing_images == "error":
                            raise DataError(f"{example.id}: {'; '.join(errors)}")
                        bundle.rejected.append({"id": example.id, "path": str(path), "reason": "; ".join(errors)})
                    else:
                        bundle.examples(split).append(example)
        return bundle

    def _split_objects(self, paths: list[Path]) -> dict[str, Literal["train", "validation", "test"]]:
        shuffled_paths = paths.copy()
        Random(self.seed).shuffle(shuffled_paths)

        test_end = int(len(shuffled_paths) * self.config.test_fraction)
        validation_end = test_end + int(len(shuffled_paths) * self.config.validation_fraction)
        assignments: dict[str, Literal["train", "validation", "test"]] = {
            path.parent.parent.name: "train" for path in shuffled_paths[validation_end:]
        }
        assignments.update({path.parent.parent.name: "validation" for path in shuffled_paths[test_end:validation_end]})
        assignments.update({path.parent.parent.name: "test" for path in shuffled_paths[:test_end]})
        return assignments

    def _read_record(self, path: Path, line: int, instance: str, row: Trajectory) -> TrajectoryRecord:
        if row.metadata.scene != self.scene or row.metadata.instance != instance:
            raise DataError("Scene or instance metadata does not match input path")
        bindings = tuple(
            tuple((path.parent / block.image_url.url).resolve()
                  for block in item.content if isinstance(block, ImageBlock))
            if isinstance(item, Message) and isinstance(item.content, list) else ()
            for item in row.messages
        )
        return TrajectoryRecord(path, line, instance, row, bindings)

    def _select(self, record: TrajectoryRecord) -> list[TrainingExample]:
        return [TrainingExample(record, tuple(step.context_indices) + (step.target_index,),
                                step.target_index, step.id)
                for step in record.row.metadata.steps
                if step.status in self.config.accepted_statuses
                and (step.target_kind != "final_reply" or self.config.include_final_reply)]

    def _example_errors(self, example: TrainingExample) -> list[str]:
        errors: list[str] = []
        for index in example.indices:
            for path in example.record.images_by_message[index]:
                if path not in self._image_errors:
                    self._image_errors[path] = validate_image(path, self.config.verify_images)
                error = self._image_errors[path]
                if error is not None:
                    errors.append(error)
        return list(dict.fromkeys(errors))
