from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Literal

from harness_training.config import DataConfig, Scene
from .contract import ImageBlock, Message, Trajectory, validate_record
from harness_training.types import Rejection
from .base import BaseDataset, DatasetBundle, PreparationWorkers, SerialPreparationWorkers, TrainingExample, TrajectoryRecord


class DataError(ValueError):
    pass


@dataclass
class FileRecords:
    index: int
    path: Path
    sha256: str
    entries: list[TrajectoryRecord | Rejection]


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
    def __init__(self, config: DataConfig, scene: Scene | list[Scene], seed: int):
        self.config, self.seed = config, seed
        self.scenes = [scene] if isinstance(scene, str) else scene
        self._image_errors: dict[Path, str | None] = {}

    def prepare(self, *, workers: PreparationWorkers | None = None) -> DatasetBundle:
        self._image_errors.clear()
        workers = workers if workers is not None else SerialPreparationWorkers()
        paths = self.input_paths()
        # Only the owning rank reads and validates each source file. Exchange
        # validated in-memory records, never export another training dataset.
        with workers.progress("读取和校验轨迹文件", len(paths)) as advance:
            local = []
            for index in range(workers.rank, len(paths), workers.world_size):
                local.append(self._read_path(index, paths[index]))
                advance()
            files = sorted((item for shard in workers.gather("records", local) for item in shard),
                           key=lambda item: item.index)
        bundle = DatasetBundle()
        entries: list[TrajectoryRecord | Rejection] = []
        sessions = set()
        for file in files:
            bundle.source_hashes[str(file.path)] = file.sha256
            for entry in file.entries:
                if isinstance(entry, TrajectoryRecord):
                    key = ((entry.row.metadata.scene, entry.instance, entry.row.metadata.session_id)
                           if self.config.files is not None else (file.index, entry.row.metadata.session_id))
                    if key in sessions:
                        entries.append({"path": str(entry.path), "line": entry.line,
                                        "reason": "Duplicate session_id within object"})
                        continue
                    sessions.add(key)
                entries.append(entry)
        records = [entry for entry in entries if isinstance(entry, TrajectoryRecord)]
        assignments = (self._split_instances(sorted({record.instance for record in records}))
                       if self.config.files is not None else self._split_objects(paths))
        bundle.assignments.update(assignments)
        selections = {(record.path, record.line): self._select(record) for record in records}
        images = sorted({path for examples in selections.values() for example in examples
                         for index in example.indices for path in example.record.images_by_message[index]})
        with workers.progress("检查去重后的图片", len(images)) as advance:
            local_errors = {}
            for path in images[workers.rank::workers.world_size]:
                local_errors[path] = self._check_image(path)
                advance()
            for shard in workers.gather("image_checks", local_errors):
                self._image_errors.update(shard)
        # Explicit file input historically reports all parse failures before
        # sample failures; directory input reports them in source/line order.
        if self.config.files is not None:
            bundle.rejected.extend(entry for entry in entries if not isinstance(entry, TrajectoryRecord))
        for entry in entries:
            if not isinstance(entry, TrajectoryRecord):
                if self.config.files is None:
                    bundle.rejected.append(entry)
                continue
            for example in selections[(entry.path, entry.line)]:
                errors = self._example_errors(example)
                if errors:
                    if self.config.missing_images == "error":
                        raise DataError(f"{example.id}: {'; '.join(errors)}")
                    bundle.rejected.append({"id": example.id, "path": str(entry.path), "reason": "; ".join(errors)})
                else:
                    bundle.examples(assignments[entry.instance]).append(example)
        return bundle

    def input_paths(self) -> list[Path]:
        if self.config.files is not None:
            return self.config.files
        paths: list[Path] = []
        for scene in self.scenes:
            root = self.config.runs_root / scene / self.config.source_model
            selected = sorted(root.glob("*/post_train_trajectory/trajectory.jsonl"))
            if self.config.instances is not None:
                selected = [p for p in selected if p.parent.parent.name in self.config.instances]
                missing = set(self.config.instances) - {p.parent.parent.name for p in selected}
                if missing:
                    raise DataError(f"Instances not found under {root}: {sorted(missing)}")
            if not selected:
                raise DataError(f"No trajectory.jsonl files found under {root}")
            paths.extend(selected)
        return paths

    def _read_path(self, index: int, path: Path) -> FileRecords:
        payload = path.read_bytes()
        result = FileRecords(index, path, hashlib.sha256(payload).hexdigest(), [])
        if not payload:
            result.entries.append({"path": str(path), "reason": "Empty trajectory file"})
        for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
            try:
                row = validate_record(line, explicit_image_root=self.config.files is not None and self.config.image_root is not None)
                if self.config.files is None and row.metadata.scene != path.parents[3].name:
                    raise DataError("Scene metadata does not match input path")
                instance = row.metadata.instance if self.config.files is not None else path.parent.parent.name
                record = self._read_record(path, line_number, instance, row)
            except ValueError as exc:
                result.entries.append({"path": str(path), "line": line_number, "reason": str(exc)})
            else:
                result.entries.append(record)
        return result

    def _check_image(self, path: Path) -> str | None:
        return validate_image(path, self.config.verify_images)

    def _split_objects(self, paths: list[Path]) -> dict[str, Literal["train", "validation", "test"]]:
        return self._split_instances(sorted({path.parent.parent.name for path in paths}))

    def _split_instances(self, instances: list[str]) -> dict[str, Literal["train", "validation", "test"]]:
        shuffled = instances.copy()
        Random(self.seed).shuffle(shuffled)

        test_end = int(len(shuffled) * self.config.test_fraction)
        validation_end = test_end + int(len(shuffled) * self.config.validation_fraction)
        assignments: dict[str, Literal["train", "validation", "test"]] = {
            instance: "train" for instance in shuffled[validation_end:]
        }
        assignments.update({instance: "validation" for instance in shuffled[test_end:validation_end]})
        assignments.update({instance: "test" for instance in shuffled[:test_end]})
        return assignments

    def _read_record(self, path: Path, line: int, instance: str, row: Trajectory) -> TrajectoryRecord:
        if row.metadata.scene not in self.scenes or row.metadata.instance != instance:
            raise DataError("Scene or instance metadata does not match input path")
        bindings = tuple(
            tuple(((self.config.image_root or path.parent) / block.image_url.url).resolve()
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
                error = self._image_errors[path]
                if error is not None:
                    errors.append(error)
        return list(dict.fromkeys(errors))
