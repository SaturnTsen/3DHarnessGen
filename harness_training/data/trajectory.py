from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from harness_training.types import JsonValue, SourceMapping, TrajectoryRow

from harness_training.config import DataConfig, Scene
from .base import BaseDataset, DatasetBundle, TrainingExample, TrajectoryRecord
from pathlib import Path

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
        for path in paths:
            instance = path.parent.parent.name
            # Hashing by object name gives the same split across four scenes.
            fraction = int(hashlib.sha256(f"{self.seed}:{instance}".encode()).hexdigest()[:16], 16) / 2**64
            split = ("test" if fraction < self.config.test_fraction else
                     "validation" if fraction < self.config.test_fraction + self.config.validation_fraction else "train")
            bundle.assignments[instance] = split
            payload = path.read_bytes()
            bundle.source_hashes[str(path)] = hashlib.sha256(payload).hexdigest()
            for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = self._read_record(path, line_number, instance, json.loads(line))
                    examples = self._select(record)
                except (DataError, json.JSONDecodeError, KeyError, TypeError) as exc:
                    bundle.rejected.append({"path": str(path), "line": line_number, "reason": str(exc)})
                    continue
                for example in examples:
                    errors = self._example_errors(example)
                    if errors:
                        if self.config.missing_images == "error":
                            raise DataError(f"{example.id}: {'; '.join(errors)}")
                        bundle.rejected.append({"id": example.id, "path": str(path), "reason": "; ".join(errors)})
                    else:
                        getattr(bundle, split).append(example)
        return bundle

    def _read_record(self, path: Path, line: int, instance: str, row: JsonValue) -> TrajectoryRecord:
        if not isinstance(row, dict) or not isinstance(row.get("conversations"), list):
            raise DataError("Expected existing ShareGPT conversations")
        if not isinstance(row.get("system", ""), str) or not isinstance(row.get("images", []), list):
            raise DataError("system must be text and images must be a list")
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            raise DataError("metadata must be an object")
        if metadata.get("benchmark", self.scene) != self.scene or metadata.get("instance", instance) != instance:
            raise DataError("Scene or instance metadata does not match input path")
        images = row.get("images", [])
        if not all(isinstance(p, str) for p in images):
            raise DataError("images must contain path strings")
        image_index = 0
        bindings = []
        for conversation in row["conversations"]:
            if not isinstance(conversation, dict) or conversation.get("from") not in {"human", "gpt", "function_call", "observation"}:
                raise DataError("Unsupported ShareGPT conversation role")
            value = conversation.get("value")
            if not isinstance(value, str):
                raise DataError("ShareGPT value must be text")
            count = value.count("<image>")
            bindings.append(tuple((path.parent / p).resolve() for p in images[image_index:image_index + count]))
            image_index += count
        if image_index != len(images):
            raise DataError("Image placeholders do not match the images list")
        return TrajectoryRecord(path, line, instance, cast(TrajectoryRow, row), tuple(bindings))

    @staticmethod
    def _mapping(record: TrajectoryRecord) -> list[SourceMapping]:
        mapping = record.row.get("metadata", {}).get("sharegpt_conversion", {}).get("conversation_source_mapping")
        if not isinstance(mapping, list) or len(mapping) != len(record.row["conversations"]):
            raise DataError("Missing or inconsistent ShareGPT source mapping; cannot select original steps safely")
        if any(not isinstance(m, dict) or not isinstance(m.get("source_message_indices"), list) for m in mapping):
            raise DataError("Invalid source message mapping")
        return mapping

    def _select(self, record: TrajectoryRecord) -> list[TrainingExample]:
        conversations = record.row["conversations"]
        metadata = record.row.get("metadata", {})
        mapping = self._mapping(record)
        examples = []
        if self.scene in {"Single-view", "Multi-view"}:
            for step in metadata.get("steps", []):
                if step.get("status") not in self.config.accepted_statuses:
                    continue
                start, end = step["message_start"], step["message_end"]
                selected = tuple(i for i, m in enumerate(mapping)
                                 if m["source_message_indices"] and all(start <= n < end for n in m["source_message_indices"]))
                if len(selected) != 2 or [conversations[i]["from"] for i in selected] != ["human", "gpt"]:
                    raise DataError("An iterative step must map to one human request and one assistant script")
                if conversations[selected[-1]]["value"].strip():
                    examples.append(TrainingExample(record, selected, selected[-1]))
        else:
            chains = metadata.get("chains", [])
            if not chains:
                raise DataError("Interactive trajectories require explicit chain boundaries")
            for chain in chains:
                start, end = chain["message_start"], chain["message_end"]
                indices = [i for i, m in enumerate(mapping) if m["source_message_indices"] and
                           all(start <= n < end for n in m["source_message_indices"])]
                for m in mapping:
                    membership = [start <= n < end for n in m["source_message_indices"]]
                    if any(membership) and not all(membership):
                        raise DataError("ShareGPT conversion merged messages across independent chains")
                if not indices or conversations[indices[0]]["from"] != "human":
                    raise DataError("Each interactive chain must start with its own human request")
                assistant_indices = [i for i in indices if conversations[i]["from"] in {"gpt", "function_call"}
                                     and conversations[i]["value"].strip()]
                final = assistant_indices[-1] if assistant_indices else None
                for index in assistant_indices:
                    is_call = conversations[index]["from"] == "function_call" or bool(mapping[index].get("tool_call_ids"))
                    if is_call or (self.config.include_final_reply and index == final):
                        prefix = tuple(i for i in indices if i <= index)
                        examples.append(TrainingExample(record, prefix, index))
        return examples

    def _example_errors(self, example: TrainingExample) -> list[str]:
        errors = []
        if example.record.row.get("metadata", {}).get("images_complete") is False:
            errors.append("Exporter reports incomplete visual context")
        for index in example.indices:
            value = example.record.row["conversations"][index]["value"]
            if "[Image unavailable:" in value or "[REDACTED_IMAGE" in value:
                errors.append("Unrecoverable image placeholder in selected context")
            for path in example.record.images_by_conversation[index]:
                if path not in self._image_errors:
                    self._image_errors[path] = validate_image(path, self.config.verify_images)
                if self._image_errors[path]:
                    errors.append(self._image_errors[path])
        if example.record.images_by_conversation[example.target_index]:
            errors.append("Image-generation targets are not supported by this SFT task")
        return list(dict.fromkeys(errors))
