from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrajectoryRecord:
    path: Path
    line: int
    instance: str
    row: dict[str, Any]
    images_by_conversation: tuple[tuple[Path, ...], ...]


@dataclass(frozen=True)
class TrainingExample:
    """A view into an existing trajectory; no additional data export."""

    record: TrajectoryRecord
    indices: tuple[int, ...]
    target_index: int

    @property
    def id(self) -> str:
        return f"{self.record.instance}:{self.record.line}:{self.target_index}"


@dataclass
class DatasetBundle:
    train: list[TrainingExample] = field(default_factory=list)
    validation: list[TrainingExample] = field(default_factory=list)
    test: list[TrainingExample] = field(default_factory=list)
    assignments: dict[str, str] = field(default_factory=dict)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    source_hashes: dict[str, str] = field(default_factory=dict)

    def report(self) -> dict[str, Any]:
        return {"objects": len(self.assignments), "samples": {name: len(getattr(self, name))
                for name in ("train", "validation", "test")}, "rejected": len(self.rejected),
                "assignments": self.assignments}


class BaseDataset(ABC):
    @abstractmethod
    def prepare(self) -> DatasetBundle:
        """Read existing records and select training targets in memory."""
