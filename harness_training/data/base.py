from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol
from harness_training.types import DataReport, Rejection
from .contract import Trajectory


@dataclass
class TrajectoryRecord:
    path: Path
    line: int
    instance: str
    row: Trajectory
    images_by_message: tuple[tuple[Path, ...], ...]


@dataclass(frozen=True)
class TrainingExample:
    """A view into an existing trajectory; no additional data export."""

    record: TrajectoryRecord
    indices: tuple[int, ...]
    target_index: int
    step_id: str

    @property
    def id(self) -> str:
        metadata = self.record.row.metadata
        return f"{metadata.scene}:{self.record.instance}:{metadata.session_id}:{self.step_id}"


@dataclass
class DatasetBundle:
    train: list[TrainingExample] = field(default_factory=list[TrainingExample])
    validation: list[TrainingExample] = field(default_factory=list[TrainingExample])
    test: list[TrainingExample] = field(default_factory=list[TrainingExample])
    assignments: dict[str, str] = field(default_factory=dict[str, str])
    rejected: list[Rejection] = field(default_factory=list[Rejection])
    source_hashes: dict[str, str] = field(default_factory=dict[str, str])

    def examples(self, split: Literal["train", "validation", "test"]) -> list[TrainingExample]:
        match split:
            case "train":
                return self.train
            case "validation":
                return self.validation
            case "test":
                return self.test

    def report(self) -> DataReport:
        return {
            "objects": len(self.assignments),
            "samples": {"train": len(self.train), "validation": len(self.validation), "test": len(self.test)},
            "rejected": len(self.rejected),
            "assignments": self.assignments,
        }


class PreparationWorkers(Protocol):
    rank: int
    world_size: int

    def gather[T](self, phase: str, value: T) -> list[T]: ...

    def progress(self, phase: str, total: int) -> AbstractContextManager[Callable[[], None]]: ...


class SerialPreparationWorkers:
    def __init__(self) -> None:
        self.rank = 0
        self.world_size = 1

    def gather[T](self, phase: str, value: T) -> list[T]:
        return [value]

    def progress(self, phase: str, total: int) -> AbstractContextManager[Callable[[], None]]:
        return nullcontext(lambda: None)


class BaseDataset(ABC):
    @abstractmethod
    def prepare(self, *, workers: PreparationWorkers | None = None) -> DatasetBundle:
        """Read existing records and select training targets in memory."""
