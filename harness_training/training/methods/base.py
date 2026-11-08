from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from harness_training.types import Manifest, Metrics

from harness_training.config import TrainingConfig
from harness_training.data.base import DatasetBundle
from harness_training.models.base import BaseModel
from harness_training.templates.base import BaseTemplate

if TYPE_CHECKING:
    from harness_training.training.preflight import PreflightWorkers


@dataclass
class ResumeState:
    manifest: Manifest
    report: dict


@dataclass
class TrainingRun:
    config: TrainingConfig
    manifest: Manifest
    resume: ResumeState | None


class BaseTrainingMethod(ABC):
    @abstractmethod
    def check_resume(self, config: TrainingConfig, world_size: int, *,
                     workers: PreflightWorkers | None = None) -> ResumeState:
        """Check a checkpoint without loading weights; retain method-owned state."""

    @abstractmethod
    def verify_resume(self, config: TrainingConfig, resume: ResumeState, manifest: Manifest) -> None:
        """Verify the prepared inputs against the saved method state."""

    @abstractmethod
    def run(self, model: BaseModel, template: BaseTemplate, data: DatasetBundle,
            output_dir: Path, resume_from: Path | None, *, context: TrainingRun) -> Metrics:
        """Execute a concrete method; methods may add their own data requirements."""
