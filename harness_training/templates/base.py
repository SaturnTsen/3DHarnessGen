from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from harness_training.types import TensorInputs

from harness_training.config import TemplateConfig
from harness_training.data.base import TrainingExample


class TemplateError(ValueError):
    pass


class SampleTooLong(TemplateError):
    pass


@dataclass
class RenderedExample:
    prompt: str
    text: str
    images: tuple[Path, ...]


class BaseTemplate(ABC):
    config: TemplateConfig

    @abstractmethod
    def render(self, example: TrainingExample) -> RenderedExample:
        """Convert existing v3 to the target model's chat format."""

    @abstractmethod
    def encode(self, example: TrainingExample) -> TensorInputs:
        """Create model inputs and labels for the selected reply only."""
