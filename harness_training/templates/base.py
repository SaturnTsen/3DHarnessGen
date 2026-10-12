from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    @abstractmethod
    def render(self, example: TrainingExample) -> RenderedExample:
        """Convert existing ShareGPT to the target model's chat format."""

    @abstractmethod
    def encode(self, example: TrainingExample) -> dict[str, Any]:
        """Create model inputs and labels for the selected reply only."""
