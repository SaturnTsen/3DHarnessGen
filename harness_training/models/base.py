from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
from harness_training.types import Processor

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedModel
from harness_training.config import ModelConfig


class BaseModel(ABC):
    config: ModelConfig

    @property
    @abstractmethod
    def processor(self) -> Processor:
        """The processor owned by this model implementation."""

    @abstractmethod
    def load(self) -> PreTrainedModel | PeftModel:
        """Load model weights and apply trainable-parameter settings."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model or adapter and the matching processor."""
