from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from harness_training.config import ModelConfig


class BaseModel(ABC):
    config: ModelConfig

    @property
    @abstractmethod
    def processor(self) -> Any:
        """The processor owned by this model implementation."""

    @abstractmethod
    def load(self) -> Any:
        """Load model weights and apply trainable-parameter settings."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model or adapter and the matching processor."""
