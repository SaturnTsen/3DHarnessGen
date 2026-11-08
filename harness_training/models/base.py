from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from harness_training.types import Processor

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedModel
from harness_training.config import LongContextConfig, ModelConfig


class BaseModel(ABC):
    config: ModelConfig
    long_context: LongContextConfig | None

    @abstractmethod
    def validate_runtime(self) -> None:
        """Check training-only requirements without allocating model weights."""

    @property
    @abstractmethod
    def processor(self) -> Processor:
        """The processor owned by this model implementation."""

    @abstractmethod
    def load(self) -> PreTrainedModel | PeftModel:
        """Load model weights and apply trainable-parameter settings."""
