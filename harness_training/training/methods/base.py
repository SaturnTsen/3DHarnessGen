from abc import ABC, abstractmethod
from pathlib import Path
from harness_training.types import Metrics

from harness_training.data.base import DatasetBundle
from harness_training.models.base import BaseModel
from harness_training.templates.base import BaseTemplate


class BaseTrainingMethod(ABC):
    @abstractmethod
    def run(self, model: BaseModel, template: BaseTemplate, data: DatasetBundle,
            output_dir: Path, resume_from: Path | None) -> Metrics:
        """Execute a concrete method; methods may add their own data requirements."""
