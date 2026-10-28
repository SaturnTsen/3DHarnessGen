"""Native Hydra entrypoint, including configuration inspection and multirun."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf

from harness_training.cli import execute
from harness_training.config import StrictConfig, parse_config


class CommandConfig(StrictConfig):
    action: Literal["audit", "prepare", "preview", "train", "check-runtime"]
    limit: int
    split: Literal["train", "validation", "test"]
    render_only: bool


@hydra.main(version_base="1.3", config_path="../configs", config_name="run")
def run(composed: DictConfig) -> None:
    runtime = HydraConfig.get()
    command = CommandConfig.model_validate(OmegaConf.to_container(composed.command, resolve=True))
    if command.limit <= 0:
        raise ValueError("command.limit must be positive")
    directory = Path(next(source.path for source in runtime.runtime.config_sources
                          if source.provider == "main" and source.schema == "file"))
    training = OmegaConf.masked_copy(composed, [str(key) for key in composed if key != "command"])
    config = parse_config(training, directory)
    if runtime.mode == RunMode.MULTIRUN:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("Hydra multirun must not run independently inside torchrun workers")
        # Keep sweep jobs separate even when only a learning rate is varied.
        config.experiment.output_dir /= Path(runtime.sweep.dir).name + f"/job_{runtime.job.num}"
    execute(config, command.action, limit=command.limit, split=command.split,
            render_only=command.render_only)


def main() -> None:
    OmegaConf.register_new_resolver(
        "project_path", lambda relative: str((Path(__file__).resolve().parents[1] / relative).resolve()),
        replace=True,
    )
    run()


if __name__ == "__main__":
    main()
