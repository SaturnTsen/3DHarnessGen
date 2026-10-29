"""Resolve opt-in automatic checkpoint selection before launching training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from harness_training.cli import main as training_main
from harness_training.config import load_config
from harness_training.training.checkpoint import checkpoint_problem


def resolve_arguments(arguments: list[str]) -> list[str]:
    automatic = "method.resume_from=auto"
    if automatic not in arguments:
        return arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", choices=["train", "check-resume"])
    parser.add_argument("--config", type=Path, required=True)
    selected, overrides = parser.parse_known_args(arguments)
    if sum(item.startswith("method.resume_from=") for item in overrides) != 1:
        raise ValueError("Specify method.resume_from only once")
    config = load_config(selected.config, ["method.resume_from=null" if item == automatic else item
                                         for item in overrides])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if (world_size == 1) != (config.distributed.strategy == "single"):
        raise ValueError("Use distributed=single for one process; launch ddp/fsdp with torchrun")
    if config.method.backend.save_only_model:
        raise ValueError("Automatic resume requires save_only_model: false")
    output = config.experiment.output_dir
    required = [output / "manifest.json", output / "resolved_config.yaml"]
    if config.tracking.mode != "disabled":
        required.append(output / "wandb_run.json")
    if any(not file.is_file() for file in required):
        raise ValueError("Automatic resume requires the original manifest, resolved config and W&B state")
    candidates = sorted((path for path in (output / "checkpoints").glob("checkpoint-*")
                         if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()),
                        key=lambda path: int(path.name.removeprefix("checkpoint-")), reverse=True)
    for checkpoint in candidates:
        problem = checkpoint_problem(checkpoint, config, world_size)
        if problem is not None:
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"[resume] Skip {checkpoint.name}: {problem}", flush=True)
            continue
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[resume] Restore {checkpoint}", flush=True)
        return [f"method.resume_from={checkpoint}" if item == automatic else item for item in arguments]
    raise ValueError("No resumable checkpoint for this process count; training was not restarted")


def main() -> None:
    try:
        sys.argv[1:] = resolve_arguments(sys.argv[1:])
    except (ValueError, OSError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    training_main()


if __name__ == "__main__":
    main()
