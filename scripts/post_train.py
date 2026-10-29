"""Resolve opt-in automatic checkpoint selection before launching training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from harness_training.cli import main as training_main
from harness_training.config import TrainingConfig, load_config


def checkpoint_problem(path: Path, config: TrainingConfig, world_size: int) -> str | None:
    state_path = path / "trainer_state.json"
    if not state_path.is_file():
        return "missing trainer_state.json"
    try:
        state = json.loads(state_path.read_text())
    except json.JSONDecodeError:
        return "incomplete trainer_state.json"
    if not isinstance(state, dict) or state.get("global_step") != int(path.name.split("-")[-1]):
        return "trainer state does not match checkpoint step"
    rng_files = {file.name for file in path.glob("rng_state*.pth")}
    expected_rng = ({"rng_state.pth"} if world_size == 1 else
                    {f"rng_state_{rank}.pth" for rank in range(world_size)})
    elastic = config.method.resume_policy == "elastic"
    if elastic:
        saved_world = len(rng_files)
        expected_rng = ({"rng_state.pth"} if saved_world == 1 else
                        {f"rng_state_{rank}.pth" for rank in range(saved_world)})
    if not rng_files or rng_files != expected_rng:
        return "RNG state missing or saved process count differs from this launch"
    required = [path / "scheduler.pt", *(path / name for name in expected_rng)]
    if config.method.backend.fp16:
        required.append(path / "scaler.pt")
    source_fsdp = (path / "pytorch_model_fsdp_0").is_dir() if elastic else config.distributed.strategy == "fsdp"
    manifest_path = path / "resume_manifest.json"
    if elastic and not manifest_path.is_file():
        manifest_path = path.parent.parent / "manifest.json"
    portable = elastic and (path / "portable_state.pt").is_file()
    if elastic and manifest_path.is_file():
        try:
            source_manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            return "incomplete resume manifest"
        portable = portable or source_manifest.get("config", {}).get("method", {}).get("resume_policy") == "elastic"
    if portable:
        required += [path / "portable_state.pt", path / "data_progress.json", path / "resume_manifest.json"]
        if (path / "data_progress.json").is_file():
            from harness_training.training.elastic import DataProgress

            try:
                progress = DataProgress.model_validate_json((path / "data_progress.json").read_text())
            except ValueError:
                return "incomplete data progress"
            if progress.global_step != state["global_step"]:
                return "data progress and checkpoint step differ"
    elif source_fsdp:
        required += [path / "pytorch_model_fsdp_0/.metadata", path / "optimizer_0/.metadata"]
    else:
        required.append(path / "optimizer.pt")
        weights = ("adapter_model.safetensors", "adapter_model.bin") if config.model.tuning == "lora" else (
            "model.safetensors", "pytorch_model.bin", "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
        if not any((path / name).is_file() for name in weights):
            return "missing model weights"
        if config.model.tuning == "lora":
            required.append(path / "adapter_config.json")
    if any(not file.is_file() or file.stat().st_size == 0 for file in required):
        return "missing or empty model/optimizer/scheduler state"
    return None


def resolve_arguments(arguments: list[str]) -> list[str]:
    automatic = "method.resume_from=auto"
    if automatic not in arguments:
        return arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", choices=["train"])
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
