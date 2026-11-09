"""Launch benchmark clients for a named post-training checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from harness_training.checkpoints import Checkpoint


def checkpoint_model_name(checkpoint_path: Path, model_name: str | None = None) -> str:
    checkpoint = checkpoint_path.expanduser().resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")
    run = Checkpoint(checkpoint).run_dir
    name = model_name or f"{run.name}-{checkpoint.name}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Model name must contain only letters, digits, dots, underscores and hyphens")
    return name


def client_paths(args: argparse.Namespace, manifest: str) -> tuple[Path, Path, str, str]:
    root = args.benchmark_root.expanduser().resolve()
    if not (root / manifest).is_file():
        raise ValueError(f"Benchmark manifest is unavailable: {root / manifest}")
    name = checkpoint_model_name(args.checkpoint_path, args.model_name)
    output = (args.output_root or root / "outputs/checkpoints").expanduser().resolve()
    return root, output, name, checkpoint_model_name(args.checkpoint_path)


def reject_options(arguments: list[str], managed: set[str]) -> None:
    conflicts = [argument for argument in arguments if argument.split("=", 1)[0] in managed]
    if conflicts:
        raise ValueError(f"Launcher-managed options cannot be passed through: {conflicts}")


def build_harness_command(args: argparse.Namespace, passthrough: list[str]) -> tuple[Path, list[str]]:
    root, output, name, run_name = client_paths(args, "pyproject.toml")
    reject_options(passthrough, {"--agent", "--output-dir", "--base-url", "--model-name", "--context-window"})
    if args.setting in ("single-view", "multi-view") and args.context_window is not None:
        raise ValueError("--context-window is for MCP settings; set the service limit using --max-model-len")
    if args.setting in ("active-visual", "full-3d-interaction") and args.context_window is None:
        snapshot = Checkpoint(args.checkpoint_path).snapshot()
        args.context_window = snapshot.get("template", {}).get("max_length")
        if args.context_window is None:
            raise ValueError("No training length found; MCP requires --context-window matching vLLM --max-model-len")
    command = ["pixi", "run", "--manifest-path", str(root / "pyproject.toml"), args.setting,
               "--agent", "qwen-checkpoint-local", "--model-name", name,
               "--base-url", args.base_url, "--output-dir", str(output / run_name)]
    if args.context_window is not None:
        command += ["--context-window", str(args.context_window)]
    return root, command + passthrough


def build_mmsi_command(args: argparse.Namespace, passthrough: list[str]) -> tuple[Path, list[str]]:
    root, output, name, run_name = client_paths(args, "pixi.toml")
    reject_options(passthrough, {"--config", "--model", "--data", "--work-dir"})
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    config = {
        "model": {run_name: {"class": "Qwen38LocalAPI", "model": name, "api_base": args.base_url,
                         "max_tokens": args.max_tokens, "timeout": 1800, "retry": 5, "verbose": False}},
        "data": {args.dataset: {"class": "MMSIBenchDataset" if args.dataset == "MMSI_Bench" else "MMSIBenchCircular",
                                "dataset": args.dataset}},
    }
    content = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    configs = Path(__file__).resolve().parents[1] / "temp/benchmark_configs"
    configs.mkdir(parents=True, exist_ok=True)
    path = configs / f"mmsi-{hashlib.sha256(content.encode()).hexdigest()}.json"
    path.write_text(content)
    command = ["pixi", "run", "--manifest-path", str(root / "pixi.toml"), "python", "run.py",
               "--config", str(path), "--work-dir", str(output), "--api-nproc", "4",
               "--judge", "exact_matching", "--reuse"]
    return root, command + passthrough
