"""Launch benchmark clients for a named post-training checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys


def checkpoint_model_name(checkpoint_path: Path, model_name: str | None = None) -> str:
    checkpoint = checkpoint_path.expanduser().resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")
    run = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
    name = model_name or f"{run.name}-{checkpoint.name}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Model name must contain only letters, digits, dots, underscores and hyphens")
    return name


def checkpoint_snapshot(checkpoint_path: Path) -> dict:
    from pydantic import RootModel
    from pydantic_yaml import parse_yaml_raw_as

    checkpoint = checkpoint_path.expanduser().resolve()
    run = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
    source = run / "resolved_config.yaml"
    return parse_yaml_raw_as(RootModel[dict], source.read_text()).root if source.is_file() else {}


def common_parser(description: str, benchmark_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--model-name", "--adapter-name", dest="model_name")
    parser.add_argument("--benchmark-root", type=Path, default=benchmark_root)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--output-root", type=Path,
                        help="Results are separated by model name beneath this directory")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running the benchmark")
    return parser


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


def parse_passthrough(parser: argparse.ArgumentParser) -> tuple[argparse.Namespace, list[str]]:
    argv = sys.argv[1:]
    boundary = argv.index("--") if "--" in argv else len(argv)
    args, extra = parser.parse_known_args(argv[:boundary])
    return args, extra + argv[boundary + 1:]


def build_harness_command(args: argparse.Namespace, passthrough: list[str]) -> tuple[Path, list[str]]:
    root, output, name, run_name = client_paths(args, "pyproject.toml")
    reject_options(passthrough, {"--agent", "--output-dir", "--base-url", "--model-name", "--context-window"})
    if args.setting in ("single-view", "multi-view") and args.context_window is not None:
        raise ValueError("--context-window is for MCP settings; set the service limit using --max-model-len")
    if args.setting in ("active-visual", "full-3d-interaction") and args.context_window is None:
        snapshot = checkpoint_snapshot(args.checkpoint_path)
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


def launch(parser: argparse.ArgumentParser, builder) -> None:
    args, passthrough = parse_passthrough(parser)
    try:
        root, command = builder(args, passthrough)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Client request model: {checkpoint_model_name(args.checkpoint_path, args.model_name)}", flush=True)
    if args.dry_run:
        print(json.dumps({"cwd": str(root), "argv": command}, ensure_ascii=False, indent=2))
        return
    os.chdir(root)
    os.execvp(command[0], command)


def main_harness() -> None:
    parser = common_parser("Run 3DHarnessBench with a checkpoint served by vLLM", Path.home() / "3DHarnessBench")
    parser.add_argument("--setting", choices=("single-view", "multi-view", "active-visual", "full-3d-interaction"),
                        default="single-view")
    parser.add_argument("--context-window", type=int, help="MCP input+output budget; match vLLM --max-model-len")
    launch(parser, build_harness_command)


def main_mmsi() -> None:
    parser = common_parser("Evaluate a served checkpoint with MMSI-Bench", Path.home() / "spatial_benchs/MMSI-Bench")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--dataset", choices=("MMSI_Bench", "MMSI_Bench_Circular"), default="MMSI_Bench")
    launch(parser, build_mmsi_command)
