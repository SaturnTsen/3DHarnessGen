"""Command arguments and process launch for benchmark clients."""

import argparse
import json
import os
from pathlib import Path

from harness_training.inference import build_harness_command, build_mmsi_command, checkpoint_model_name


def add_common_arguments(parser: argparse.ArgumentParser, benchmark_root: Path) -> None:
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--model-name", "--adapter-name", dest="model_name")
    parser.add_argument("--benchmark-root", type=Path, default=benchmark_root)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--output-root", type=Path,
                        help="Results are separated by model name beneath this directory")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running the benchmark")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a served checkpoint", allow_abbrev=False)
    commands = parser.add_subparsers(dest="benchmark", required=True)
    harness = commands.add_parser("harness", help="Evaluate with 3DHarnessBench", allow_abbrev=False)
    add_common_arguments(harness, Path.home() / "3DHarnessBench")
    harness.add_argument("--setting", choices=("single-view", "multi-view", "active-visual", "full-3d-interaction"),
                         default="single-view")
    harness.add_argument("--context-window", type=int, help="MCP input+output budget; match vLLM --max-model-len")
    mmsi = commands.add_parser("mmsi", help="Evaluate with MMSI-Bench", allow_abbrev=False)
    add_common_arguments(mmsi, Path.home() / "spatial_benchs/MMSI-Bench")
    mmsi.add_argument("--max-tokens", type=int, default=8192)
    mmsi.add_argument("--dataset", choices=("MMSI_Bench", "MMSI_Bench_Circular"), default="MMSI_Bench")

    args, passthrough = parser.parse_known_args()
    if "--" in passthrough:
        passthrough.remove("--")
    builder = build_harness_command if args.benchmark == "harness" else build_mmsi_command
    root, command = builder(args, passthrough)
    print(f"Client request model: {checkpoint_model_name(args.checkpoint_path, args.model_name)}", flush=True)
    if args.dry_run:
        print(json.dumps({"cwd": str(root), "argv": command}, ensure_ascii=False, indent=2))
        return
    os.chdir(root)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
