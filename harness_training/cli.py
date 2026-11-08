"""Public CLI. Compose configuration once, then dispatch to package services."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Literal

from harness_training.config import TrainingConfig, load_config
from harness_training.training.checkpoint import resolve_resume
from harness_training.training.trainer import Trainer


def execute(config: TrainingConfig, command: str, *, limit: int = 1,
            split: Literal["train", "validation", "test"] = "train", render_only: bool = False,
            encode: bool = False):
    if command in {"train", "check-resume"}:
        config = resolve_resume(config)
    elif config.method.resume_from == "auto":
        raise ValueError("Automatic resume is supported only for train and check-resume")
    smoke_failed = False
    if command == "smoke":
        from harness_training.training.smoke import run_smoke

        result = run_smoke(config, limit=limit, encode=encode)
        smoke_failed = not result["passed"]
    else:
        trainer = Trainer(config)
        if command == "check-runtime":
            trainer.model.validate_runtime()
            result = {"runtime_check": "passed", "weights_loaded": False}
        elif command == "check-resume":
            result = trainer.check_resume()
        elif command == "audit":
            result = trainer.audit()
        elif command == "prepare":
            data = trainer.prepare()
            trainer.save_report(data)
            result = data.report()
        elif command == "preview":
            result = trainer.preview(limit, split=split, encode=not render_only)
        elif command == "train":
            result = trainer.run()
        else:
            raise ValueError(f"Unsupported command: {command}")
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if smoke_failed:
        raise ValueError("Smoke checks failed; see the reported sample errors")
    return result


def main(argv: list[str] | None = None) -> None:
    from harness_training import inference, serving
    from harness_training.data.contract import main as validate_trajectories

    parser = argparse.ArgumentParser(description="3DHarnessGen experiments", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "prepare", "preview", "train", "check-runtime", "check-resume", "smoke"):
        entry = commands.add_parser(command, allow_abbrev=False)
        entry.add_argument("--config", type=Path, required=True)
        if command in {"preview", "smoke"}:
            entry.add_argument("--limit", type=int, default=1)
        if command == "preview":
            entry.add_argument("--split", choices=["train", "validation", "test"], default="train")
            entry.add_argument("--render-only", action="store_true")
        if command == "smoke":
            entry.add_argument("--encode", action="store_true", help="Also check Processor encoding; never train")
        entry.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. distributed=single")
    serve = commands.add_parser("serve", help="Serve a checkpoint with vLLM", allow_abbrev=False)
    serving.add_arguments(serve)
    evaluate = commands.add_parser("eval", help="Evaluate a served checkpoint", allow_abbrev=False)
    inference.add_arguments(evaluate.add_subparsers(dest="benchmark", required=True))
    validate = commands.add_parser("validate-trajectory", help="Validate v3 JSON/JSONL", allow_abbrev=False)
    validate.add_argument("paths", nargs="+", type=Path)
    arguments = list(sys.argv[1:] if argv is None else argv)
    boundary = arguments.index("--") if "--" in arguments else len(arguments)
    args, extra = parser.parse_known_args(arguments[:boundary])
    extra += arguments[boundary + 1:]
    if extra and args.command not in {"serve", "eval"}:
        parser.error(f"Unrecognized arguments: {' '.join(extra)}")
    try:
        if args.command == "serve":
            serving.serve(args, extra)
        elif args.command == "eval":
            inference.launch(args, extra)
        elif args.command == "validate-trajectory":
            validate_trajectories(args.paths)
        else:
            if getattr(args, "limit", 1) <= 0:
                raise ValueError("--limit must be positive")
            config = load_config(args.config, args.overrides)
            execute(config, args.command, limit=getattr(args, "limit", 1),
                    split=getattr(args, "split", "train"), render_only=getattr(args, "render_only", False),
                    encode=getattr(args, "encode", False))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
