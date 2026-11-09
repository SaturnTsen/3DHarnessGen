"""Training workflow: data inspection, preparation, training and resume checks."""

import argparse
import json
import os
from pathlib import Path

from harness_training.config import load_config
from harness_training.training.checkpoint import resolve_resume
from harness_training.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Training workflow", allow_abbrev=False)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("audit", "prepare", "preview", "train", "check-runtime", "check-resume", "smoke"):
        command = commands.add_parser(action, allow_abbrev=False)
        command.add_argument("--config", type=Path, required=True)
        if action in {"preview", "smoke"}:
            command.add_argument("--limit", type=int, default=1)
        if action == "preview":
            command.add_argument("--split", choices=("train", "validation", "test"), default="train")
            command.add_argument("--render-only", action="store_true")
        if action == "smoke":
            command.add_argument("--encode", action="store_true", help="Also check Processor encoding")
        command.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. distributed=single")
    args = parser.parse_args()
    if args.action in {"preview", "smoke"} and args.limit <= 0:
        parser.error("--limit must be positive")
    config = load_config(args.config, args.overrides)
    if args.action in {"train", "check-resume"}:
        config = resolve_resume(config)
    smoke_failed = False
    if args.action == "smoke":
        from harness_training.training.smoke import run_smoke

        result = run_smoke(config, limit=args.limit, encode=args.encode)
        smoke_failed = not result["passed"]
    else:
        trainer = Trainer(config)
        match args.action:
            case "audit":
                result = trainer.audit()
            case "prepare":
                data = trainer.prepare()
                trainer.save_report(data)
                result = data.report()
            case "preview":
                result = trainer.preview(args.limit, split=args.split, encode=not args.render_only)
            case "train":
                result = trainer.run()
            case "check-resume":
                result = trainer.check_resume()
            case "check-runtime":
                trainer.model.validate_runtime()
                result = {"runtime_check": "passed", "weights_loaded": False}
            case _:
                raise ValueError(f"Unsupported training action: {args.action}")
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if smoke_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
