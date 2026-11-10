"""Training workflow: data inspection, preparation and fresh training runs."""

import argparse
import json
import os
from pathlib import Path

from harness_training.config import load_config
from harness_training.trainers.sft_trainer import SFTTrainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Training workflow",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="action", required=True)

    for action in ("audit", "prepare", "preview", "train", "check-runtime"):
        command = commands.add_parser(action, allow_abbrev=False)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. distributed=single")

    preview = commands.choices["preview"]
    preview.add_argument("--limit", type=int, default=1)
    preview.add_argument("--split",choices=("train", "validation", "test"), default="train")
    preview.add_argument("--render-only", action="store_true")

    args = parser.parse_args()
    
    if args.action == "preview" and args.limit <= 0:
        parser.error("--limit must be positive")
    
    config = load_config(args.config, args.overrides)
    
    match config.method.name:
        case "sft":
            trainer = SFTTrainer(config)
        case _:
            raise ValueError(f"Unsupported training method: {config.method.name}")
    
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
        case "check-runtime":
            trainer.model.validate_runtime()
            result = {"runtime_check": "passed", "weights_loaded": False}
        case _:
            raise ValueError(f"Unsupported training action: {args.action}")
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
