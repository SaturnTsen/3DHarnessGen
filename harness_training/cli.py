from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness_training.config import load_config
from harness_training.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-training on existing 3DHarnessBench ShareGPT trajectories")
    parser.add_argument("command", choices=["audit", "prepare", "preview", "train"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1, help="Number of preview samples")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--render-only", action="store_true", help="Preview without pixel/token encoding")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    try:
        trainer = Trainer(load_config(args.config))
        if args.command == "audit":
            result = trainer.audit()
        elif args.command == "prepare":
            data = trainer.prepare()
            trainer.save_report(data)
            result = data.report()
        elif args.command == "preview":
            result = trainer.preview(args.limit, split=args.split, encode=not args.render_only)
        else:
            result = trainer.run()
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
