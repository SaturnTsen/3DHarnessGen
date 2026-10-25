from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from harness_training.config import load_config
from harness_training.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-training on existing 3DHarnessBench v3 trajectories")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "prepare", "preview", "train"):
        entry = commands.add_parser(command)
        entry.add_argument("--config", type=Path, required=True)
        entry.add_argument("--limit", type=int, default=1, help="Number of preview samples")
        entry.add_argument("--split", choices=["train", "validation", "test"], default="train")
        entry.add_argument("--render-only", action="store_true", help="Preview without pixel/token encoding")
    smoke = commands.add_parser("smoke", help="检查真实数据和 Processor，并运行小模型训练测试")
    smoke.add_argument("--scene", choices=["Single-view", "Multi-view", "ActiveVisual", "Full3DInteraction"])
    smoke.add_argument("--instance", help="指定对象，需同时指定 --scene")
    smoke.add_argument("--include-unknown", action="store_true", default=True,
                       help="接受 UNKNOWN 步骤（默认启用，保留此参数以兼容旧命令）")
    smoke.add_argument("--limit", type=int, default=1, help="每个场景最多预览的样本数")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.command == "smoke":
        if args.instance and not args.scene:
            parser.error("--instance requires --scene")
        from harness_training.training.smoke import run_smoke

        run_smoke(scene=args.scene, instance=args.instance, include_unknown=args.include_unknown, limit=args.limit)
        return
    try:
        config = load_config(args.config)
        if "UNKNOWN" not in config.data.accepted_statuses:
            config.data.accepted_statuses.append("UNKNOWN")
        print("警告：本次运行接受 UNKNOWN 状态的步骤；这些步骤的执行状态尚未确认。", file=sys.stderr)
        trainer = Trainer(config)
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
