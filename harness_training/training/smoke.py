"""Run bounded v3 data/Processor checks and the existing tiny-model tests."""

from __future__ import annotations

import json
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from harness_training.config import load_config, save_config
from harness_training.training.trainer import Trainer


CONFIGS = {
    "Single-view": "sft_single_view.yaml",
    "Multi-view": "sft_multi_view.yaml",
    "ActiveVisual": "sft_active_visual.yaml",
    "Full3DInteraction": "sft_full_3d_interaction.yaml",
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_smoke(*, scene: str | None, instance: str | None, include_unknown: bool, limit: int) -> None:
    root = Path(__file__).resolve().parents[2]
    (root / "temp").mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="smoke-post-training-", dir=root / "temp"))
    print(f"报告目录：{output}", flush=True)
    subprocess.run(["pixi", "lock", "--check"], cwd=root, check=True)

    failures: list[str] = []
    summaries: list[dict[str, object]] = []
    for scene in ([scene] if scene else CONFIGS):
        folder = output / scene
        folder.mkdir()
        config = load_config(root / "configs" / CONFIGS[scene])
        source = config.data.runs_root / scene / config.data.source_model
        paths = sorted(source.glob("*/post_train_trajectory/trajectory.jsonl"))
        if config.data.instances is not None:
            paths = [path for path in paths if path.parent.parent.name in config.data.instances]
        if instance:
            paths = [path for path in paths if path.parent.parent.name == instance]
        if not paths:
            failures.append(f"{scene}：未找到所选对象的轨迹")
            continue
        config.data.instances = [paths[0].parent.parent.name]
        # This bounded check puts the selected object in train for preview.
        config.data.validation_fraction = 0
        config.data.test_fraction = 0
        config.experiment.output_dir = folder
        if include_unknown and "UNKNOWN" not in config.data.accepted_statuses:
            config.data.accepted_statuses.append("UNKNOWN")
        save_config(config, folder / "config.yaml")
        print(f"检查 {scene} / {config.data.instances[0]}，接受状态：{config.data.accepted_statuses}", flush=True)
        trainer = Trainer(config)
        report = trainer.audit()
        write_json(folder / "audit.json", report)
        summary: dict[str, object] = {
            "scene": scene, "instance": config.data.instances[0],
            "samples": report["samples"], "rejected": report["rejected"],
        }
        summaries.append(summary)
        if report["rejected"]:
            failures.append(f"{scene}：有 {report['rejected']} 条拒绝记录，详见 audit.json")
        if not report["samples"]["train"]:
            failures.append(f"{scene}：无可用样本；检查状态筛选与 audit.json")
            continue
        preview = trainer.preview(limit)
        write_json(folder / "preview.json", preview)
        valid = [item for item in preview if "error" not in item and item.get("target_tokens", 0) > 0]
        summary["previewed"] = len(valid)
        if len(valid) != len(preview) or not valid:
            failures.append(f"{scene}：预览编码失败或目标为空，详见 preview.json")
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    print("运行配置、图片、标签及小 Qwen 训练/保存/恢复测试……", flush=True)
    tests = subprocess.run([
        "pixi", "run", "test-post-training", "-q", "-ra",
        f"--basetemp={output / 'pytest'}", f"--junitxml={output / 'tests.xml'}",
    ], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (output / "tests.log").write_text(tests.stdout, encoding="utf-8")
    print(tests.stdout, end="", flush=True)
    if tests.returncode:
        failures.append(f"测试失败，退出码 {tests.returncode}")
    elif any(int(suite.get("skipped", "0")) for suite in ET.parse(output / "tests.xml").iter("testsuite")):
        failures.append("存在跳过的测试，不能视为完整通过；检查本地 Processor 是否可用")
    write_json(output / "summary.json", {"passed": not failures, "scenes": summaries, "failures": failures})
    if failures:
        print("冒烟测试未通过：\n" + "\n".join(f"- {reason}" for reason in failures))
        raise SystemExit(1)
    print(f"冒烟测试通过。报告：{output / 'summary.json'}")

