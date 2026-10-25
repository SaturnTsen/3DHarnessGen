"""Run bounded v3 data/Processor checks and the existing tiny-model tests."""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime
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


def run_smoke(*, scene: str | None, instance: str | None, include_unknown: bool = True, limit: int) -> None:
    root = Path(__file__).resolve().parents[2]
    (root / "temp").mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="smoke-post-training-", dir=root / "temp"))
    started = time.monotonic()

    def log(message: str) -> None:
        line = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with (output / "smoke.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    log(f"报告目录：{output}")
    log(f"开始冒烟测试：场景={scene or '全部'}，每场景预览上限={limit}，额外接受 UNKNOWN={include_unknown}")
    if include_unknown:
        log("警告：本次检查接受 UNKNOWN 状态的步骤；这些步骤的执行状态尚未确认。")
    log("[锁文件] 开始执行 pixi lock --check")
    subprocess.run(["pixi", "lock", "--check"], cwd=root, check=True)
    log(f"[锁文件] 检查通过，累计耗时 {time.monotonic() - started:.1f} 秒")

    failures: list[str] = []
    summaries: list[dict[str, object]] = []
    for scene in ([scene] if scene else CONFIGS):
        scene_started = time.monotonic()
        folder = output / scene
        folder.mkdir()
        config = load_config(root / "configs" / CONFIGS[scene])
        source = config.data.runs_root / scene / config.data.source_model
        log(f"[{scene}] 读取配置：{root / 'configs' / CONFIGS[scene]}")
        log(f"[{scene}] 查找轨迹：{source}")
        paths = sorted(source.glob("*/post_train_trajectory/trajectory.jsonl"))
        if config.data.instances is not None:
            paths = [path for path in paths if path.parent.parent.name in config.data.instances]
        if instance:
            paths = [path for path in paths if path.parent.parent.name == instance]
        log(f"[{scene}] 匹配到 {len(paths)} 个对象，将选择排序后的第一个对象")
        if not paths:
            failures.append(f"{scene}：未找到所选对象的轨迹")
            log(f"[{scene}] 失败：未找到轨迹")
            continue
        config.data.instances = [paths[0].parent.parent.name]
        # This bounded check puts the selected object in train for preview.
        config.data.validation_fraction = 0
        config.data.test_fraction = 0
        config.experiment.output_dir = folder
        if include_unknown and "UNKNOWN" not in config.data.accepted_statuses:
            config.data.accepted_statuses.append("UNKNOWN")
        save_config(config, folder / "config.yaml")
        log(f"[{scene}] 选中 {config.data.instances[0]}，接受状态：{config.data.accepted_statuses}")
        log(f"[{scene}] 轨迹文件：{paths[0]}，大小 {paths[0].stat().st_size} 字节")
        log(f"[{scene}] 开始检查契约、步骤和图片；本次将所选对象放入训练集合用于预览")
        trainer = Trainer(config)
        report = trainer.audit()
        write_json(folder / "audit.json", report)
        log(f"[{scene}] 数据检查完成：样本={report['samples']}，拒绝={report['rejected']}")
        for rejected in report["rejections"][:5]:
            log(f"[{scene}] 拒绝原因：{rejected['reason']}")
        if report["rejected"] > 5:
            log(f"[{scene}] 其余拒绝原因见 {folder / 'audit.json'}")
        summary: dict[str, object] = {
            "scene": scene, "instance": config.data.instances[0],
            "samples": report["samples"], "rejected": report["rejected"],
        }
        summaries.append(summary)
        if report["rejected"]:
            failures.append(f"{scene}：有 {report['rejected']} 条拒绝记录，详见 audit.json")
        if not report["samples"]["train"]:
            failures.append(f"{scene}：无可用样本；检查状态筛选与 audit.json")
            log(f"[{scene}] 无可用样本，跳过预览；耗时 {time.monotonic() - scene_started:.1f} 秒")
            continue
        log(f"[{scene}] 开始加载 Processor 并编码预览，最多 {limit} 条；不加载基础模型权重")
        preview = trainer.preview(limit)
        write_json(folder / "preview.json", preview)
        for index, item in enumerate(preview, 1):
            if "error" in item:
                log(f"[{scene}] 预览 {index}/{len(preview)} 失败：{item['error']}")
            else:
                log(f"[{scene}] 预览 {index}/{len(preview)}：id={item.get('id')}，"
                    f"图片={len(item.get('images', []))}，总 token={item.get('tokens')}，"
                    f"目标 token={item.get('target_tokens')}")
        valid = [item for item in preview if "error" not in item and item.get("target_tokens", 0) > 0]
        summary["previewed"] = len(valid)
        if len(valid) != len(preview) or not valid:
            failures.append(f"{scene}：预览编码失败或目标为空，详见 preview.json")
        log(f"[{scene}] 完成，耗时 {time.monotonic() - scene_started:.1f} 秒；详情：{folder}")

    log("[测试] 开始配置、图片、标签及小 Qwen 训练/保存/恢复测试；逐项实时输出")
    tests_started = time.monotonic()
    command = [
        "pixi", "run", "test-post-training", "-v", "-s", "-ra", "--durations=10",
        f"--basetemp={output / 'pytest'}", f"--junitxml={output / 'tests.xml'}",
    ]
    log("[测试] 命令：" + " ".join(command))
    with (output / "tests.log").open("w", encoding="utf-8") as stream:
        with subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as tests:
            assert tests.stdout is not None
            for line in tests.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            returncode = tests.wait()
    log(f"[测试] 结束：退出码={returncode}，耗时 {time.monotonic() - tests_started:.1f} 秒；日志：{output / 'tests.log'}")
    if returncode:
        failures.append(f"测试失败，退出码 {returncode}")
    elif any(int(suite.get("skipped", "0")) for suite in ET.parse(output / "tests.xml").iter("testsuite")):
        failures.append("存在跳过的测试，不能视为完整通过；检查本地 Processor 是否可用")
    write_json(output / "summary.json", {"passed": not failures, "scenes": summaries, "failures": failures})
    if failures:
        log("冒烟测试未通过：\n" + "\n".join(f"- {reason}" for reason in failures))
        log(f"总耗时 {time.monotonic() - started:.1f} 秒；汇总：{output / 'summary.json'}")
        raise SystemExit(1)
    log(f"冒烟测试通过，总耗时 {time.monotonic() - started:.1f} 秒。报告：{output / 'summary.json'}")

