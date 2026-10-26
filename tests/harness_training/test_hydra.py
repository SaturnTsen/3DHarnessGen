import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from harness_training.config import load_config, save_config
from harness_training.models.qwen import QwenModel
from harness_training.training.trainer import Trainer
from conftest import write_trajectory


ROOT = Path(__file__).resolve().parents[2]


def test_composition_paths_ignore_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config(ROOT / "configs/sft_single_view.yaml", [
        "scene=active_visual", "distributed=fsdp", "quantization=4bit", "method.learning_rate=0.001",
    ])
    assert config.experiment.scene == "ActiveVisual"
    assert config.experiment.output_dir == ROOT / "outputs/post_training/ActiveVisual/sft_v1"
    assert config.data.runs_root == ROOT / "3DHarnessBench/runs"
    assert config.model.path == Path.home() / "models/Qwen3.8-27B"
    assert config.method.learning_rate == .001
    assert config.method.backend.fsdp_config is not None
    assert config.method.backend.fsdp_config.version == 2
    assert config.model.quantization is not None
    assert config.model.quantization.use_gradient_checkpointing is True
    snapshot = tmp_path / "resolved.yaml"
    save_config(config, snapshot)
    assert load_config(snapshot) == config


def test_snapshot_preserves_yaml_boolean_like_strings(config, tmp_path):
    config.method.backend.eval_strategy = "no"
    config.method.backend.do_eval = False
    config.method.backend.include_num_input_tokens_seen = "no"
    path = tmp_path / "snapshot.yaml"
    save_config(config, path)
    assert load_config(path) == config


@pytest.mark.parametrize("overrides, message", [
    (["distributed=ddp", "model.device_map=balanced"], "device_map"),
    (["distributed=fsdp", "method.backend.fsdp_config.activation_checkpointing=true"], "checkpointing"),
    (["quantization=4bit", "tuning=full"], "LoRA"),
    (["quantization=4bit", "method.use_cpu=true"], "GPU"),
    (["distributed=fsdp", "quantization=4bit", "model.dtype=float32"], "storage"),
])
def test_incompatible_combinations_fail_before_loading(overrides, message):
    with pytest.raises(ValidationError, match=message):
        load_config(ROOT / "configs/sft_single_view.yaml", overrides)


def test_distributed_requires_explicit_launcher(config):
    config.distributed.strategy = "ddp"
    trainer = Trainer(config)
    with pytest.raises(ValueError, match="torchrun"):
        trainer.run()
    assert isinstance(trainer.model, QwenModel)
    assert trainer.model._processor is None


def test_native_hydra_multirun_keeps_outputs_separate(tmp_path):
    write_trajectory(tmp_path / "runs")
    result = subprocess.run([
        sys.executable, "-m", "harness_training.hydra_cli", "--multirun",
        "command.action=prepare", f"data.runs_root={tmp_path / 'runs'}",
        f"experiment.output_dir={tmp_path / 'output'}", "method.learning_rate=0.001,0.002",
        f"hydra.sweep.dir={tmp_path / 'sweep'}",
    ], cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    for index in [0, 1]:
        report = json.loads((tmp_path / f"output/sweep/job_{index}/data_report.json").read_text())
        assert report["samples"]["train"] == 2
        assert (tmp_path / f"sweep/{index}/.hydra/config.yaml").is_file()


def test_legacy_cli_accepts_hydra_overrides(tmp_path):
    write_trajectory(tmp_path / "runs")
    result = subprocess.run([
        sys.executable, "-m", "harness_training.cli", "audit", "--config",
        str(ROOT / "configs/sft_single_view.yaml"), f"data.runs_root={tmp_path / 'runs'}",
        "data.validation_fraction=0", "data.test_fraction=0",
    ], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["samples"]["train"] == 2
