import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from pydantic import ValidationError

from harness_training.config import load_config, save_config
from harness_training.models.qwen import QwenModel
from harness_training.training.trainer import Trainer
from conftest import write_trajectory


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("filename", sorted(path.name for path in (ROOT / "configs").glob("*.yaml")))
def test_top_level_config_uses_only_common_defaults(tmp_path, filename):
    from hydra import compose, initialize_config_dir
    from omegaconf import DictConfig, OmegaConf

    source = ROOT / "configs" / filename
    raw = OmegaConf.load(source)
    assert isinstance(raw, DictConfig)
    groups = {"data", "template", "model", "method", "scene", "tuning", "quantization",
              "distributed", "tracking"}
    for entry in raw.defaults:
        assert entry == "_self_" or isinstance(entry, DictConfig) and set(entry).issubset(groups)
    assert raw.method.epochs > 0
    with initialize_config_dir(version_base="1.3", config_dir=str(source.parent)):
        expected = compose(config_name=filename)
    for group in groups:
        shutil.copytree(source.parent / group, tmp_path / group)
    isolated = tmp_path / filename
    isolated.write_text(source.read_text())
    with initialize_config_dir(version_base="1.3", config_dir=str(tmp_path)):
        composed = compose(config_name=filename)
    assert OmegaConf.to_container(composed, resolve=True) == OmegaConf.to_container(expected, resolve=True)
    if filename != "run.yaml":
        selected = load_config(isolated, ["method.epochs=3", "model.max_length=65536"])
        assert selected.method.epochs == 3
        assert selected.model.max_length == selected.template.max_length == 65536


def test_blender_config_keeps_single_gpu_and_wandb_overrides():
    config = load_config(ROOT / "configs/sft_blender_9b_fsdp.yaml", [
        "distributed=single", "tracking.entity=test-workspace", "tracking.project=test-project",
    ])
    assert config.method.epochs == 2
    assert config.distributed.strategy == "single"
    assert config.method.backend.fsdp is None
    assert config.method.backend.fsdp_config is None
    assert config.tracking.entity == "test-workspace"
    assert config.tracking.project == "test-project"


@pytest.mark.parametrize("task", ["view", "interaction"])
def test_disabled_tracking_preserves_mixed_task_settings(task):
    path = ROOT / f"configs/sft_{task}_mix_27b_fsdp.yaml"
    baseline = load_config(path)
    config = load_config(path, ["tracking=disabled"])
    assert config.tracking == baseline.tracking.model_copy(update={"mode": "disabled"})
    assert config.method.backend.report_to == ["none"]
    assert config.method.backend.run_name is None
    assert config.model == baseline.model


def test_model_preset_keeps_explicit_parallel_and_training_settings():
    path = ROOT / "configs/sft_view_mix_27b_fsdp.yaml"
    baseline = load_config(path)
    config = load_config(path, ["model=qwen35_9b"])
    assert config.model.path == Path.home() / "models/Qwen3.5-9B"
    assert config.model.max_length == config.template.max_length == 98304
    assert config.model.device_map is None
    assert config.method == baseline.method


@pytest.mark.parametrize("trial", [False, True])
def test_official_9b_config_selects_model_and_training_mode(trial):
    suffix = "_trial" if trial else ""
    config = load_config(ROOT / f"configs/sft_single_view_9b{suffix}.yaml")
    assert config.model.path == Path.home() / "models/Qwen3.5-9B"
    assert config.distributed.strategy == "ddp"
    assert config.model.quantization is not None
    assert config.model.quantization.bnb_4bit_quant_type == "nf4"
    assert config.method.backend.bf16 is True
    assert config.method.backend.loss_type == "chunked_nll"
    assert config.tracking.mode == "online"
    assert config.method.max_steps == (3 if trial else -1)
    assert config.method.backend.do_eval is not trial


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
        "distributed=single", "tracking=disabled",
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
        "distributed=single", "tracking=disabled",
        "data.validation_fraction=0", "data.test_fraction=0",
    ], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["samples"]["train"] == 2
