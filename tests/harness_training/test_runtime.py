"""Exercise public dispatch and method boundaries without model weights or training."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from harness_training.config import MethodConfig, SFTMethodConfig, TrainingConfig, load_config, save_config
from harness_training.models.qwen import QwenModel
from harness_training.trainers.sft_trainer import SFTTrainer
from test_serve_checkpoint import serving_checkpoint


ROOT = Path(__file__).resolve().parents[2]


def test_script_does_not_expand_the_configured_status_filter(config, tmp_path):
    config.data.accepted_statuses = ["OK"]
    source = next(config.data.runs_root.rglob("trajectory.jsonl"))
    record = json.loads(source.read_text())
    for step in record["metadata"]["steps"]:
        step["status"] = "UNKNOWN"
    source.write_text(json.dumps(record) + "\n")
    snapshot = tmp_path / "config.yaml"
    save_config(config, snapshot)
    result = subprocess.run([sys.executable, "-m", "harness_training.cli.training", "audit", "--config", str(snapshot)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["samples"]["train"] == 0


def test_script_rejects_removed_resume_action(tmp_path):
    result = subprocess.run([sys.executable, "-m", "harness_training.cli.training", "check-resume"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_concrete_trainer_owns_public_workflow(config):
    from harness_training.trainers.base import BaseTrainer

    trainer = SFTTrainer(config)
    assert isinstance(trainer, BaseTrainer)
    assert trainer.audit()["samples"]["train"] == 2
    assert trainer.backend is None
    assert isinstance(trainer.model, QwenModel)
    assert trainer.model.model is None
    assert trainer.model._processor is None


@pytest.mark.parametrize('artifact', ['final', 'checkpoints/checkpoint-1'])
def test_new_training_rejects_existing_outputs_before_loading_weights(config, artifact):
    config.method.use_cpu = True
    path = config.experiment.output_dir / artifact
    path.mkdir(parents=True)
    trainer = SFTTrainer(config)
    with pytest.raises(ValueError, match='choose another output_dir'):
        trainer.run()
    assert isinstance(trainer.model, QwenModel)
    assert trainer.model.model is None


def test_method_config_separates_shared_contract_from_sft_fields(config):
    assert "backend" not in MethodConfig.model_fields
    assert "trainer" not in MethodConfig.model_fields
    assert isinstance(config.method, SFTMethodConfig)
    config.method.validate_training(config.model, config.distributed, config.tracking)
    assert TrainingConfig.model_validate(config.model_dump()).model_dump() == config.model_dump()


def test_pixi_tasks_run_workflow_modules_directly():
    import importlib.metadata
    import tomllib

    manifest = tomllib.loads((ROOT / "pixi.toml").read_text())
    for task, module in (("training", "training"), ("serve", "serve"),
                         ("validate-trajectory", "validate")):
        assert manifest["tasks"][task] == f"python -m harness_training.cli.{module}"
    assert not {"audit", "prepare", "preview", "train", "check-runtime", "check-resume", "smoke"} & manifest["tasks"].keys()
    assert "post-train" not in manifest["tasks"]
    assert not (ROOT / "harness_training/cli.py").exists()
    package = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "scripts" not in package["project"]
    assert not importlib.metadata.distribution("3DHarnessGen").entry_points


def test_script_serves_checkpoint(serving_checkpoint, tmp_path):
    _, checkpoint = serving_checkpoint
    result = subprocess.run([sys.executable, "-m", "harness_training.cli.serve",
                             "--checkpoint-path", str(checkpoint), "--dry-run", "--api-key", "dummy-test-key"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Client request model:" in result.stdout
    assert "dummy-test-key" not in result.stdout


def test_evaluation_requires_samples_before_loading_weights(config, processor):
    config.method.use_cpu = True
    config.method.backend.eval_strategy = "steps"
    config.method.backend.do_eval = True
    trainer = SFTTrainer(config)
    with pytest.raises(ValueError, match="nonempty validation split"):
        trainer.run()
    assert isinstance(trainer.model, QwenModel)
    assert trainer.model.model is None
    assert trainer.backend is None
