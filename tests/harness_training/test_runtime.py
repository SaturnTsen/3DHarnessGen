"""Exercise public dispatch and method boundaries without model weights or training."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from harness_training.config import MethodConfig, SFTMethodConfig, TrainingConfig, load_config, save_config
from harness_training.training.methods.base import BaseTrainingMethod
from harness_training.templates.base import BaseTemplate
from harness_training.training.smoke import run_smoke
from harness_training.training.trainer import Trainer
from conftest import write_trajectory
from test_resume_precheck import saved_resume
from test_serve_checkpoint import serving_checkpoint


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("explicit", [False, True])
def test_smoke_is_bounded_uses_configured_sources_and_does_not_mutate_config(config, explicit):
    if explicit:
        config.data.files = list(config.data.runs_root.rglob("trajectory.jsonl"))
    before = config.model_dump(mode="json")
    result = run_smoke(config, limit=1)
    assert result["passed"]
    assert result["checked_records"] == len(result["samples"]) == 1
    assert not result["processor_checked"] and not result["training_run"] and not result["benchmark_run"]
    assert config.model_dump(mode="json") == before
    report = Path(result["report_dir"])
    assert load_config(report / "resolved_config.yaml") == config
    assert not (report / "tests.log").exists()
    assert not config.experiment.output_dir.exists()


def test_smoke_reports_bad_first_record_without_scanning_entire_file(config):
    path = next(config.data.runs_root.rglob("trajectory.jsonl"))
    path.write_text('{invalid}\n' + path.read_text())
    result = run_smoke(config, limit=1)
    assert not result["passed"]
    assert result["checked_records"] == 1
    assert result["samples"][0]["line"] == 1
    assert "error" in result["samples"][0]


def test_cli_does_not_expand_the_configured_status_filter(config, tmp_path):
    config.data.accepted_statuses = ["OK"]
    source = next(config.data.runs_root.rglob("trajectory.jsonl"))
    record = json.loads(source.read_text())
    for step in record["metadata"]["steps"]:
        step["status"] = "UNKNOWN"
    source.write_text(json.dumps(record) + "\n")
    snapshot = tmp_path / "config.yaml"
    save_config(config, snapshot)
    result = subprocess.run([sys.executable, "-m", "harness_training.cli", "audit", "--config", str(snapshot)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["samples"]["train"] == 0


@pytest.mark.parametrize("entry", ["cli", "hydra_cli"])
def test_all_training_entries_resolve_auto_checkpoint_before_checking_resume(saved_resume, tmp_path, entry):
    config, _ = saved_resume
    config.method.resume_from = "auto"
    snapshot = tmp_path / "resume.yaml"
    save_config(config, snapshot)
    if entry == "cli":
        command = ["check-resume", "--config", str(snapshot)]
    else:
        with snapshot.open("a") as stream:
            stream.write("\ncommand:\n  action: check-resume\n  limit: 1\n  split: train\n  render_only: false\n")
        command = ["--config-dir", str(tmp_path), "--config-name", "resume",
                   f"hydra.run.dir={tmp_path / 'hydra'}"]
    result = subprocess.run([sys.executable, "-m", f"harness_training.{entry}", *command],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Restore " in result.stdout and '"status": "passed"' in result.stdout
    assert '"weights_loaded": false' in result.stdout


class RecordingMethod(BaseTrainingMethod):
    def check_resume(self, config, world_size, *, workers=None):
        raise AssertionError("A new run must not check a checkpoint")

    def verify_resume(self, config, resume, manifest):
        raise AssertionError("A new run must not verify a checkpoint")

    def run(self, model, template, data, output_dir, resume_from, *, context):
        assert model.model is None
        assert resume_from is None and context.resume is None
        assert context.manifest["targets"]["train"] == [example.id for example in data.train]
        assert context.config.experiment.output_dir == output_dir
        return {"recorded_samples": len(data.train)}


class NoEncodingTemplate(BaseTemplate):
    def render(self, example):
        raise AssertionError("The recording method must not render inputs")

    def encode(self, example):
        raise AssertionError("The recording method must not encode inputs")


class PreparedTrainer(Trainer):
    def prepare(self, *, encode=False, resume_targets=None):
        return self.dataset.prepare()

    @property
    def template(self):
        # The recording method only checks dispatch and never encodes a model input.
        return NoEncodingTemplate()


def test_trainer_passes_run_context_without_injecting_sft_state(config):
    config.method.use_cpu = True
    trainer = PreparedTrainer(config)
    trainer.method = RecordingMethod()
    result = trainer.run()
    assert result == {"recorded_samples": 2}
    assert not hasattr(trainer.method, "resume_manifest")
    assert not hasattr(trainer.method, "resume_progress")
    assert trainer.model.model is None


def test_method_config_separates_shared_contract_from_sft_fields(config):
    assert "backend" not in MethodConfig.model_fields
    assert "trainer" not in MethodConfig.model_fields
    assert isinstance(config.method, SFTMethodConfig)
    config.method.validate_training(config.model, config.distributed, config.tracking)
    assert TrainingConfig.model_validate(config.model_dump()).model_dump() == config.model_dump()


def test_pixi_and_installed_entrypoints_use_the_package_cli():
    import importlib.metadata
    import tomllib

    manifest = tomllib.loads((ROOT / "pixi.toml").read_text())
    assert manifest["tasks"]["post-train"] == "python -m harness_training.cli"
    assert manifest["tasks"]["post-train-distributed"].endswith("-m harness_training.cli")
    entry = next(item for item in importlib.metadata.distribution("3DHarnessGen").entry_points
                 if item.name == "post-train")
    assert entry.value == "harness_training.cli:main"


def test_hydra_adapter_serves_through_the_same_package_cli(serving_checkpoint, tmp_path):
    _, checkpoint = serving_checkpoint
    args = json.dumps(["--checkpoint-path", str(checkpoint), "--dry-run", "--api-key", "dummy-test-key"])
    result = subprocess.run([sys.executable, "-m", "harness_training.hydra_cli", "command.action=serve",
                             f"+command.args={args}", f"hydra.run.dir={tmp_path / 'hydra'}"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Client request model:" in result.stdout
    assert "dummy-test-key" not in result.stdout
