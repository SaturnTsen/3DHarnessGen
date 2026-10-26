import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness_training.config import TrainingConfig, load_config
from harness_training.training.tracking import tracking_session
from harness_training.training.trainer import Trainer
from conftest import read_wandb_records


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("scene", ["single_view", "multi_view", "active_visual", "full_3d_interaction"])
def test_scene_defaults_record_online_in_renamed_project(scene):
    config = load_config(ROOT / f"configs/sft_{scene}.yaml")
    assert config.tracking.mode == "online"
    assert config.tracking.project == "3DHarnessGen"
    assert config.method.backend.report_to == ["wandb"]


def enable_tracking(config):
    selected = load_config(ROOT / "configs/sft_single_view.yaml", ["tracking=wandb", "tracking.mode=offline"])
    config.tracking = selected.tracking
    config.tracking.settings.x_disable_stats = True
    config.method.backend.report_to = ["wandb"]
    config.method.backend.run_name = config.tracking.name


@pytest.mark.parametrize("mode, report_to", [("disabled", ["wandb"]), ("offline", ["none"]), ("disabled", "all")])
def test_tracking_requires_matching_backend(config, mode, report_to):
    payload = config.model_dump(mode="json")
    payload["tracking"]["mode"] = mode
    payload["method"]["backend"]["report_to"] = report_to
    with pytest.raises(ValidationError, match="report_to"):
        TrainingConfig.model_validate(payload)


def test_audit_does_not_start_wandb(config):
    import wandb

    enable_tracking(config)
    assert Trainer(config).audit()["samples"]["train"] == 2
    assert wandb.run is None
    assert not (config.experiment.output_dir / "wandb_run.json").exists()


def test_non_main_process_does_not_start_wandb(config):
    import wandb

    enable_tracking(config)
    with tracking_session(config, None, is_main_process=False):
        assert wandb.run is None
    assert not config.experiment.output_dir.exists()


def test_failure_finishes_run_and_restores_environment(config):
    import wandb

    enable_tracking(config)
    before = {key: os.environ.get(key) for key in ("WANDB_LOG_MODEL", "WANDB_WATCH")}
    with pytest.raises(RuntimeError, match="training failed"):
        with tracking_session(config, None, is_main_process=True):
            assert os.environ["WANDB_LOG_MODEL"] == "false"
            wandb.log({"train/global_step": 1, "train/loss": 1.5})
            raise RuntimeError("training failed")
    assert wandb.run is None
    assert {key: os.environ.get(key) for key in before} == before
    path, = (config.experiment.output_dir / "wandb").glob("offline-run-*/run-*.wandb")
    records = read_wandb_records(path)
    assert any(record.HasField("exit") and record.exit.exit_code != 0 for record in records)


def test_offline_tiny_training_logs_metrics_and_reuses_run_id(config, tiny_qwen_path):
    import wandb

    enable_tracking(config)
    config.model.path = tiny_qwen_path
    config.method.use_cpu = True
    config.method.max_steps = 1
    config.method.gradient_accumulation_steps = 1
    config.method.gradient_checkpointing = False
    config.method.save_steps = 1
    config.method.warmup_ratio = 0
    config.method.backend.eval_strategy = "no"
    config.method.backend.do_eval = False
    config.method.backend.disable_tqdm = True
    config.model.lora.rank = 2
    config.model.lora.alpha = 4
    config.model.lora.target_modules = ["q_proj", "v_proj"]
    output = config.experiment.output_dir
    # Explicitly check a resolved credential is excluded from the W&B config.
    config.method.backend.hub_token = "test-local-only-token"
    import torch

    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        Trainer(config).run()
        first = json.loads((output / "wandb_run.json").read_text())
        config.method.max_steps = 2
        config.method.resume_from = output / "checkpoints/checkpoint-1"
        Trainer(config).run()
    finally:
        torch.set_num_threads(threads)
    assert wandb.run is None
    assert json.loads((output / "wandb_run.json").read_text())["id"] == first["id"]
    paths = list((output / "wandb").glob("offline-run-*/run-*.wandb"))
    assert len(paths) == 2
    records = [record for path in paths for record in read_wandb_records(path)]
    history = [{item.key or ".".join(item.nested_key): json.loads(item.value_json) for item in record.history.item}
               for record in records if record.HasField("history")]
    assert any(item.get("train/loss", 0) > 0 and item.get("train/global_step") == 1 for item in history)
    assert any(item.get("train/global_step") == 2 for item in history)
    assert all("test-local-only-token" not in str(record) for record in records)
    assert not any(record.HasField("artifact") for record in records)


def test_resume_requires_saved_wandb_identity(config):
    enable_tracking(config)
    with pytest.raises(FileNotFoundError):
        with tracking_session(config, Path("checkpoint"), is_main_process=True):
            pytest.fail("Should not create a new W&B run when restoring a checkpoint")
