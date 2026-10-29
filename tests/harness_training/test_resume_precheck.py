"""Resume failures must be actionable and precede Processor/image encoding."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from harness_training.config import save_config
from harness_training.training.elastic import validate_manifest
from harness_training.training.manifest import ResumeMismatch, differences
from harness_training.training.trainer import Trainer
from test_auto_resume import checkpoint_files
from test_elastic_resume import elastic_qwen_path, run_training


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def saved_resume(config, tmp_path):
    config.data.accepted_statuses.append("UNKNOWN")
    config.method.use_cpu = True
    config.model.path = tmp_path / "model"
    config.model.path.mkdir()
    (config.model.path / "config.json").write_text('{"model_type":"qwen3_5"}')
    trainer = Trainer(config)
    prior = trainer._manifest(trainer.prepare())
    checkpoint = config.experiment.output_dir / "checkpoints/checkpoint-1"
    checkpoint_files(checkpoint)
    manifest = config.experiment.output_dir / "manifest.json"
    manifest.write_text(json.dumps(prior))
    save_config(config, config.experiment.output_dir / "resolved_config.yaml")
    config.method.resume_from = checkpoint
    config.method.resume_policy = "elastic"
    return config, prior


@pytest.mark.parametrize("change", ["trajectory", "image", "model", "versions", "length", "batch"])
def test_failures_are_before_preflight_and_do_not_overwrite_run(saved_resume, change, capsys):
    config, prior = saved_resume
    output = config.experiment.output_dir
    path = output / "manifest.json"
    if change == "trajectory":
        selected = next(config.data.runs_root.rglob("trajectory.jsonl"))
        selected.write_text(selected.read_text() + "\n")
        expected = str(selected)
    elif change == "image":
        selected = next(config.data.runs_root.rglob("reference.png"))
        selected.write_bytes(selected.read_bytes() + b"changed")
        expected = str(selected)
    elif change == "model":
        selected = config.model.path / "config.json"
        selected.write_text('{"model_type":"changed"}')
        expected = str(selected)
    elif change == "versions":
        prior["versions"]["torch"] = "old-version"
        path.write_text(json.dumps(prior))
        expected = "versions.torch"
    elif change == "length":
        config.template.max_length = 128
        expected = "config.template.max_length"
    else:
        config.method.batch_size = 2
        expected = "Keep global batch size"
    original = path.read_bytes()
    trainer = Trainer(config)
    with pytest.raises(ValueError) as exc:
        trainer.run()
    assert expected in str(exc.value)
    assert "[preflight]" not in capsys.readouterr().out
    assert trainer._template is None
    assert trainer.model.model is None
    assert trainer.model._processor is None
    assert path.read_bytes() == original
    report = json.loads((ROOT / "temp/resume_check.json").read_text())
    assert report["status"] == "failed" and report["processor_loaded"] is False


def test_cli_check_resume_and_auto_load_neither_processor_nor_weights(saved_resume, tmp_path):
    config, _ = saved_resume
    path = tmp_path / "config.yaml"
    save_config(config, path)
    result = subprocess.run([sys.executable, "-m", "scripts.post_train", "check-resume", "--config", str(path),
                             "method.resume_from=auto"], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"status": "passed"' in result.stdout
    assert '"processor_loaded": false' in result.stdout
    assert '"tensor_load_verification": "pending"' in result.stdout
    assert "[preflight]" not in result.stdout


def test_increasing_length_keeps_checkpoint_targets(saved_resume):
    config, prior = saved_resume
    config.model.max_length *= 2
    config.template.max_length *= 2
    report = Trainer(config).check_resume()
    assert report["status"] == "passed"
    trainer = Trainer(config)
    # A saved run may have rejected an overlength target in its original manifest.
    original = deepcopy(prior["targets"])
    original["train"] = original["train"][:1]
    bundle = trainer.prepare(resume_targets=original)
    assert [example.id for example in bundle.train] == original["train"]
    assert trainer._template is None


def test_exact_differences_redact_credentials():
    result = differences({"method": {"backend": {"hub_token": "old-secret"}}},
                         {"method": {"backend": {"hub_token": "new-secret"}}}, "config")
    assert result[0]["field"] == "config.method.backend.hub_token"
    assert "old-secret" not in json.dumps(result) and "new-secret" not in json.dumps(result)


def test_length_increase_allowed_but_unrelated_template_change_rejected(config):
    current = {"config": config.model_dump(mode="json"), "targets": {}, "sources": {}, "versions": {}, "code": {}}
    prior = deepcopy(current)
    current["config"]["model"]["max_length"] *= 2
    current["config"]["template"]["max_length"] *= 2
    validate_manifest(prior, current)
    current["config"]["template"]["enable_thinking"] = not prior["config"]["template"]["enable_thinking"]
    with pytest.raises(ResumeMismatch, match="config.template.enable_thinking"):
        validate_manifest(prior, current)


def test_real_resume_after_length_increase_does_not_add_new_targets_or_images(config, elastic_qwen_path, tmp_path):
    from PIL import Image
    import torch

    config.data.accepted_statuses.append("UNKNOWN")

    source = next(config.data.runs_root.rglob("trajectory.jsonl"))
    row = json.loads(source.read_text())
    row["messages"][4]["content"] = "\n".join(["print('long')"] * 400)
    row["messages"][3]["content"][1]["image_url"]["url"] = "images/second.png"
    Image.new("RGB", (32, 32), "blue").save(source.parent / "images/second.png")
    source.write_text(json.dumps(row) + "\n")
    config.model.path = elastic_qwen_path
    config.method.use_cpu = True
    config.method.max_steps = 1
    config.method.gradient_accumulation_steps = 1
    config.method.gradient_checkpointing = False
    config.method.save_steps = 1
    config.method.backend.lr_scheduler_type = "constant"
    config.method.backend.eval_strategy = "no"
    config.method.backend.do_eval = False
    config.method.backend.disable_tqdm = True
    config.method.warmup_ratio = 0
    trainer = Trainer(config)
    data = trainer.prepare()
    short_length = trainer.template.encode(data.train[0])["input_ids"].shape[1]
    config.model.max_length = config.template.max_length = short_length + 8
    run_training(config, tmp_path / "first.yaml", 1)
    output = config.experiment.output_dir
    original = json.loads((output / "manifest.json").read_text())
    assert len(original["targets"]["train"]) == 1
    assert str(source.parent / "images/second.png") not in original["sources"]
    checkpoint = output / "checkpoints/checkpoint-1"
    config.method.resume_from = checkpoint
    config.method.resume_policy = "elastic"
    config.model.max_length = config.template.max_length = 4096
    config.method.max_steps = 2
    # This reproduces the previous bug: fresh filtering admits a second target and image.
    assert len(Trainer(config).prepare(encode=True).train) == 2
    assert Trainer(config).check_resume()["status"] == "passed"
    run_training(config, tmp_path / "resume.yaml", 1)
    final = output / "checkpoints/checkpoint-2"
    saved = json.loads((final / "resume_manifest.json").read_text())
    assert saved["targets"] == original["targets"]
    assert saved["sources"] == original["sources"]
    progress = json.loads((final / "data_progress.json").read_text())
    assert progress["sample_count"] == 1 and progress["global_step"] == 2
    state = torch.load(final / "portable_state.pt", map_location="cpu", weights_only=True)
    assert all(int(item["step"]) == 2 for item in state["optimizer"]["state"].values())
