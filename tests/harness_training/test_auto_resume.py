"""Automatic selection and real checkpoint restoration through the Pixi launcher."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from harness_training.config import save_config
from scripts.post_train import checkpoint_problem, resolve_arguments
from test_tracking import enable_tracking


ROOT = Path(__file__).resolve().parents[2]


def checkpoint_files(path):
    path.mkdir(parents=True)
    step = int(path.name.removeprefix("checkpoint-"))
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (path / name).write_bytes(b"state")


def test_auto_selects_numeric_latest_and_skips_incomplete_checkpoint(config, tmp_path):
    output = config.experiment.output_dir
    checkpoint_files(output / "checkpoints/checkpoint-9")
    checkpoint_files(output / "checkpoints/checkpoint-20")
    partial = output / "checkpoints/checkpoint-30"
    partial.mkdir()
    (partial / "trainer_state.json").write_text('{"global_step":')
    for name in ("manifest.json", "resolved_config.yaml"):
        (output / name).write_text("{}")
    path = tmp_path / "config.yaml"
    save_config(config, path)
    arguments = ["train", "--config", str(path), "method.resume_from=auto"]
    resolved = resolve_arguments(arguments)
    assert resolved[-1] == f"method.resume_from={output / 'checkpoints/checkpoint-20'}"
    (output / "checkpoints/checkpoint-20/optimizer.pt").unlink()
    assert resolve_arguments(arguments)[-1].endswith("checkpoint-9")
    (output / "checkpoints/checkpoint-9/scheduler.pt").unlink()
    with pytest.raises(ValueError, match="No resumable checkpoint"):
        resolve_arguments(arguments)


def test_rejects_process_count_change(config, tmp_path):
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint_files(checkpoint)
    assert "process count" in checkpoint_problem(checkpoint, config, 2)


def test_checks_fsdp_optimizer_and_all_rank_states(config, tmp_path):
    config.distributed.strategy = "fsdp"
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint_files(checkpoint)
    (checkpoint / "rng_state.pth").unlink()
    for name in ("pytorch_model_fsdp_0/.metadata", "optimizer_0/.metadata",
                 "rng_state_0.pth", "rng_state_1.pth"):
        path = checkpoint / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"state")
    assert checkpoint_problem(checkpoint, config, 2) is None
    (checkpoint / "optimizer_0/.metadata").unlink()
    assert "missing or empty" in checkpoint_problem(checkpoint, config, 2)
    (checkpoint / "rng_state_1.pth").unlink()
    assert "process count" in checkpoint_problem(checkpoint, config, 2)


def test_passthrough_preserves_existing_commands():
    arguments = ["train", "--config", "config.yaml", "method.resume_from=/saved/checkpoint-10"]
    assert resolve_arguments(arguments) == arguments


def test_elastic_auto_checks_completion_but_accepts_old_process_count(config, tmp_path):
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint_files(checkpoint)
    config.method.resume_policy = "elastic"
    config.distributed.strategy = "fsdp"
    assert checkpoint_problem(checkpoint, config, 2) is None
    (checkpoint / "resume_manifest.json").write_text('{"config":{"method":{"resume_policy":"elastic"}}}')
    problem = checkpoint_problem(checkpoint, config, 2)
    assert problem is not None and "missing or empty" in problem
    (checkpoint / "data_progress.json").write_text('{"global_step":')
    problem = checkpoint_problem(checkpoint, config, 2)
    assert problem is not None and "incomplete data progress" in problem


def test_auto_resume_small_model_preserves_training_and_wandb_state(config, tiny_qwen_path, tmp_path):
    import torch
    from safetensors.torch import load_file

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
    config.method.backend.lr_scheduler_type = "constant"
    config.model.lora.rank = 2
    config.model.lora.alpha = 4
    config.model.lora.target_modules = ["q_proj", "v_proj"]
    path = tmp_path / "config.yaml"
    save_config(config, path)
    command = [sys.executable, "-m", "scripts.post_train", "train", "--config", str(path)]
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": ""}

    def run(arguments):
        result = subprocess.run(command + arguments, cwd=ROOT, env=environment,
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    run([])
    output = config.experiment.output_dir
    identity = json.loads((output / "wandb_run.json").read_text())["id"]
    fingerprint = json.loads((output / "manifest.json").read_text())["fingerprint"]
    before = load_file(output / "final/adapter_model.safetensors")
    (output / "checkpoints/checkpoint-999").mkdir()
    log = run(["method.resume_from=auto", "method.max_steps=2"])
    assert "Skip checkpoint-999" in log and "Restore " in log
    state = json.loads((output / "checkpoints/checkpoint-2/trainer_state.json").read_text())
    assert state["global_step"] == 2
    optimizer = torch.load(output / "checkpoints/checkpoint-2/optimizer.pt", weights_only=True)
    assert all(int(value["step"]) == 2 for value in optimizer["state"].values())
    after = load_file(output / "final/adapter_model.safetensors")
    assert any(not torch.equal(after[name], value) for name, value in before.items())
    assert json.loads((output / "wandb_run.json").read_text())["id"] == identity
    assert json.loads((output / "manifest.json").read_text())["fingerprint"] == fingerprint
    incompatible = subprocess.run(command + ["method.resume_from=auto", "method.batch_size=2"],
                                  cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120)
    assert incompatible.returncode != 0
    assert "changed since this checkpoint" in incompatible.stderr
