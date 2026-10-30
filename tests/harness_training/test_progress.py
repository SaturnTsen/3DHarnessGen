"""Restored steps contribute to progress, while speed uses only new updates."""

import json

import pytest
from transformers import TrainerControl, TrainerState

from harness_training.training.progress import ResumeProgressCallback
from test_elastic_resume import elastic_qwen_path, run_training


@pytest.mark.parametrize("start", [0, 300])
def test_resume_rate_and_eta_exclude_checkpoint_steps(start):
    callback = ResumeProgressCallback()
    state = TrainerState(global_step=start, max_steps=start + 36, is_world_process_zero=True)
    control = TrainerControl()
    callback.on_train_begin(None, state, control)
    bar = callback.training_bar
    assert bar.n == bar.initial == callback.current_step == start
    for step in range(start + 1, start + 7):
        state.global_step = step
        callback.on_step_end(None, state, control)
    assert bar.n == start + 6 and callback.current_step == start + 6
    # Reproduce the actual report: six new updates in 14:51, thirty left.
    text = bar.format_meter(n=bar.n, total=bar.total, initial=bar.initial, elapsed=891, rate=None, ncols=100)
    assert "148.50s/it" in text
    assert "14:51<1:14:15" in text
    callback.on_log(None, state, control, logs={"loss": .3})
    callback.on_train_end(None, state, control)
    assert callback.training_bar is None


def test_nonzero_rank_does_not_create_a_progress_bar():
    callback = ResumeProgressCallback()
    state = TrainerState(global_step=300, max_steps=336, is_world_process_zero=False)
    control = TrainerControl()
    callback.on_train_begin(None, state, control)
    state.global_step = 301
    callback.on_step_end(None, state, control)
    callback.on_train_end(None, state, control)
    assert callback.training_bar is None


def test_real_resumed_training_starts_at_restored_step(config, elastic_qwen_path, tmp_path):
    import torch

    config.model.path = elastic_qwen_path
    config.method.use_cpu = True
    config.method.max_steps = 1
    config.method.save_steps = 1
    config.method.gradient_accumulation_steps = 1
    config.method.gradient_checkpointing = False
    config.method.backend.lr_scheduler_type = "constant"
    config.method.backend.eval_strategy = "no"
    config.method.backend.do_eval = False
    config.method.backend.disable_tqdm = False
    config.method.warmup_ratio = 0
    run_training(config, tmp_path / "initial.yaml", 1)
    checkpoint = config.experiment.output_dir / "checkpoints/checkpoint-1"
    config.method.resume_from = checkpoint
    config.method.resume_policy = "elastic"
    config.method.max_steps = 3
    run_training(config, tmp_path / "resumed.yaml", 1)
    log = (tmp_path / "resumed.log").read_text()
    assert "1/3 [00:00" in log
    assert "0/3 [" not in log
    final = config.experiment.output_dir / "checkpoints/checkpoint-3"
    state = torch.load(final / "portable_state.pt", map_location="cpu", weights_only=True)
    assert all(int(value["step"]) == 3 for value in state["optimizer"]["state"].values())
    progress = json.loads((final / "data_progress.json").read_text())
    assert progress["global_step"] == 3 and progress["next_sample"] == 1
