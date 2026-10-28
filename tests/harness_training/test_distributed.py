import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from harness_training.config import load_config, save_config
from conftest import write_trajectory


@pytest.mark.parametrize("strategy, quantization, loss_type", [
    ("single", "none", "nll"), ("single", "4bit", "nll"),
    ("ddp", "none", "nll"), ("ddp", "4bit", "nll"),
    ("fsdp", "none", "nll"), ("fsdp", "4bit", "nll"),
    ("fsdp", "none", "chunked_nll"),
])
def test_gpu_training_save_resume(tiny_qwen_path, tmp_path, strategy, quantization, loss_type):
    """Run real BF16/4-bit training and collective checkpoint recovery on bounded inputs."""
    import torch
    from safetensors.torch import load_file

    devices = 1 if strategy == "single" else 2
    if torch.cuda.device_count() < devices:
        pytest.skip(f"Requires {devices} CUDA GPUs")
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/sft_single_view.yaml", [
        f"distributed={strategy}", f"quantization={quantization}",
        "method.gradient_checkpointing=true", "method.backend.bf16=true",
        "method.backend.fp16=false", "method.backend.eval_strategy=no", "method.backend.do_eval=false",
        f"method.backend.loss_type={loss_type}",
        f"method.backend.prediction_loss_only={'true' if loss_type == 'chunked_nll' else 'false'}",
        f"tracking={'disabled' if strategy == 'single' else 'wandb'}",
    ])
    if strategy != "single":
        config.tracking.mode = "offline"
    config.tracking.settings.x_disable_stats = True
    config.model.path = tiny_qwen_path
    config.data.runs_root = tmp_path / "runs"
    config.data.validation_fraction = 0
    config.data.test_fraction = 0
    for index in range(2):
        write_trajectory(config.data.runs_root, instance=f"Object{index}Factory")
    config.experiment.output_dir = tmp_path / "output"
    config.template.min_image_pixels = 1024
    config.template.max_image_pixels = 1024
    config.template.max_length = 4096
    config.model.lora.rank = 2
    config.model.lora.alpha = 4
    config.model.lora.dropout = 0
    config.model.lora.target_modules = ["q_proj", "v_proj"]
    config.method.max_steps = 1
    config.method.gradient_accumulation_steps = 1
    config.method.learning_rate = .01
    config.method.save_steps = 1
    config.method.warmup_ratio = 0
    config.method.backend.disable_tqdm = True
    path = tmp_path / "config.yaml"
    command = [sys.executable]
    if devices > 1:
        command += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2"]
    command += ["-m", "harness_training.cli", "train", "--config", str(path)]
    env = {**os.environ, "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}
    if devices == 1:
        env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    wandb_id = None
    for step in [1, 2]:
        config.method.max_steps = step
        if step == 2:
            config.method.resume_from = config.experiment.output_dir / "checkpoints/checkpoint-1"
        save_config(config, path)
        result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180)
        (tmp_path / f"step_{step}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, (result.stdout + result.stderr)[-18000:]
        state = json.loads((config.experiment.output_dir / f"checkpoints/checkpoint-{step}/trainer_state.json").read_text())
        assert state["global_step"] == step
        assert json.loads((config.experiment.output_dir / "metrics.json").read_text())["train_loss"] > 0
        weights = load_file(config.experiment.output_dir / "final/adapter_model.safetensors")
        assert any(torch.count_nonzero(value).item() > 0 for key, value in weights.items() if "lora_B" in key)
        assert (config.experiment.output_dir / "final/preprocessor_config.json").is_file() or (
            config.experiment.output_dir / "final/processor_config.json").is_file()
        if strategy != "single":
            state = json.loads((config.experiment.output_dir / "wandb_run.json").read_text())
            if wandb_id is None:
                wandb_id = state["id"]
            assert state["id"] == wandb_id
            # Exactly one offline segment per launch, despite using two GPUs.
            assert len(list((config.experiment.output_dir / "wandb").glob("offline-run-*/run-*.wandb"))) == step
