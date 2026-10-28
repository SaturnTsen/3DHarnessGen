"""Exercise the two 27B task entrypoints with a bounded hybrid Qwen model.

The four-GPU variants run on machines with four CUDA devices; the same suite
uses two GPUs locally. Production model files and trajectories remain read-only.
"""

import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from conftest import MODEL_PATH, write_trajectory
from test_templates import interactive_record


@pytest.fixture(scope="module")
def hybrid_model_path(processor, tmp_path_factory):
    import torch
    from transformers import AutoConfig, Qwen3_5Config, Qwen3_5ForConditionalGeneration

    config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True).to_dict()
    config["dtype"] = "bfloat16"
    config["text_config"].update(
        dtype="bfloat16", hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        layer_types=["linear_attention"] * 3 + ["full_attention"], partial_rotary_factor=1.0,
        linear_num_key_heads=1, linear_num_value_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16,
        rope_parameters={"rope_type": "default", "rope_theta": 10000, "mrope_section": [2, 3, 3]},
    )
    config["vision_config"].update(
        depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
        out_hidden_size=32, deepstack_visual_indexes=[],
    )
    destination = tmp_path_factory.mktemp("hybrid_mix_model")
    torch.manual_seed(42)
    model = Qwen3_5ForConditionalGeneration(Qwen3_5Config.from_dict(config))
    model.bfloat16()
    model.save_pretrained(destination)
    processor.save_pretrained(destination)
    return destination


def write_mix_fixture(root: Path, task: str) -> None:
    for index in range(10):
        instance = f"Object{index}Factory"
        if task == "view":
            for scene in ("Single-view", "Multi-view"):
                write_trajectory(root, scene, instance)
        else:
            active = interactive_record(root, instance=instance)
            destination = root / "Full3DInteraction/gpt-6-astra" / instance / "post_train_trajectory"
            shutil.copytree(active.parent, destination)
            copied = destination / "trajectory.jsonl"
            row = json.loads(copied.read_text())
            row["metadata"]["scene"] = "Full3DInteraction"
            copied.write_text(json.dumps(row) + "\n")


@pytest.mark.parametrize("task", ["view", "interaction"])
@pytest.mark.parametrize("devices", [2, 4])
def test_mix_task_torchrun_training_eval_save_resume(task, devices, hybrid_model_path, tmp_path):
    import torch
    from safetensors.torch import load_file

    if torch.cuda.device_count() < devices:
        pytest.skip(f"Requires {devices} CUDA GPUs; this machine has {torch.cuda.device_count()}")
    root = Path(__file__).resolve().parents[2]
    runs = tmp_path / "runs"
    output = tmp_path / "output"
    write_mix_fixture(runs, task)
    config = f"configs/sft_{task}_mix_27b_fsdp.yaml"
    command = [
        "pixi", "run", "python", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        f"--nproc-per-node={devices}", "-m", "harness_training.cli", "train", "--config", config,
        f"model.path={hybrid_model_path}", f"data.runs_root={runs}", f"experiment.output_dir={output}",
        "tracking=disabled", "method.save_steps=1", "method.eval_steps=1",
        "method.warmup_ratio=0",
        "method.backend.disable_tqdm=true",
    ]
    environment = {
        **os.environ, "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    }
    # Keep the task's image budgets, NF4, SDPA fused-only policy, adapter sizes,
    # gradient accumulation, and all chunk sizes. Only model/data, step count,
    # output, save/eval frequency, warmup, and online tracking are changed.
    previous_weights = None
    for step in (2, 3):
        args = command + [f"method.max_steps={step}"]
        if step == 3:
            args += [f"method.resume_from={output / 'checkpoints/checkpoint-1'}"]
        (tmp_path / f"command_{step}.json").write_text(json.dumps(args, indent=2))
        with (tmp_path / f"step_{step}.log").open("w") as stream:
            result = subprocess.run(args, cwd=root, env=environment, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=300)
        assert result.returncode == 0, (tmp_path / f"step_{step}.log").read_text()[-24000:]
        metrics = json.loads((output / "metrics.json").read_text())
        assert math.isfinite(metrics["train_loss"]) and metrics["train_loss"] > 0
        assert math.isfinite(metrics["eval_loss"]) and metrics["eval_loss"] > 0
        state = json.loads((output / f"checkpoints/checkpoint-{step}/trainer_state.json").read_text())
        assert state["global_step"] == step
        weights = load_file(output / "final/adapter_model.safetensors")
        assert weights and all(torch.isfinite(value).all() for value in weights.values())
        assert any(value.count_nonzero() for name, value in weights.items() if "lora_B" in name)
        if previous_weights is not None:
            assert any(not torch.equal(weights[name], value) for name, value in previous_weights.items())
        previous_weights = weights
        assert (output / "final/processor_config.json").is_file() or (
            output / "final/preprocessor_config.json").is_file()
    report = json.loads((output / "data_report.json").read_text())
    assert report["objects"] == 10 and report["rejected"] == 0
    assert all(report["samples"][split] > 0 for split in ("train", "validation", "test"))
