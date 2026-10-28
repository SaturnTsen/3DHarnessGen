"""Carry trained adapters into another dataset with a fresh training state."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from harness_training.config import load_config, save_config
from harness_training.models.qwen import QwenModel
from harness_training.training.trainer import Trainer
from test_mix_distributed import hybrid_model_path, write_mix_fixture
from test_templates import interactive_record


@pytest.mark.parametrize("source_kind", ["checkpoint", "final"])
def test_adapter_transfer_and_new_stage_resume(config, tiny_qwen_path, tmp_path, source_kind):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file

    torch.set_num_threads(2)
    config.model.path = tiny_qwen_path
    config.model.lora.rank = 2
    config.model.lora.alpha = 4
    config.model.lora.dropout = 0
    config.method.max_steps = 2
    config.method.gradient_accumulation_steps = 1
    config.method.gradient_checkpointing = False
    config.method.use_cpu = True
    config.method.learning_rate = .01
    config.method.warmup_ratio = 0
    config.method.save_steps = 1
    config.method.backend.eval_strategy = "no"
    config.method.backend.do_eval = False
    config.method.backend.disable_tqdm = True
    Trainer(config).run()
    original_output = config.experiment.output_dir
    source = original_output / ("checkpoints/checkpoint-2" if source_kind == "checkpoint" else "final")
    saved = load_file(source / "adapter_model.safetensors")
    original_bytes = (source / "adapter_model.safetensors").read_bytes()

    config.model.init_from = source
    incompatible = config.model.model_copy(update={"path": tmp_path / "another_base"})
    with pytest.raises(ValueError, match="base model"):
        QwenModel(incompatible, config.template).validate_runtime()
    incompatible = config.model.model_copy(deep=True)
    incompatible.lora.alpha += 1
    with pytest.raises(ValueError, match="LoRA settings"):
        QwenModel(incompatible, config.template).validate_runtime()

    initialized = QwenModel(config.model, config.template).load()
    actual = get_peft_model_state_dict(initialized)
    assert actual.keys() == saved.keys()
    assert all(torch.equal(actual[name], value) for name, value in saved.items())
    assert all(parameter.requires_grad for name, parameter in initialized.named_parameters() if "lora_" in name)
    del initialized

    # A different config selects interactive data and a new W&B run/output.
    config.experiment.scene = "ActiveVisual"
    config.data.runs_root = tmp_path / "interaction_runs"
    interactive_record(config.data.runs_root)
    config.experiment.output_dir = tmp_path / "stage_two"
    config.method.max_steps = 1
    config.method.learning_rate = .003
    config.tracking.mode = "offline"
    config.method.backend.report_to = ["wandb"]
    config.tracking.settings.x_disable_stats = True
    data = Trainer(config).prepare(encode=True)
    assert data.train and all(example.record.row.metadata.scene == "ActiveVisual" for example in data.train)
    trainer = Trainer(config)
    encoded = trainer.template.encode(data.train[1])
    assert encoded["pixel_values"].numel() > 0
    assert (encoded["labels"] == -100).any() and (encoded["labels"] != -100).any()
    metrics = trainer.run()
    assert metrics["train_loss"] > 0
    output = config.experiment.output_dir
    checkpoint = output / "checkpoints/checkpoint-1"
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    assert state["global_step"] == 1
    optimizer = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    assert all(int(value["step"]) == 1 for value in optimizer["state"].values())
    updated = load_file(output / "final/adapter_model.safetensors")
    assert any(not torch.equal(updated[name], value) for name, value in saved.items())
    assert (source / "adapter_model.safetensors").read_bytes() == original_bytes
    run_id = json.loads((output / "wandb_run.json").read_text())["id"]

    # Resuming the new stage retains init_from as provenance and restores step 1.
    config.method.resume_from = checkpoint
    config.method.max_steps = 2
    Trainer(config).run()
    assert json.loads((output / "checkpoints/checkpoint-2/trainer_state.json").read_text())["global_step"] == 2
    assert json.loads((output / "wandb_run.json").read_text())["id"] == run_id


def test_fsdp_qlora_transfer_between_mix_configs(hybrid_model_path, tmp_path):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file

    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA GPUs")
    root = Path(__file__).resolve().parents[2]
    runs = tmp_path / "runs"
    for task in ("view", "interaction"):
        write_mix_fixture(runs, task)
    source = None
    source_config = None
    for task in ("view", "interaction"):
        config = load_config(root / f"configs/sft_{task}_mix_27b_fsdp.yaml", ["tracking=disabled"])
        config.model.path = hybrid_model_path
        config.model.init_from = source
        config.data.runs_root = runs
        config.experiment.output_dir = tmp_path / task
        config.method.max_steps = 1
        config.method.save_steps = 1
        config.method.warmup_ratio = 0
        config.method.backend.disable_tqdm = True
        path = tmp_path / f"{task}.yaml"
        save_config(config, path)
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                   "--nproc-per-node=2", "-m", "harness_training.cli", "train", "--config", str(path)]
        with (tmp_path / f"{task}.log").open("w") as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "OMP_NUM_THREADS": "2",
                                         "TOKENIZERS_PARALLELISM": "false"}, timeout=300)
        assert result.returncode == 0, (tmp_path / f"{task}.log").read_text()[-16000:]
        source = config.experiment.output_dir / "checkpoints/checkpoint-1"
        assert (source / "pytorch_model_fsdp_0/.metadata").is_file()
        assert json.loads((source / "trainer_state.json").read_text())["global_step"] == 1
        if source_config is None:
            source_config = config
            # Compare selectively loaded sharded tensors with the collective final export.
            config.model.init_from = source
            config.model.quantization = None
            model = QwenModel(config.model, config.template, config.method.long_context)
            actual = get_peft_model_state_dict(model.load())
            saved = load_file(config.experiment.output_dir / "final/adapter_model.safetensors")
            assert actual.keys() == saved.keys()
            assert all(torch.equal(actual[name].cpu(), value) for name, value in saved.items())
            del model
            config.model.lora.alpha += 1
            with pytest.raises(ValueError, match="must match the source run"):
                QwenModel(config.model, config.template).validate_runtime()

    first = load_file(tmp_path / "view/final/adapter_model.safetensors")
    second = load_file(tmp_path / "interaction/final/adapter_model.safetensors")
    assert any(not torch.equal(first[name], second[name]) for name in first)
