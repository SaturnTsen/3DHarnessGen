import json

import pytest

from harness_training.training.trainer import Trainer
from harness_training.training.methods.sft import SFTMethod
from conftest import write_trajectory


def test_backend_arguments_use_explicit_config(config):
    from dataclasses import fields
    from inspect import signature
    from trl import SFTConfig, SFTTrainer

    config.method.use_cpu = True
    backend = config.method.backend
    backend.adam_beta1 = .8
    backend.max_grad_norm = .7
    backend.lr_scheduler_type = "cosine"
    backend.per_device_eval_batch_size = 3
    backend.gradient_checkpointing_kwargs.use_reentrant = True
    backend.report_to = []
    backend.data_seed = 17
    backend.eval_strategy = "no"
    backend.do_eval = False
    backend.loss_type = "dft"
    method = SFTMethod(config.method, config.experiment.seed)
    args = method.build_arguments(config.experiment.output_dir, config.method.resume_from)
    assert args.adam_beta1 == .8
    assert args.max_grad_norm == .7
    assert args.lr_scheduler_type == "cosine"
    assert args.per_device_eval_batch_size == 3
    assert args.gradient_checkpointing_kwargs == {"use_reentrant": True}
    assert args.report_to == []
    assert args.data_seed == 17
    assert args.eval_strategy == "no"
    assert args.loss_type == "dft"
    configured = set(type(backend).model_fields) | {
        "output_dir", "learning_rate", "num_train_epochs", "max_steps", "per_device_train_batch_size",
        "gradient_accumulation_steps", "gradient_checkpointing", "warmup_steps", "weight_decay",
        "logging_steps", "save_steps", "save_total_limit", "eval_steps", "use_cpu", "dataloader_num_workers",
        "seed", "resume_from_checkpoint",
    }
    assert configured == {field.name for field in fields(SFTConfig) if field.init}
    trainer_fields = set(type(config.method.trainer).model_fields) | {
        "model", "args", "data_collator", "train_dataset", "eval_dataset", "processing_class",
    }
    assert trainer_fields == set(signature(SFTTrainer).parameters)


def test_small_qwen_sft_save_and_resume(config, processor, tmp_path):
    """Exercise real model loading, LoRA gradients, backend training and checkpoint restore."""
    import torch
    from transformers import AutoConfig, AutoProcessor, Qwen3_5Config, Qwen3_5ForConditionalGeneration

    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        cfg = AutoConfig.from_pretrained(str(config.model.path), local_files_only=True).to_dict()
        cfg["dtype"] = "float32"
        cfg["text_config"].update(
            dtype="float32", hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            layer_types=["full_attention"], partial_rotary_factor=1.0,
            rope_parameters={"rope_type": "default", "rope_theta": 10000, "mrope_section": [2, 3, 3]},
        )
        cfg["vision_config"].update(depth=1, hidden_size=32, intermediate_size=64,
                                    num_heads=2, out_hidden_size=32, deepstack_visual_indexes=[])
        model_path = tmp_path / "tiny_qwen"
        tiny = Qwen3_5ForConditionalGeneration(Qwen3_5Config.from_dict(cfg))
        tiny.save_pretrained(model_path)
        processor.save_pretrained(model_path)
        del tiny
        config.model.path = model_path
        # Ten objects give one held-out object at a 10% validation fraction.
        config.data.validation_fraction = .1
        for index in range(9):
            write_trajectory(config.data.runs_root, instance=f"Object{index}Factory")
        config.model.lora = config.model.lora.model_copy(update={
            "rank": 2, "alpha": 4, "dropout": 0, "target_modules": ["q_proj", "v_proj"],
        })
        config.method = config.method.model_copy(update={
            "max_steps": 1, "batch_size": 1, "gradient_accumulation_steps": 1,
            "learning_rate": .01, "gradient_checkpointing": False, "logging_steps": 1,
            "save_steps": 1, "warmup_ratio": 0, "use_cpu": True,
        })
        trainer = Trainer(config)
        model = trainer.model
        metrics = trainer.run()
        assert metrics["train_loss"] > 0
        assert metrics["eval_loss"] > 0
        assert any(torch.count_nonzero(p).item() > 0 for n, p in model.model.named_parameters() if "lora_B" in n)
        assert (config.experiment.output_dir / "final/adapter_config.json").is_file()
        restored_processor = AutoProcessor.from_pretrained(config.experiment.output_dir / "final", local_files_only=True)
        assert restored_processor.tokenizer.eos_token_id == processor.tokenizer.eos_token_id
        assert restored_processor.chat_template == model.processor.chat_template
        checkpoint = config.experiment.output_dir / "checkpoints/checkpoint-1"
        assert (checkpoint / "trainer_state.json").is_file()
        config.method.max_steps = 2
        config.method.resume_from = checkpoint
        Trainer(config).run()
        state = json.loads((config.experiment.output_dir / "checkpoints/checkpoint-2/trainer_state.json").read_text())
        assert state["global_step"] == 2
        # Same checkpoint must not silently resume on changed training data.
        source = next(config.data.runs_root.glob("*/gpt-6-astra/*/post_train_trajectory/trajectory.jsonl"))
        row = json.loads(source.read_text())
        row["conversations"][1]["value"] += "\n# changed"
        source.write_text(json.dumps(row))
        with pytest.raises(ValueError, match="changed"):
            Trainer(config).run()
    finally:
        torch.set_num_threads(threads)
