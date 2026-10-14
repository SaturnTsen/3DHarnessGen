import json
import hashlib

import pytest

from harness_training.config import LoRAConfig
from harness_training.training.trainer import Trainer
from conftest import write_trajectory


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
        # Include a distinct held-out object to exercise backend evaluation too.
        config.data.validation_fraction = .1
        validation_name = next(f"Holdout{i}Factory" for i in range(1000)
                               if int(hashlib.sha256(f"42:Holdout{i}Factory".encode()).hexdigest()[:16], 16) / 2**64 < .1)
        write_trajectory(config.data.runs_root, instance=validation_name)
        training_name = next(f"Train{i}Factory" for i in range(1000)
                             if int(hashlib.sha256(f"42:Train{i}Factory".encode()).hexdigest()[:16], 16) / 2**64 >= .1)
        write_trajectory(config.data.runs_root, instance=training_name)
        config.model.lora = LoRAConfig(rank=2, alpha=4, dropout=0, target_modules=["q_proj", "v_proj"])
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
