from __future__ import annotations

from pathlib import Path
from typing import Any

from harness_training.config import MethodConfig
from harness_training.data.base import DatasetBundle
from harness_training.models.base import BaseModel
from harness_training.templates.base import BaseTemplate
from harness_training.training.collator import SFTCollator
from .base import BaseTrainingMethod


class SFTMethod(BaseTrainingMethod):
    def __init__(self, config: MethodConfig, seed: int = 42):
        self.config, self.seed = config, seed

    def run(self, model: BaseModel, template: BaseTemplate, data: DatasetBundle,
            output_dir: Path, resume_from: Path | None = None) -> dict[str, Any]:
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
        from transformers import set_seed

        if not data.train:
            raise ValueError("No usable training samples")

        cfg = self.config
        set_seed(self.seed)
        args = SFTConfig(
            output_dir=str(output_dir / "checkpoints"),
            learning_rate=cfg.learning_rate, num_train_epochs=cfg.epochs, max_steps=cfg.max_steps,
            per_device_train_batch_size=cfg.batch_size, per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            gradient_checkpointing=cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            warmup_steps=cfg.warmup_ratio, weight_decay=cfg.weight_decay,
            logging_steps=cfg.logging_steps, save_steps=cfg.save_steps, save_total_limit=cfg.save_total_limit,
            eval_strategy="steps" if data.validation else "no", eval_steps=cfg.eval_steps,
            bf16=model.config.dtype == "bfloat16" and not cfg.use_cpu,
            fp16=model.config.dtype == "float16" and not cfg.use_cpu,
            use_cpu=cfg.use_cpu, dataloader_num_workers=cfg.dataloader_workers,
            seed=self.seed, data_seed=self.seed, report_to="none",
            remove_unused_columns=False, max_length=None, packing=False,
            assistant_only_loss=False, completion_only_loss=False,
            dataset_kwargs={"skip_prepare_dataset": True}, loss_type="nll",
        )
        # The backend's Dataset carries only integer references to in-memory views.
        # Original conversations, images and provenance remain in the existing records.
        examples = data.train + data.validation
        backend = SFTTrainer(
            model=model.load(), args=args, processing_class=model.processor,
            train_dataset=Dataset.from_dict({"example_index": list(range(len(data.train)))}),
            eval_dataset=Dataset.from_dict({"example_index": list(range(len(data.train), len(examples)))}) if data.validation else None,
            data_collator=SFTCollator(template, model.processor.tokenizer.pad_token_id, examples),
        )
        result = backend.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
        metrics = dict(result.metrics)
        if data.validation:
            metrics.update(backend.evaluate())
        backend.save_state()
        if backend.is_world_process_zero():
            model.save(output_dir / "final")
            backend.save_metrics("train", metrics)
        return metrics
