from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from accelerate.utils import FullyShardedDataParallelPlugin
    from trl import SFTConfig

from harness_training.config import MethodConfig
from harness_training.data.base import DatasetBundle
from harness_training.models.base import BaseModel
from harness_training.templates.base import BaseTemplate
from harness_training.training.collator import SFTCollator
from harness_training.types import JsonObject, Metrics
from .base import BaseTrainingMethod


class SFTMethod(BaseTrainingMethod):
    def __init__(self, config: MethodConfig, seed: int):
        self.config, self.seed = config, seed

    def build_arguments(self, output_dir: Path, resume_from: Path | None) -> SFTConfig:
        from trl import SFTConfig

        cfg = self.config
        options: JsonObject = cfg.backend.model_dump(mode="json")
        options.update(
            output_dir=str(output_dir / "checkpoints"),
            learning_rate=cfg.learning_rate, num_train_epochs=cfg.epochs, max_steps=cfg.max_steps,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            gradient_checkpointing=cfg.gradient_checkpointing,
            warmup_steps=cfg.warmup_ratio, weight_decay=cfg.weight_decay,
            logging_steps=cfg.logging_steps, save_steps=cfg.save_steps, save_total_limit=cfg.save_total_limit,
            eval_steps=cfg.eval_steps, use_cpu=cfg.use_cpu, dataloader_num_workers=cfg.dataloader_workers,
            seed=self.seed, resume_from_checkpoint=str(resume_from) if resume_from else None,
        )
        required = {field.name for field in fields(SFTConfig) if field.init}
        if options.keys() != required:
            raise ValueError(f"SFTConfig fields changed: missing={sorted(required - options.keys())}, "
                             f"unknown={sorted(options.keys() - required)}")
        return SFTConfig(**options)

    def run(self, model: BaseModel, template: BaseTemplate, data: DatasetBundle,
            output_dir: Path, resume_from: Path | None) -> Metrics:
        from datasets import Dataset
        from trl import SFTTrainer
        from transformers import set_seed

        if not data.train:
            raise ValueError("No usable training samples")

        cfg = self.config
        set_seed(self.seed)
        if (cfg.backend.eval_strategy != "no" or cfg.backend.do_eval) and not data.validation:
            raise ValueError("Configured evaluation requires a nonempty validation split")
        args = self.build_arguments(output_dir, resume_from)
        # The backend's Dataset carries only integer references to in-memory views.
        # Original conversations, images and provenance remain in the existing records.
        examples = data.train + data.validation
        backend = SFTTrainer(
            model=model.load(), args=args, processing_class=model.processor,
            train_dataset=Dataset.from_dict({"example_index": list(range(len(data.train)))}),
            eval_dataset=Dataset.from_dict({"example_index": list(range(len(data.train), len(examples)))}) if data.validation else None,
            data_collator=SFTCollator(template, model.processor.tokenizer.pad_token_id, examples,
                                      label_ignore_index=template.config.label_ignore_index,
                                      attention_padding_value=template.config.attention_padding_value,
                                      token_type_padding_value=template.config.token_type_padding_value),
            **cfg.trainer.model_dump(),
        )
        result = backend.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
        metrics: Metrics = dict(result.metrics)
        if cfg.backend.do_eval:
            metrics.update(backend.evaluate())
        backend.save_state()
        # FSDP gathers final weights collectively; every rank must participate.
        if backend.is_fsdp_enabled:
            fsdp_plugin = cast("FullyShardedDataParallelPlugin", backend.accelerator.state.fsdp_plugin)
            fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        backend.save_model(str(output_dir / "final"))
        if backend.is_world_process_zero():
            backend.save_metrics("train", metrics)
        return metrics
