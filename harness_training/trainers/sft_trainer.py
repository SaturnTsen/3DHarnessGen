from __future__ import annotations

import os
from dataclasses import fields
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

from transformers.trainer_pt_utils import BatchRebalanceSampler
from trl.trainer.sft_trainer import SFTTrainer as TRLSFTTrainer

from harness_training.config import TrainingConfig
from harness_training.data.base import DatasetBundle
from harness_training.dataloader import SFTDataLoader
from harness_training.chat_templates.base import BaseTemplate
from harness_training.tracking import tracking_session
from harness_training.types import Metrics
from .base import BaseTrainer

if TYPE_CHECKING:
    from accelerate.utils import FullyShardedDataParallelPlugin
    from datasets import Dataset
    from transformers.processing_utils import ProcessorMixin
    from trl.trainer.sft_config import SFTConfig


class SFTTrainer(BaseTrainer):
    def __init__(self, config: TrainingConfig):
        if config.method.name != "sft":
            raise ValueError(f"Unsupported training method: {config.method.name}")
        super().__init__(config)
        self.backend: DataLoaderSFTTrainer | None = None

    def build_arguments(self, output_dir: Path) -> SFTConfig:
        from trl.trainer.sft_config import SFTConfig

        cfg = self.config.method
        options = cfg.backend.model_dump(mode="json")
        options.update(
            output_dir=str(output_dir / "checkpoints"),
            learning_rate=cfg.learning_rate, num_train_epochs=cfg.epochs, max_steps=cfg.max_steps,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            gradient_checkpointing=cfg.gradient_checkpointing,
            warmup_steps=cfg.warmup_ratio, weight_decay=cfg.weight_decay,
            logging_steps=cfg.logging_steps, save_steps=cfg.save_steps, save_total_limit=cfg.save_total_limit,
            eval_steps=cfg.eval_steps, use_cpu=cfg.use_cpu, dataloader_num_workers=cfg.dataloader_workers,
            seed=self.config.experiment.seed,
        )
        required = {field.name for field in fields(SFTConfig) if field.init}
        if options.keys() != required:
            raise ValueError(f"SFTConfig fields changed: missing={sorted(required - options.keys())}, "
                             f"unknown={sorted(options.keys() - required)}")
        return SFTConfig(**options)

    def run(self) -> Metrics:
        from transformers import set_seed

        data = self._prepare_run()
        cfg = self.config.method
        if (cfg.backend.eval_strategy != "no" or cfg.backend.do_eval) and not data.validation:
            raise ValueError("Configured evaluation requires a nonempty validation split")
        output = self.config.experiment.output_dir
        args = self.build_arguments(output)
        set_seed(self.config.experiment.seed)
        trainer_class = DataLoaderSFTTrainer
        backend_options = {}
        if cfg.long_context is not None:
            from .long_context import LongContextSFTTrainer

            trainer_class = LongContextSFTTrainer
            backend_options["loss_chunk_size"] = cfg.long_context.loss_chunk_size
        with tracking_session(self.config, is_main_process=int(os.environ.get("RANK", "0")) == 0):
            self.backend = trainer_class(
                model=self.model.load(), args=args, processing_class=cast("ProcessorMixin", self.model.processor),
                data=data, template=self.template, pad_token_id=cast(int, self.model.processor.tokenizer.pad_token_id),
                **backend_options, **cfg.trainer.model_dump(),
            )
            result = self.backend.train()
            metrics: Metrics = dict(result.metrics)
            if cfg.backend.do_eval:
                metrics.update(self.backend.evaluate())
            self.backend.save_state()
            # All ranks participate in gathering the final FSDP weights.
            if self.backend.is_fsdp_enabled:
                fsdp_plugin = cast("FullyShardedDataParallelPlugin", self.backend.accelerator.state.fsdp_plugin)
                fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
            self.backend.save_model(str(output / "final"))
            if self.backend.is_world_process_zero():
                self.backend.save_metrics("train", metrics)
                self._write_json(output / "metrics.json", metrics)
        return metrics


class DataLoaderSFTTrainer(TRLSFTTrainer):
    def __init__(self, *, data: DatasetBundle, template: BaseTemplate, pad_token_id: int, **kwargs):
        self.examples = data.train + data.validation
        self.template = template
        self.pad_token_id = pad_token_id
        self._eval_dataloaders = {}
        super().__init__(
            train_dataset=SFTDataLoader.index_dataset(range(len(data.train))),
            eval_dataset=SFTDataLoader.index_dataset(range(len(data.train), len(self.examples)))
            if data.validation else None,
            data_collator=partial(SFTDataLoader.collate, template=template,
                                  examples=self.examples, pad_token_id=pad_token_id),
            **kwargs,
        )

    def _get_dataloader(self, dataset, description, batch_size, sampler_fn=None,
                        is_training=False, dataloader_key=None):
        sampler = sampler_fn(dataset) if sampler_fn is not None else None
        loader = SFTDataLoader(
            cast("Dataset", dataset), self.template, self.examples, self.pad_token_id,
            args=cast("SFTConfig", self.args),
            batch_size=batch_size, sampler=sampler, is_training=is_training,
        )
        prepared = self.accelerator.prepare(loader)
        if isinstance(sampler, BatchRebalanceSampler):
            # The backend sampler already assigns batches to ranks.
            prepared.batch_sampler.num_processes = 1
            prepared.batch_sampler.process_index = 0
        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            self._eval_dataloaders[dataloader_key] = prepared
        return prepared
