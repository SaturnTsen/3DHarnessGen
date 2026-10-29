"""SFT checkpointing with topology-independent data and trainable state."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal, cast

import torch
from accelerate.optimizer import AcceleratedOptimizer
from datasets import Dataset
from torch.optim.lr_scheduler import LRScheduler
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_optimizer_state_dict,
    set_model_state_dict, set_optimizer_state_dict,
)
from transformers import Trainer, set_seed
from trl.trainer.sft_trainer import SFTTrainer

from harness_training.data.elastic import (
    ElasticCollator, EpochDataLoader, GlobalBatchSampler, IndexedTrainingDataset, epoch_order,
)
from harness_training.training.elastic import DataProgress, target_hash
from .long_context import LongContextSFTTrainer


class ElasticTrainerMixin(SFTTrainer):
    def __init__(self, *args, resume_manifest, resume_progress, **kwargs):
        self.resume_manifest = resume_manifest
        self.resume_progress = resume_progress
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        sampler = GlobalBatchSampler(
            len(cast(Dataset, self.train_dataset)), self.args.data_seed if self.args.data_seed is not None else self.args.seed,
            self.args.train_sampling_strategy, self.args.per_device_train_batch_size,
            self.args.gradient_accumulation_steps, self.args.world_size, self.args.process_index,
        )
        # Already split by rank. Do not let Accelerate shard it a second time.
        return EpochDataLoader(
            IndexedTrainingDataset(len(cast(Dataset, self.train_dataset))), batch_sampler=sampler,
            collate_fn=ElasticCollator(self.data_collator), num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor,
            generator=torch.Generator().manual_seed(self.args.seed),
        )

    def _init_training_state(self, *args, **kwargs):
        epochs, batches = super()._init_training_state(*args, **kwargs)
        if self.resume_progress is not None:
            progress = self.resume_progress
            epochs = progress.epoch
            batches = (progress.next_sample // progress.global_batch_size) * self.args.gradient_accumulation_steps
        return epochs, batches

    def _state_model(self):
        return self.accelerator.unwrap_model(self.model_wrapped)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None) -> None:
        checkpoint = Path(resume_from_checkpoint)
        if (checkpoint / "portable_state.pt").is_file():
            # Restored together with the optimizer, after FSDP has wrapped parameters.
            return
        if (checkpoint / "pytorch_model_fsdp_0/.metadata").is_file():
            from accelerate.utils import load_fsdp_model

            if self.is_fsdp_enabled:
                load_fsdp_model(self.accelerator.state.fsdp_plugin, self.accelerator,
                                self.model_wrapped, str(checkpoint), adapter_only=True)
                return
            from torch.distributed.checkpoint.state_dict_loader import load
            from torch.distributed.checkpoint.filesystem import FileSystemReader

            base = self._state_model()
            metadata = FileSystemReader(checkpoint / "pytorch_model_fsdp_0").read_metadata()
            adapter = base.active_adapter
            names = {}
            for name, parameter in base.named_parameters():
                if parameter.requires_grad:
                    key = name if f"model.{name}" in metadata.state_dict_metadata else name.replace(f".{adapter}.", ".")
                    names[key] = name
            weights = {key: base.get_parameter(name).detach().clone() for key, name in names.items()}
            load({"model": weights}, checkpoint_id=checkpoint / "pytorch_model_fsdp_0")
            set_model_state_dict(base, {names[key]: tensor for key, tensor in weights.items()},
                                 options=StateDictOptions(strict=False))
            return
        if not self.is_fsdp_enabled:
            return super()._load_from_checkpoint(resume_from_checkpoint, model)
        from peft.utils.save_and_load import load_peft_weights

        base = self._state_model()
        adapter = base.active_adapter
        saved = load_peft_weights(str(checkpoint), device="cpu", local_files_only=True)
        weights = {name: saved[name.replace(f".{adapter}.", ".")] for name, parameter in base.named_parameters()
                   if parameter.requires_grad}
        set_model_state_dict(base, weights, options=StateDictOptions(full_state_dict=True, strict=False))

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is None:
            return
        path = Path(checkpoint)
        model = self._state_model()
        optimizer = cast(AcceleratedOptimizer, self.optimizer).optimizer
        scheduler = cast(LRScheduler, self.lr_scheduler)
        options = StateDictOptions(full_state_dict=True, strict=False)
        if (path / "portable_state.pt").is_file():
            saved = torch.load(path / "portable_state.pt", map_location="cpu", weights_only=True)
            set_model_state_dict(model, saved["model"], options=options)
            set_optimizer_state_dict(model, optimizer, saved["optimizer"], options=options)
            scheduler.load_state_dict(saved["scheduler"])
        elif (path / "optimizer_0/.metadata").is_file():
            if self.is_fsdp_enabled:
                super()._load_optimizer_and_scheduler(checkpoint)
            else:
                from torch.distributed.checkpoint.state_dict_loader import load

                state = get_optimizer_state_dict(model, optimizer)
                load({"optimizer": state}, checkpoint_id=path / "optimizer_0")
                set_optimizer_state_dict(model, optimizer, state)
                scheduler.load_state_dict(torch.load(path / "scheduler.pt", weights_only=True))
        else:
            # Native single/DDP integer parameter IDs -> canonical parameter names.
            saved = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
            named = cast(dict[str, Any], get_optimizer_state_dict(model, optimizer, options=options))
            if len(saved["param_groups"]) != len(named["param_groups"]):
                raise ValueError("Legacy optimizer parameter groups changed")
            state = {}
            groups = []
            for old, current in zip(saved["param_groups"], named["param_groups"], strict=True):
                if len(old["params"]) != len(current["params"]):
                    raise ValueError("Legacy optimizer parameter ordering changed")
                state.update({name: saved["state"][index] for index, name in
                              zip(old["params"], current["params"], strict=True) if index in saved["state"]})
                groups.append({**old, "params": current["params"]})
            set_optimizer_state_dict(model, optimizer, {"state": state, "param_groups": groups}, options=options)
            scheduler.load_state_dict(torch.load(path / "scheduler.pt", weights_only=True))

    def _load_rng_state(self, checkpoint):
        if self.resume_progress is not None and self.resume_progress.world_size != self.args.world_size:
            # A new topology cannot reproduce per-device dropout streams bit for bit.
            # Sample order has its own generator and remains identical.
            set_seed(self.args.seed + self.state.global_step * self.args.world_size + self.args.process_index)
        else:
            super()._load_rng_state(checkpoint)

    def _save_checkpoint(self, model, trial):
        checkpoint = Path(cast(str, self.args.output_dir)) / f"checkpoint-{self.state.global_step}"
        if self.is_world_process_zero():
            checkpoint.mkdir(parents=True, exist_ok=True)
            (checkpoint / "resume_manifest.json").write_text(json.dumps(self.resume_manifest))
        self.accelerator.wait_for_everyone()
        super()._save_checkpoint(model, trial)
        self.accelerator.wait_for_everyone()
        base = self._state_model()
        options = StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=True)
        from torch.distributed.tensor import DTensor

        # Enumerate only trainable parameters: bitsandbytes state_dict contains
        # quantization metadata keys which are not real parameter FQNs.
        weights = {}
        for name, parameter in base.named_parameters():
            if parameter.requires_grad:
                tensor = parameter.detach()
                if isinstance(tensor, DTensor):
                    tensor = tensor.full_tensor()
                if self.is_world_process_zero():
                    weights[name] = tensor.cpu()
        optimizer = get_optimizer_state_dict(base, cast(AcceleratedOptimizer, self.optimizer).optimizer, options=options)
        if self.is_world_process_zero():
            checkpoint = Path(cast(str, self.args.output_dir)) / f"checkpoint-{self.state.global_step}"
            temporary = checkpoint / "portable_state.pt.tmp"
            torch.save({"model": weights, "optimizer": optimizer,
                        "scheduler": cast(LRScheduler, self.lr_scheduler).state_dict()}, temporary)
            temporary.replace(checkpoint / "portable_state.pt")
            count = len(cast(Dataset, self.train_dataset))
            global_batch = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
            epoch, updates = divmod(self.state.global_step, math.ceil(count / global_batch))
            seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
            progress = DataProgress(version=1, global_step=self.state.global_step, epoch=epoch,
                                    next_sample=updates * global_batch, sample_count=count,
                                    global_batch_size=global_batch, world_size=self.args.world_size,
                                    seed=seed, strategy=cast(Literal["random", "sequential"], self.args.train_sampling_strategy),
                                    order=epoch_order(count, seed, epoch, self.args.train_sampling_strategy),
                                    target_hash=target_hash(self.resume_manifest["targets"]["train"]))
            temporary = checkpoint / "data_progress.json.tmp"
            temporary.write_text(progress.model_dump_json())
            temporary.replace(checkpoint / "data_progress.json")
        self.accelerator.wait_for_everyone()


class ElasticSFTTrainer(ElasticTrainerMixin):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if model.training:
            # Token-normalized NLL also handles ranks with an entirely masked tail.
            return Trainer.compute_loss(self, model, inputs, return_outputs, num_items_in_batch)
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


class ElasticLongContextSFTTrainer(ElasticTrainerMixin, LongContextSFTTrainer):
    pass
