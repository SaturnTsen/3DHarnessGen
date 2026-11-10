"""Keep indexed sample loading and multimodal batch assembly together."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, TypedDict, cast

import torch
from datasets import Dataset
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Sampler
from torch.utils.data import Dataset as TorchDataset
from transformers.trainer_pt_utils import BatchRebalanceSampler
from transformers.trainer_utils import seed_worker

from harness_training.data.base import TrainingExample
from harness_training.chat_templates.base import BaseTemplate
from harness_training.types import TensorInputs

if TYPE_CHECKING:
    from trl.trainer.sft_config import SFTConfig


class IndexedSample(TypedDict):
    example_index: int


class SFTDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, template: BaseTemplate, examples: list[TrainingExample],
                 pad_token_id: int, *, args: SFTConfig, batch_size: int,
                 sampler: Sampler | None, is_training: bool):
        options = {
            "num_workers": args.dataloader_num_workers,
            "pin_memory": args.dataloader_pin_memory,
            "persistent_workers": args.dataloader_persistent_workers,
            "multiprocessing_context": args.dataloader_multiprocessing_context,
            "prefetch_factor": args.dataloader_prefetch_factor,
            "in_order": args.dataloader_in_order,
        }
        if isinstance(sampler, BatchRebalanceSampler):
            options["batch_sampler"] = sampler
        else:
            options.update(batch_size=batch_size, sampler=sampler, drop_last=args.dataloader_drop_last)
        if is_training:
            options["worker_init_fn"] = partial(
                seed_worker, num_workers=args.dataloader_num_workers, rank=args.process_index,
            )
        super().__init__(
            cast(TorchDataset[IndexedSample], dataset),
            collate_fn=partial(self.collate, template=template, examples=examples, pad_token_id=pad_token_id),
            **options,
        )

    @staticmethod
    def index_dataset(indices: range) -> Dataset:
        return Dataset.from_dict({"example_index": list(indices)})

    @staticmethod
    def collate(samples: list[IndexedSample], *, template: BaseTemplate,
                examples: list[TrainingExample], pad_token_id: int) -> TensorInputs:
        encoded = [template.encode(examples[item["example_index"]]) for item in samples]
        length = max(item["input_ids"].shape[1] for item in encoded)
        cfg = template.config
        sequence_values = {
            "input_ids": pad_token_id,
            "labels": cfg.label_ignore_index,
            "attention_mask": cfg.attention_padding_value,
            "token_type_ids": cfg.token_type_padding_value,
            "mm_token_type_ids": cfg.token_type_padding_value,
        }
        result: TensorInputs = {}
        for key in set().union(*(item.keys() for item in encoded)):
            if key in sequence_values:
                values = []
                for item in encoded:
                    tensor = item.get(key)
                    if tensor is None:
                        if key not in {"token_type_ids", "mm_token_type_ids"}:
                            raise ValueError(f"Missing sequence input: {key}")
                        tensor = torch.full_like(item["input_ids"], cfg.token_type_padding_value)
                    values.append(pad(tensor, (0, length - tensor.shape[1]), value=sequence_values[key]))
                result[key] = torch.cat(values, dim=0)
            elif key in {"pixel_values", "image_grid_thw"}:
                result[key] = torch.cat([item[key] for item in encoded if key in item], dim=0)
            else:
                raise ValueError(f"Unsupported Qwen processor output: {key}")
        return result
