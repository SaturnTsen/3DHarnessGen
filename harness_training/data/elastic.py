"""A global sample order whose optimizer-update groups do not depend on GPU count."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


def epoch_order(size: int, seed: int, epoch: int, strategy: str) -> list[int]:
    if strategy == "sequential":
        return list(range(size))
    return torch.randperm(size, generator=torch.Generator().manual_seed(seed + epoch)).tolist()


class GlobalBatchSampler(Sampler[list[int]]):
    def __init__(self, size: int, seed: int, strategy: str, batch_size: int,
                 accumulation: int, world_size: int, rank: int):
        self.size, self.seed, self.strategy = size, seed, strategy
        self.batch_size, self.accumulation = batch_size, accumulation
        self.world_size, self.rank = world_size, rank
        self.global_batch = batch_size * accumulation * world_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return math.ceil(self.size / self.global_batch) * self.accumulation

    def __iter__(self):
        order = epoch_order(self.size, self.seed, self.epoch, self.strategy)
        # -1 participates in forward/collectives, but has no target labels.
        # Never train an extra real example to equalize rank lengths.
        order += [-1] * (-len(order) % self.global_batch)
        width = self.batch_size * self.world_size
        for start in range(0, len(order), self.global_batch):
            for microstep in range(self.accumulation):
                offset = start + microstep * width + self.rank * self.batch_size
                yield order[offset:offset + self.batch_size]


class IndexedTrainingDataset(Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        return {"example_index": max(index, 0), "elastic_padding": index == -1}


class ElasticCollator:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, examples):
        inputs = self.collator([{"example_index": item["example_index"]} for item in examples])
        for row, item in enumerate(examples):
            if item.get("elastic_padding", False):
                inputs["labels"][row] = self.collator.label_ignore_index
        return inputs


class EpochDataLoader(DataLoader):
    def set_epoch(self, epoch: int):
        cast(GlobalBatchSampler, self.batch_sampler).set_epoch(epoch)
