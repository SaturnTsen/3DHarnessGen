"""Verify the project's loader preserves multimodal batches and backend sampling."""

import json
from functools import partial
from typing import cast

import pytest
import torch
from torch.utils.data import DataLoader, SequentialSampler

from harness_training.data.trajectory import TrajectoryDataset
from harness_training.dataloader import SFTDataLoader
from harness_training.chat_templates.qwen import QwenTemplate
from harness_training.trainers.sft_trainer import DataLoaderSFTTrainer, SFTTrainer


@pytest.mark.parametrize("workers", [0, 1])
def test_loader_preserves_repeated_images_labels_and_order(config, processor, workers):
    from PIL import Image

    source = next(config.data.runs_root.rglob('trajectory.jsonl'))
    Image.new('RGB', (64, 32), 'red').save(source.parent / 'images/second.png')
    row = json.loads(source.read_text())
    row['messages'][1]['content'].extend([
        {'type': 'image_url', 'image_url': {'url': 'images/second.png'}},
        {'type': 'image_url', 'image_url': {'url': 'images/reference.png'}},
    ])
    source.write_text(json.dumps(row) + '\n')
    data = TrajectoryDataset(config.data, config.experiment.scene, config.experiment.seed).prepare()
    template = QwenTemplate(config.template, processor)
    expected = [template.encode(example) for example in data.train]
    config.method.use_cpu = True
    config.method.dataloader_workers = workers
    config.method.backend.dataloader_multiprocessing_context = 'spawn' if workers else None
    config.method.backend.dataloader_persistent_workers = bool(workers)
    config.method.backend.dataloader_prefetch_factor = 2 if workers else None
    args = SFTTrainer(config).build_arguments(config.experiment.output_dir)
    loader = SFTDataLoader(
        SFTDataLoader.index_dataset(range(len(data.train))), template, data.train,
        processor.tokenizer.pad_token_id, args=args, batch_size=2, sampler=None, is_training=False,
    )
    assert isinstance(loader, DataLoader)
    for _ in range(2):
        batch, = list(loader)
        for index, item in enumerate(expected):
            length = item['input_ids'].shape[1]
            assert torch.equal(batch['input_ids'][index, :length], item['input_ids'][0])
            assert torch.equal(batch['labels'][index, :length], item['labels'][0])
        assert torch.all(batch['labels'][batch['attention_mask'] == 0] == -100)
        for key in ('pixel_values', 'image_grid_thw'):
            assert torch.equal(batch[key], torch.cat([item[key] for item in expected]))


def test_backend_uses_loader_for_training_and_validation(config, tiny_qwen_path):
    from harness_training.models.qwen import QwenModel

    config.model.path = tiny_qwen_path
    config.method.use_cpu = True
    config.method.gradient_checkpointing = False
    config.method.backend.train_sampling_strategy = 'sequential'
    data = TrajectoryDataset(config.data, config.experiment.scene, config.experiment.seed).prepare()
    data.validation = data.train[1:]
    data.train = data.train[:1]
    model = QwenModel(config.model, config.template)
    template = QwenTemplate(config.template, model.processor)
    backend = DataLoaderSFTTrainer(
        model=model.load(), args=SFTTrainer(config).build_arguments(
            config.experiment.output_dir),
        processing_class=model.processor, data=data, template=template,
        pad_token_id=cast(int, model.processor.tokenizer.pad_token_id), **config.method.trainer.model_dump(),
    )
    assert isinstance(backend._get_train_sampler(), SequentialSampler)
    for loader, examples in ((backend.get_train_dataloader(), data.train),
                             (backend.get_eval_dataloader(), data.validation)):
        collate_fn = loader.collate_fn
        assert isinstance(collate_fn, partial)
        assert collate_fn.func is SFTDataLoader.collate
        actual = next(iter(loader))
        expected = template.encode(examples[0])
        assert torch.equal(actual['labels'], expected['labels'])
        assert torch.equal(actual['pixel_values'], expected['pixel_values'])
