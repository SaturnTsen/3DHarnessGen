import json
from pathlib import Path

import pytest
import torch

from harness_training.config import DataConfig, TemplateConfig
from harness_training.data.trajectory import TrajectoryDataset
from harness_training.templates.base import SampleTooLong, TemplateError
from harness_training.templates.qwen import QwenTemplate
from harness_training.training.collator import SFTCollator


TOOLS = Path(__file__).resolve().parents[2] / "configs/tools/agent_runtime.json"


def test_target_only_labels_and_padding(config, processor):
    data = TrajectoryDataset(config.data, config.experiment.scene).prepare()
    template = QwenTemplate(config.template, processor)
    first = template.encode(data.train[0])
    second = template.encode(data.train[1])
    rendered = template.render(data.train[0])
    assert rendered.text.startswith(rendered.prompt)
    assert first["labels"][0, 0] == -100
    target_ids = first["labels"][first["labels"] != -100]
    target = processor.tokenizer.decode(target_ids)
    assert "print('first')" in target and "Write a Blender script" not in target
    assert processor.tokenizer.eos_token_id in target_ids
    batch = SFTCollator(template, processor.tokenizer.pad_token_id)([{"example": e} for e in data.train])
    assert batch["input_ids"].shape[0] == 2
    assert batch["image_grid_thw"].shape[0] == 2
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)
    assert batch["labels"].shape[1] == max(first["labels"].shape[1], second["labels"].shape[1])


def test_no_silent_truncation(config, processor):
    example = TrajectoryDataset(config.data, "Single-view").prepare().train[0]
    cfg = config.template.model_copy(update={"max_length": 16})
    with pytest.raises(SampleTooLong):
        QwenTemplate(cfg, processor).encode(example)


def interactive_record(tmp_path, *, two_chains=False):
    from PIL import Image

    path = tmp_path / "ActiveVisual/gpt-6-astra/ObjectFactory/post_train_trajectory/trajectory.jsonl"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(path.parent / "shot.png")
    conversations = [
        {"from": "human", "value": "<recommended_plugins>Ignore this list</recommended_plugins>Build the object."},
        {"from": "gpt", "value": "Inspect first.\n<tool_call>\n<function=exec>\n<parameter=input>\ntext('inspect');\n</parameter>\n</function>\n</tool_call>"},
        {"from": "observation", "value": "Current screenshot: <image>"},
        {"from": "function_call", "value": json.dumps({"name": "exec", "arguments": {"input": "text('build');"}})},
        {"from": "observation", "value": "Completed."},
        {"from": "gpt", "value": "Saved the reconstruction."},
    ]
    mapping = [
        {"source_message_indices": [1], "tool_call_ids": [], "observation_call_ids": []},
        {"source_message_indices": [2, 3], "tool_call_ids": ["a"], "observation_call_ids": []},
        {"source_message_indices": [4], "tool_call_ids": [], "observation_call_ids": ["a"]},
        {"source_message_indices": [5], "tool_call_ids": ["b"], "observation_call_ids": []},
        {"source_message_indices": [6], "tool_call_ids": [], "observation_call_ids": ["b"]},
        {"source_message_indices": [7], "tool_call_ids": [], "observation_call_ids": []},
    ]
    chains = [{"message_start": 1, "message_end": 8}]
    if two_chains:
        conversations += [{"from": "human", "value": "A separate attempt."}, {"from": "gpt", "value": "Another final reply."}]
        mapping += [{"source_message_indices": [8], "tool_call_ids": [], "observation_call_ids": []},
                    {"source_message_indices": [9], "tool_call_ids": [], "observation_call_ids": []}]
        chains += [{"message_start": 8, "message_end": 10}]
    row = {"system": "Reconstruct the object.", "conversations": conversations, "images": ["shot.png"],
           "metadata": {"chains": chains, "sharegpt_conversion": {"conversation_source_mapping": mapping,
                        "mixed_assistant_tool_format": "qwen3_8"}}}
    path.write_text(json.dumps(row) + "\n")
    return path


def test_tool_screenshot_and_native_mixed_reply(tmp_path, processor):
    interactive_record(tmp_path)
    data = TrajectoryDataset(DataConfig(runs_root=tmp_path, validation_fraction=0, test_fraction=0), "ActiveVisual").prepare()
    assert [e.target_index for e in data.train] == [1, 3, 5]
    template = QwenTemplate(TemplateConfig(tools_path=TOOLS, min_image_pixels=1024, max_image_pixels=1024), processor)
    first = template.render(data.train[0])
    assert "Ignore this list" not in first.prompt
    assert len(first.images) == 0
    second = template.encode(data.train[1])
    assert second["image_grid_thw"].shape[0] == 1
    target = processor.tokenizer.decode(second["labels"][second["labels"] != -100])
    assert "text('build');" in target and "Current screenshot" not in target
    assert "text('inspect');" not in target


def test_chain_history_does_not_leak(tmp_path):
    interactive_record(tmp_path, two_chains=True)
    data = TrajectoryDataset(DataConfig(runs_root=tmp_path, validation_fraction=0, test_fraction=0), "ActiveVisual").prepare()
    assert data.train[-1].indices == (6, 7)


def test_unknown_tool_and_unmatched_observation_fail(tmp_path, processor):
    path = interactive_record(tmp_path)
    cfg = DataConfig(runs_root=tmp_path, validation_fraction=0, test_fraction=0)
    template = QwenTemplate(TemplateConfig(tools_path=TOOLS), processor)
    row = json.loads(path.read_text())
    row["metadata"]["sharegpt_conversion"]["conversation_source_mapping"][2]["observation_call_ids"] = ["wrong"]
    path.write_text(json.dumps(row))
    data = TrajectoryDataset(cfg, "ActiveVisual").prepare()
    with pytest.raises(TemplateError, match="matched"):
        template.render(data.train[1])
