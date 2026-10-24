import json

import pytest
import torch

from harness_training.data.trajectory import TrajectoryDataset
from harness_training.templates.base import SampleTooLong, TemplateError
from harness_training.templates.qwen import QwenTemplate
from harness_training.training.collator import SFTCollator


def test_target_only_labels_and_padding(config, processor):
    data = TrajectoryDataset(config.data, config.experiment.scene, config.experiment.seed).prepare()
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
    batch = SFTCollator(template, processor.tokenizer.pad_token_id, data.train,
                        label_ignore_index=config.template.label_ignore_index,
                        attention_padding_value=config.template.attention_padding_value,
                        token_type_padding_value=config.template.token_type_padding_value)([{"example": e} for e in data.train])
    assert batch["input_ids"].shape[0] == 2
    assert batch["image_grid_thw"].shape[0] == 2
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)
    assert batch["labels"].shape[1] == max(first["labels"].shape[1], second["labels"].shape[1])


def test_no_silent_truncation(config, processor):
    example = TrajectoryDataset(config.data, "Single-view", config.experiment.seed).prepare().train[0]
    cfg = config.template.model_copy(update={"max_length": 16})
    with pytest.raises(SampleTooLong):
        QwenTemplate(cfg, processor).encode(example)


def interactive_record(tmp_path, *, two_chains=False, instance="ObjectFactory"):
    from PIL import Image
    path = tmp_path / f"ActiveVisual/gpt-6-astra/{instance}/post_train_trajectory/trajectory.jsonl"
    (path.parent / "images").mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(path.parent / "images/shot.png")
    source = {"path": "source.json", "sha256": "a" * 64, "locator": "/"}
    custom = {"type": "custom", "name": "exec", "description": "Execute code", "format": {"type": "text"}}
    messages = [
        {"index": 0, "role": "developer", "content": "Reconstruct the object."},
        {"index": 1, "role": "developer", "type": "additional_tools", "tools": [
            {"type": "namespace", "name": "functions", "description": "Execution", "tools": [custom]}]},
        {"index": 2, "role": "user", "content": "Build the object."},
        {"index": 3, "type": "custom_tool_call", "call_id": "a", "name": "exec", "input": "text('inspect');"},
        {"index": 4, "type": "custom_tool_call_output", "call_id": "a", "output": [
            {"type": "input_text", "text": "Current screenshot:"},
            {"type": "input_image", "image_url": "images/shot.png"},
            {"type": "input_text", "text": "Repeated screenshot:"},
            {"type": "input_image", "image_url": "images/shot.png"}]},
        {"index": 5, "type": "custom_tool_call", "call_id": "b", "name": "exec", "input": "text('build');"},
        {"index": 6, "type": "custom_tool_call_output", "call_id": "b", "output": "Completed."},
        {"index": 7, "role": "assistant", "content": "Saved the reconstruction."},
    ]
    def step(id, target, indices, kind):
        return {"id": id, "context_indices": indices, "target_index": target, "target_kind": kind,
                "status": "OK", "source": source}
    steps = [step("inspect", 3, [0, 1, 2], "tool_call"),
             step("build", 5, [0, 1, 2, 3, 4], "tool_call"),
             step("final", 7, list(range(7)), "final_reply")]
    if two_chains:
        messages += [{"index": 8, "role": "user", "content": "A separate attempt."},
                     {"index": 9, "role": "assistant", "content": "Another final reply."}]
        steps.append(step("separate", 9, [0, 1, 8], "final_reply"))
    row = {"schema_version": 3, "model": "gpt-6-astra", "messages": messages,
           "tools": [[{"type": "custom", "custom": {k: v for k, v in custom.items() if k != "type"},
                       "namespace": "functions", "namespace_description": "Execution"}]],
           "metadata": {"scene": "ActiveVisual", "instance": instance, "session_id": "session",
                        "view": "agent", "sources": [source], "steps": steps}}
    path.write_text(json.dumps(row) + "\n")
    return path


def test_tool_screenshot_and_custom_reply(tmp_path, processor, config):
    interactive_record(tmp_path)
    cfg = config.data.model_copy(update={"runs_root": tmp_path})
    data = TrajectoryDataset(cfg, "ActiveVisual", config.experiment.seed).prepare()
    assert [e.target_index for e in data.train] == [3, 5, 7]
    template = QwenTemplate(config.template, processor)
    first = template.render(data.train[0])
    assert "Reconstruct the object." in first.prompt
    assert len(first.images) == 0
    assert "Current screenshot" not in first.text
    second = template.encode(data.train[1])
    assert second["image_grid_thw"].shape[0] == 2
    rendered = template.render(data.train[1])
    assert rendered.images[0] == rendered.images[1]
    target = processor.tokenizer.decode(second["labels"][second["labels"] != -100])
    assert "text('build');" in target and "Current screenshot" not in target
    assert "text('inspect');" not in target
    messages, _, tools = template._messages(data.train[1])
    assert messages[-2]["role"] == "tool" and messages[-2].get("tool_call_id") == "a"
    assert tools is not None
    assert tools[0]["function"].get("parameters", {}).get("required") == ["input"]


def test_selected_history_does_not_leak(tmp_path, config):
    interactive_record(tmp_path, two_chains=True)
    cfg = config.data.model_copy(update={"runs_root": tmp_path})
    data = TrajectoryDataset(cfg, "ActiveVisual", config.experiment.seed).prepare()
    assert data.train[-1].indices == (0, 1, 8, 9)


def test_unmatched_observation_is_rejected(tmp_path, config):
    path = interactive_record(tmp_path)
    row = json.loads(path.read_text())
    row["messages"][4]["call_id"] = "wrong"
    path.write_text(json.dumps(row))
    cfg = config.data.model_copy(update={"runs_root": tmp_path})
    data = TrajectoryDataset(cfg, "ActiveVisual", config.experiment.seed).prepare()
    assert not data.train
    assert "missing earlier call" in data.rejected[0]["reason"]


def test_unknown_custom_mapping_is_rejected(tmp_path, processor, config):
    interactive_record(tmp_path)
    cfg = config.data.model_copy(update={"runs_root": tmp_path})
    data = TrajectoryDataset(cfg, "ActiveVisual", config.experiment.seed).prepare()
    config.template.custom_tools = {}
    with pytest.raises(TemplateError, match="No custom tool mapping"):
        QwenTemplate(config.template, processor).render(data.train[0])


def test_selected_tool_context_and_instructions_exclude_future(tmp_path, config, processor):
    import copy

    path = interactive_record(tmp_path)
    row = json.loads(path.read_text())
    declaration = copy.deepcopy(row["messages"][1])
    declaration["index"] = 7
    declaration["tools"][0]["tools"][0]["description"] = "Later definition"
    row["messages"].insert(7, declaration)
    row["messages"].insert(8, {"index": 8, "role": "developer", "content": "Future instruction"})
    row["messages"][9]["index"] = 9
    row["metadata"]["steps"][-1].update(target_index=9, context_indices=[0, 2, 3, 4, 5, 6, 7, 8])
    context = copy.deepcopy(row["tools"][0])
    context[0]["custom"]["description"] = "Later definition"
    row["tools"].append(context)
    path.write_text(json.dumps(row))
    cfg = config.data.model_copy(update={"runs_root": tmp_path})
    data = TrajectoryDataset(cfg, "ActiveVisual", config.experiment.seed).prepare()
    assert not data.rejected
    template = QwenTemplate(config.template, processor)
    early = template.render(data.train[0])
    later = template.render(data.train[-1])
    assert "Later definition" not in early.text and "Future instruction" not in early.text
    assert "Later definition" in later.prompt and "Future instruction" in later.prompt
    assert later.prompt.count('"name": "exec"') == 1


def test_function_arguments_preserve_json_types(tmp_path, config, processor):
    path = interactive_record(tmp_path)
    row = json.loads(path.read_text())
    definition = {"name": "exec", "description": "Function", "parameters": {
        "type": "object", "properties": {"nested": {"type": "object"}}}}
    row["tools"][0][0].update(type="function", function=definition)
    del row["tools"][0][0]["custom"]
    row["messages"][1]["tools"][0]["tools"] = [{"type": "function", **definition}]
    arguments = {"nested": {"boolean": True, "number": 3, "null": None}, "code": "text('x');\ntext(2);"}
    for index, call_id in [(3, "a"), (5, "b")]:
        row["messages"][index] = {"index": index, "type": "function_call", "call_id": call_id,
                                   "name": "exec", "arguments": json.dumps(arguments)}
        row["messages"][index + 1]["type"] = "function_call_output"
    path.write_text(json.dumps(row))
    cfg = config.data.model_copy(update={"runs_root": tmp_path})
    data = TrajectoryDataset(cfg, "ActiveVisual", config.experiment.seed).prepare()
    template = QwenTemplate(config.template, processor)
    messages, _, _ = template._messages(data.train[0])
    calls = messages[-1].get("tool_calls")
    assert calls is not None
    assert calls[0]["function"]["arguments"] == arguments
    encoded = template.encode(data.train[0])
    target = processor.tokenizer.decode(encoded["labels"][encoded["labels"] != -100])
    assert "text('x');\ntext(2);" in target
