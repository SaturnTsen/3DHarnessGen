import copy
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from harness_training.config import DataConfig
from harness_training.data.contract import AdditionalTools, Trajectory
from harness_training.data.trajectory import TrajectoryDataset
from scripts.clean_astra_trajectories import (
    UNUSED_TOOLS, clean_dataset, clean_targets, failure_response, failure_text, image_blocks,
    prune_declaration, used_tool_names,
)


def agent_record(round_count: int = 8, failed_round: int | None = 3):
    source = {"path": "../evidence.json", "sha256": "a" * 64, "locator": "/"}
    declaration = {
        "index": 0, "role": "developer", "type": "additional_tools", "tools": [
            {"type": "namespace", "name": "functions", "description": "", "tools": [
                {"type": "custom", "name": "exec", "format": {"type": "text"}, "description":
                 "Execute JavaScript.\n### `create_goal`\nGoal management.\n"
                 "### `exec_command`\nExecute shell command.\n## image_gen\n"
                 "### `image_gen__imagegen`\nGenerate images.\n## web\n"
                 "### `web__run`\nSearch.\n## Citations\nCitation rules.\n"},
                {"type": "function", "name": "request_user_input", "description": "Ask user.",
                 "parameters": {"type": "object"}},
            ]},
            {"type": "namespace", "name": "collaboration", "description": "Agents", "tools": [
                {"type": "function", "name": "spawn_agent", "description": "Spawn.",
                 "parameters": {"type": "object"}},
            ]},
        ],
    }
    initial = {"type": "image_url", "image_url": {"url": "images/initial.png"}}
    messages = [declaration, {"index": 1, "role": "developer", "content": "Preserve the reference."},
                {"index": 2, "role": "user", "content": [
                    {"type": "text", "text": "Reconstruct this object."}, initial, copy.deepcopy(initial),
                ]}]
    steps = []
    for turn in range(1, round_count + 1):
        messages.append({"index": len(messages), "role": "assistant", "content": f"Working on round {turn}."})
        index = len(messages)
        messages.append({"index": index, "role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{turn}", "type": "custom", "custom": {
                "name": "exec", "input": f"text(await tools.exec_command({{cmd:'step {turn}'}}));",
            }},
        ]})
        steps.append({"id": f"step_{index}", "context_indices": list(range(index)),
                      "target_index": index, "target_kind": "tool_call", "status": "UNKNOWN", "source": source})
        messages.append({"index": len(messages), "role": "tool", "tool_call_id": f"call_{turn}", "content": [
            {"type": "text", "text": json.dumps({"isError": turn == failed_round, "result": {"turn": turn}})},
            {"type": "image_url", "image_url": {"url": f"images/round_{turn}.png"}},
        ]})
    final_index = len(messages)
    messages.append({"index": final_index, "role": "assistant", "content": "Complete."})
    steps.append({"id": "final", "context_indices": list(range(final_index)), "target_index": final_index,
                  "target_kind": "final_reply", "status": "UNKNOWN", "source": source})
    return {"schema_version": 3, "model": "gpt-6-astra", "messages": messages,
            "tools": [[t.model_dump(exclude_unset=True) for t in AdditionalTools.model_validate(declaration).tool_context()]],
            "metadata": {"scene": "ActiveVisual", "instance": "object", "view": "agent", "session_id": "session",
                         "sources": [copy.deepcopy(source)], "steps": copy.deepcopy(steps)}}


def write_source(tmp_path, row):
    source = tmp_path / "runs/ActiveVisual/gpt-6-astra/object/post_train_trajectory/trajectory.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(row) + "\n")
    references = {b["image_url"]["url"] for b in image_blocks(row)}
    for i, reference in enumerate(sorted(references)):
        image = source.parent / reference
        image.parent.mkdir(exist_ok=True)
        Image.new("RGB", (4, 4), (i * 17, 0, 0)).save(image)
    return source


def test_prune_unused_declarations_preserves_execution_and_source():
    original = agent_record()["messages"][0]
    before = copy.deepcopy(original)
    cleaned = prune_declaration(original, set(UNUSED_TOOLS))
    assert [ns["name"] for ns in cleaned["tools"]] == ["functions"]
    assert [t["name"] for t in cleaned["tools"][0]["tools"]] == ["exec"]
    description = cleaned["tools"][0]["tools"][0]["description"]
    assert "### `exec_command`\nExecute shell command." in description
    assert "web" not in description and "image_gen" not in description and "create_goal" not in description
    assert "Citation rules" not in description
    assert original == before


def test_each_target_gets_last_five_successful_rounds_and_no_future():
    row = agent_record()
    before = copy.deepcopy(row)
    outputs = list(clean_targets(row, 5, set(UNUSED_TOOLS)))
    final, info = outputs[-1]
    assert final is not None
    assert len(info["history_tool_rounds"]) == 5
    assert [m["tool_call_id"] for m in final["messages"] if m["role"] == "tool"] == [
        "call_4", "call_5", "call_6", "call_7", "call_8",
    ]
    assert final["messages"][2] == row["messages"][2]
    assert "Working on round 4." in [m.get("content") for m in final["messages"]]
    assert "Working on round 2." not in [m.get("content") for m in final["messages"]]
    references = [b["image_url"]["url"] for b in image_blocks(final)]
    assert references[:2] == ["images/initial.png", "images/initial.png"]
    assert "images/round_2.png" not in references and "images/round_3.png" not in references
    assert sum(cleaned is None for cleaned, _ in outputs) == 1
    for cleaned, entry in outputs:
        if cleaned is None:
            continue
        Trajectory.model_validate(cleaned)
        indices = entry["selected_original_indices"]
        assert indices[-1] == entry["original_target_index"]
        assert all(index < indices[-1] for index in indices[:-1])
        original_target = {k: v for k, v in row["messages"][indices[-1]].items() if k != "index"}
        target = {k: v for k, v in cleaned["messages"][-1].items() if k != "index"}
        assert original_target == target
        assert len(entry["history_tool_rounds"]) <= 5
    assert row == before


def test_error_assistant_response_is_removed_from_later_history():
    row = agent_record(1, failed_round=None)
    call_index = row["metadata"]["steps"][0]["target_index"]
    status_index = call_index - 1
    row["metadata"]["steps"].insert(0, {
        "id": "failed-status", "context_indices": list(range(status_index)),
        "target_index": status_index, "target_kind": "status", "status": "ERROR",
        "source": copy.deepcopy(row["metadata"]["sources"][0]),
    })
    outputs = list(clean_targets(row, 5, set(UNUSED_TOOLS)))
    assert outputs[0][0] is None
    for cleaned, info in outputs[1:]:
        assert cleaned is not None
        assert status_index not in info["selected_original_indices"]
        Trajectory.model_validate(cleaned)


@pytest.mark.parametrize(("text", "failed"), [
    ("Script failed\nWall time 0.1 seconds\nOutput:\nSyntaxError", True),
    ('Script completed\nOutput:\n{"isError":true}', True),
    ('{"content":[{"type":"text","text":"{\\"status\\":\\"error\\"}"}]}', True),
    ('{"exit_code":1,"output":"bad command"}', True),
    ('{"status":"fulfilled","value":{"isError":true}}', True),
    ('{"isError":false,"result":{"status":"ok"}}', False),
    ('{"exit_code":0,"output":"error handling documentation"}', False),
    ('Documentation: set isError=true on failure.', False),
    ('print("Script failed")', False),
    ('{"result":{"measurement_error":0.01}}', False),
])
def test_failure_detection_uses_result_flags(text, failed):
    assert failure_text(text) is failed


@pytest.mark.parametrize("failed", [True, False])
def test_failure_response_keeps_viewport_label_and_result_boundaries(failed):
    message = {"role": "tool", "content": [
        {"type": "text", "text": "Script completed\nOutput:\n"},
        {"type": "text", "text": "front GT"},
        {"type": "image_url", "image_url": {"url": "images/front.png"}},
        {"type": "text", "text": "front reconstruction"},
        {"type": "text", "text": json.dumps({"content": [
            {"type": "text", "text": json.dumps({"status": "error" if failed else "ok"})},
        ]})},
    ]}
    assert failure_response(message) is failed


def test_export_copies_only_referenced_images_and_loads_for_training(tmp_path):
    row = agent_record()
    source = write_source(tmp_path, row)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "cleaned"
    report = clean_dataset(tmp_path / "runs", destination, ["ActiveVisual"], 5)
    output = destination / "ActiveVisual/object/trajectory.jsonl"
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 8
    assert len({r["metadata"]["session_id"] for r in records}) == len(records)
    references = set()
    for cleaned in records:
        Trajectory.model_validate(cleaned)
        for block in image_blocks(cleaned):
            reference = block["image_url"]["url"]
            assert reference.startswith("imgs/")
            image = output.parent / reference
            assert hashlib.sha256(image.read_bytes()).hexdigest() == image.stem
            references.add(image)
    assert set((output.parent / "imgs").glob("*.png")) == references
    assert report["settings"]["ActiveVisual"]["removed_failed_targets"] == 1
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    config = DataConfig(
        schema_version=3, files=[output], image_root=None, runs_root=tmp_path / "runs",
        source_model="gpt-6-astra", instances=None, validation_fraction=0, test_fraction=0,
        verify_images=True, missing_images="error", accepted_statuses=["OK", "UNKNOWN"], include_final_reply=True,
    )
    data = TrajectoryDataset(config, "ActiveVisual", 42).prepare()
    assert len(data.train) == 8 and not data.rejected


def test_used_tools_are_not_removed(tmp_path):
    row = agent_record(1, failed_round=None)
    row["messages"][4]["tool_calls"][0]["custom"]["input"] = "text(await tools.web__run({}));"
    source = write_source(tmp_path, row)
    assert "web__run" in used_tool_names([source])
    result, _ = next(clean_targets(row, 5, set(UNUSED_TOOLS) - used_tool_names([source])))
    assert result is not None
    assert "### `web__run`" in result["messages"][0]["tools"][0]["tools"][0]["description"]


def test_missing_retained_image_does_not_publish_output(tmp_path):
    source = write_source(tmp_path, agent_record(1, failed_round=None))
    (source.parent / "images/initial.png").unlink()
    with pytest.raises(FileNotFoundError):
        clean_dataset(tmp_path / "runs", tmp_path / "cleaned", ["ActiveVisual"], 5)
    assert not (tmp_path / "cleaned").exists()


def test_existing_output_is_not_overwritten(tmp_path):
    destination = tmp_path / "cleaned"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("existing output")
    with pytest.raises(FileExistsError):
        clean_dataset(tmp_path / "runs", destination, ["ActiveVisual"], 5)
    assert sentinel.read_text() == "existing output"
