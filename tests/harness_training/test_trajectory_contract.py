import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from harness_training.data.contract import validate_record
from harness_training.data.contract import AdditionalTools, GrammarFormat


@pytest.fixture
def record():
    source = {"path": "source.json", "sha256": "a" * 64, "locator": "/"}
    return {
        "schema_version": 3,
        "model": "test-teacher",
        "messages": [
            {"index": 0, "role": "developer", "content": "Inspect the reference."},
            {"index": 1, "role": "developer", "type": "additional_tools", "tools": [{"type": "namespace", "name": "functions", "description": "Execution", "tools": [{"type": "custom", "name": "exec", "description": "Synthetic test tool", "format": {"type": "text"}}]}]},
            {"index": 2, "role": "user", "content": [
                {"type": "text", "text": "Inspect this."},
                {"type": "image_url", "image_url": {"url": "images/reference.png"}},
            ]},
            {"index": 3, "role": "assistant", "content": None, "tool_calls": [
                {"id": "call_render_1", "type": "custom", "custom": {
                    "name": "exec", "input": "image(result);"}},
            ]},
            {"index": 4, "role": "tool", "tool_call_id": "call_render_1", "content": [
                {"type": "image_url", "image_url": {"url": "images/result.png"}},
            ]},
            {"index": 5, "role": "assistant", "content": "The base is too wide."},
        ],
        "tools": [[{"type": "custom", "namespace": "functions", "namespace_description": "Execution", "custom": {
            "name": "exec", "description": "Synthetic test tool", "format": {"type": "text"}}}]],
        "metadata": {
            "scene": "ActiveVisual", "instance": "test-object", "session_id": "test-session",
            "view": "agent", "sources": [source],
            "steps": [
                {"id": "call", "context_indices": [0, 1, 2],
                 "target_index": 3, "target_kind": "tool_call", "status": "UNKNOWN",
                 "source": source},
                {"id": "reply", "context_indices": [0, 1, 2, 3, 4],
                 "target_index": 5, "target_kind": "final_reply", "status": "UNKNOWN",
                 "source": source},
            ],
        },
    }


def test_shared_tool_context(record):
    parsed = validate_record(json.dumps(record))
    declaration = parsed.messages[1]
    assert isinstance(declaration, AdditionalTools)
    assert declaration.tool_context()[0] == parsed.tools[0][0]
    assert parsed.tools[0][0].custom is not None
    assert parsed.tools[0][0].custom.name == "exec"


def test_delivered_example():
    parsed = validate_record(Path("docs/trajectory_example.json").read_text())
    assert parsed.schema_version == 3
    assert parsed.metadata.steps


@pytest.mark.parametrize("change", [
    lambda r: r["messages"][1].update(index=0),
    lambda r: r["messages"][1].update(index=True),
    lambda r: r["messages"][1].pop("index"),
    lambda r: r.update(schema_version=True),
    lambda r: r.update(schema_version=1),
    lambda r: r.update(unexpected=True),
    lambda r: r["metadata"]["steps"][0].update(context_indices=[0, 1, 3]),
    lambda r: r["metadata"]["steps"][0].update(context_indices=[1, 0]),
    lambda r: r["metadata"]["steps"][0].update(target_index=99),
    lambda r: r["metadata"]["steps"][0].update(target_index=2, context_indices=[0, 1]),
    lambda r: r["metadata"]["steps"][0].update(target_kind="final_reply"),
    lambda r: r["metadata"]["steps"][1].update(context_indices=[0, 1, 2, 4]),
    lambda r: r["messages"][4].update(tool_call_id="missing"),
    lambda r: r["messages"][2]["content"][1]["image_url"].update(url="images/../outside.png"),
    lambda r: r["metadata"]["sources"][0].update(sha256="placeholder"),
    lambda r: r["tools"][0].append(copy.deepcopy(r["tools"][0][0])),
    lambda r: r["tools"][0][0]["custom"].update(name="other"),
    lambda r: r["metadata"].update(view="script"),
])
def test_invalid_relationships(record, change):
    change(record)
    with pytest.raises(ValueError):
        validate_record(json.dumps(record))


@pytest.mark.parametrize("tools", [None, [], [[]], [None]])
def test_empty_or_unknown_tool_contexts_are_rejected(record, tools):
    record["tools"] = tools
    with pytest.raises(ValueError):
        validate_record(json.dumps(record))


def test_flat_tools_are_rejected(record):
    record["tools"] = record["tools"][0]
    with pytest.raises(ValueError):
        validate_record(json.dumps(record))


@pytest.mark.parametrize("field,value", [
    ("available_tools", None), ("available_tools", []),
    ("available_tool_indices", [0]), ("tools_provenance", None),
])
def test_old_selection_fields_are_rejected(record, field, value):
    record["metadata"]["steps"][0][field] = value
    with pytest.raises(ValueError, match="Extra inputs"):
        validate_record(json.dumps(record))


def test_function_context_does_not_require_exec(record):
    record["tools"] = [[{"type": "function", "function": {
        "name": "render", "description": "Render", "parameters": {"type": "object"}}}]]
    sync_tools(record)
    record["messages"][3]["tool_calls"] = [{"id": "call_render_1", "type": "function", "function": {
        "name": "render", "arguments": '{"view":"front"}'}}]
    validate_record(json.dumps(record))
    for invalid in ['[]', '{"view":NaN}', '{"view":1,"view":2}', 'not json']:
        record["messages"][3]["tool_calls"][0]["function"]["arguments"] = invalid
        with pytest.raises(ValueError):
            validate_record(json.dumps(record))


def test_current_target_must_match_tool_type(record):
    record["tools"][0] = [{"type": "function", "function": {
        "name": "exec", "description": "Different type", "parameters": {"type": "object"}}}]
    sync_tools(record)
    with pytest.raises(ValueError, match="wrong type"):
        validate_record(json.dumps(record))


def test_different_contexts_and_historical_calls(record):
    later = copy.deepcopy(record["tools"][0])
    later[0]["custom"]["name"] = "later_tool"
    record["tools"].append(later)
    add_later_input(record)
    validate_record(json.dumps(record))
    record["metadata"]["steps"][0]["context_indices"] = [0, 2, 5]
    with pytest.raises(ValueError):
        validate_record(json.dumps(record))


def test_unreferenced_empty_context_is_rejected(record):
    record["tools"].append([])
    with pytest.raises(ValueError, match="must not be empty"):
        validate_record(json.dumps(record))


def test_tool_indices_are_local_to_context(record):
    record["tools"][0][0]["index"] = 0
    validate_record(json.dumps(record))
    record["tools"][0][0]["index"] = 1
    with pytest.raises(ValueError, match="context array position"):
        validate_record(json.dumps(record))


@pytest.mark.parametrize("scene", ["Single-view", "Multi-view"])
def test_script_view_omits_codex_context(record, scene):
    record["metadata"].update(scene=scene, view="script")
    record["tools"] = []
    record["messages"] = [record["messages"][i] for i in (0, 2, 5)]
    for i, message in enumerate(record["messages"]):
        message["index"] = i
    target = record["metadata"]["steps"][1]
    target.update(target_kind="script", target_index=2, context_indices=[0, 1])
    record["metadata"]["steps"] = [target]
    validate_record(json.dumps(record))
    record["tools"] = [[{"type": "custom", "custom": {
        "name": "exec", "description": "Codex runtime", "format": {"type": "text"}}}]]
    with pytest.raises(ValueError, match="must omit Codex"):
        validate_record(json.dumps(record))


@pytest.mark.parametrize("raw", ['model: example', '{"schema_version":2,"schema_version":2}', 'NaN'])
def test_strict_json(raw):
    with pytest.raises(ValueError):
        validate_record(raw)


def test_cli_json_and_jsonl(record, tmp_path):
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(record, indent=2))
    command = [sys.executable, "tools/validate_trajectory.py"]
    result = subprocess.run([*command, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 条通过" in result.stdout
    path = tmp_path / "trajectory.jsonl"
    path.write_text(json.dumps(record) + '\n{"messages":[]}\n' + json.dumps(record) + '\n')
    result = subprocess.run([*command, str(path)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "1 条通过，2 条失败" in result.stdout
    assert "记录/行 2" in result.stdout
    assert "重复的 session_id" in result.stdout


def sync_tools(record):
    for tool in record["tools"][0]:
        tool.update(namespace="functions", namespace_description="Execution")
    record["messages"][1]["tools"][0]["tools"] = [{"type": t["type"], **copy.deepcopy(t[t["type"]])} for t in record["tools"][0]]


def add_later_input(record):
    native = [{"type": t["type"], **copy.deepcopy(t[t["type"]])} for t in record["tools"][1]]
    record["messages"].insert(5, {"index": 5, "role": "developer", "type": "additional_tools", "tools": [{"type": "namespace", "name": "functions", "description": "Execution", "tools": native}]})
    record["messages"][6]["index"] = 6
    record["metadata"]["steps"][1].update(target_index=6, context_indices=[0, 2, 3, 4, 5])


@pytest.mark.parametrize("change", [
    lambda r: r["messages"][1].update(role="system"),
    lambda r: r["messages"][0].update(role="system"),
    lambda r: r["messages"][1].update(content="tool definitions"),
    lambda r: r["messages"][1]["tools"][0].update(description="wrong description"),
    lambda r: r["metadata"]["steps"][0].update(context_indices=[0, 2]),
    lambda r: r["messages"][1].update(tools=[]),
])
def test_additional_tools_are_required_and_exact(record, change):
    change(record)
    with pytest.raises(ValueError):
        validate_record(json.dumps(record))


def test_namespace_and_grammar_are_preserved(record):
    tool = record["tools"][0][0]
    tool.update(namespace="functions", namespace_description="Execution tools")
    tool["custom"]["format"] = {"type": "grammar", "syntax": "lark", "definition": 'start: "text(1);"'}
    record["messages"][1]["tools"] = [{"type": "namespace", "name": "functions",
        "description": "Execution tools", "tools": [{"type": "custom", **copy.deepcopy(tool["custom"])}]}]
    parsed = validate_record(json.dumps(record))
    declaration = parsed.messages[1]
    assert isinstance(declaration, AdditionalTools)
    tool_format = declaration.tools[0].tools[0].format
    assert isinstance(tool_format, GrammarFormat)
    assert tool_format.definition == 'start: "text(1);"'
    record["messages"][1]["tools"][0]["description"] = "Different namespace description"
    with pytest.raises(ValueError, match="does not match any tool context"):
        validate_record(json.dumps(record))


def test_discovered_descriptions_remain_tool_results(record):
    original_tools = copy.deepcopy(record["tools"])
    original_additional_tools = copy.deepcopy(record["messages"][1])
    image = {"type": "image_url", "image_url": {"url": "images/result.png"}}
    content = [
        {"type": "text", "text": '[{"name":"blender_render","description":"Render a scene"}]'},
        image,
        {"type": "text", "text": "The second image follows."},
        copy.deepcopy(image),
    ]
    record["messages"][4]["content"] = content
    parsed = validate_record(json.dumps(record))
    restored = parsed.model_dump(exclude_unset=True)
    assert restored == record
    assert restored["tools"] == original_tools
    assert restored["messages"][1] == original_additional_tools
    assert restored["messages"][4]["role"] == "tool"
    assert restored["messages"][4]["tool_call_id"] == "call_render_1"
    assert restored["messages"][4]["content"] == content


@pytest.mark.parametrize("change", [
    lambda r: r["messages"][3]["tool_calls"][0].update(id="call_render_1_duplicate"),
    lambda r: r["messages"][4].pop("tool_call_id"),
    lambda r: r["messages"][4].update(role="user"),
    lambda r: r["messages"][4].update(type="custom_tool_call_output", call_id="call_render_1"),
    lambda r: r["messages"][2]["content"][0].update(type="input_text"),
])
def test_trajectory_fields_and_tool_result_links_are_required(record, change):
    change(record)
    with pytest.raises(ValueError):
        validate_record(json.dumps(record))


@pytest.mark.parametrize("definition", [
    {"type": "function", "name": "exec", "description": "Missing parameters"},
    {"type": "custom", "name": "exec", "description": "Missing format"},
    {"type": "custom", "name": "exec", "description": "Mixed fields",
     "format": {"type": "text"}, "parameters": {"type": "object"}},
    {"type": "namespace", "name": "functions", "description": "Empty group", "tools": []},
    {"type": "namespace", "name": "functions", "description": "Missing member type", "tools": [
        {"name": "exec", "description": "Run code", "format": {"type": "text"}}]},
    {"name": "exec", "description": "Missing type", "format": {"type": "text"}},
])
def test_unified_tool_definitions_reject_invalid_fields(record, definition):
    record["messages"][1]["tools"] = [definition]
    with pytest.raises(ValueError):
        validate_record(json.dumps(record))
