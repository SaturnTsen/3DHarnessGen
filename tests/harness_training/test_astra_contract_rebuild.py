import json

import pytest

from scripts.export_astra_chat_completions import ArchiveImages
from scripts.rebuild_astra_contract import Evidence, environment_message, interactive, runtime_sources
from scripts.recover_interactive_contract import recover
from tools.validate_trajectory import Trajectory


def test_interactive_generation_boundaries_and_tool_images(tmp_path):
    instance = tmp_path / "ActiveVisual/gpt-6-astra/object"
    raw = instance / "raw"
    raw.mkdir(parents=True)
    (raw / "session_order.json").write_text(json.dumps({"attempts": [{
        "resume_mode": "fresh", "supporting_sources": [{"path": "codex.rollout.jsonl"}]
    }]}))
    meta = {"turn_id": "turn", "create_time": 1}
    records = [
        {"type": "session_meta", "payload": {"session_id": "session", "base_instructions": {"text": "system"}}},
        {"type": "response_item", "payload": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": "request"}]}},
        {"type": "response_item", "payload": {
            "type": "message", "role": "assistant", "phase": "commentary",
            "content": [{"type": "output_text", "text": "Inspecting."}],
            "internal_chat_message_metadata_passthrough": meta}},
        {"type": "response_item", "payload": {
            "type": "custom_tool_call", "call_id": "call", "name": "exec", "input": "image(result);",
            "internal_chat_message_metadata_passthrough": meta}},
        {"type": "response_item", "payload": {
            "type": "custom_tool_call_output", "call_id": "call", "output": [
                {"type": "text", "text": "before"},
                {"type": "image_url", "image_url": {"url": "screenshot", "detail": "high"}},
                {"type": "text", "text": "after"},
                {"type": "image_url", "image_url": {"url": "screenshot", "detail": "high"}}]}},
        {"type": "response_item", "payload": {"type": "reasoning", "summary": ["private"]}},
        {"type": "response_item", "payload": {
            "type": "message", "role": "assistant", "phase": "final_answer",
            "content": [{"type": "output_text", "text": "Done."}],
            "internal_chat_message_metadata_passthrough": {"turn_id": "turn", "create_time": 2}}},
    ]
    (raw / "codex.rollout.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))

    class Images:
        def save(self, value):
            assert value == {"url": "screenshot", "detail": "high"}
            return "images/screenshot.png"

    rows, mappings = interactive(instance, Images(), Evidence())
    row = rows[0]
    assert row["messages"][0]["role"] == "developer"
    assert row["tools"] is None
    assert len(row["messages"]) == 6
    assert row["messages"][2]["content"] == [{"type": "text", "text": "Inspecting."}]
    assert row["messages"][3] == {"index": 3, "role": "assistant", "content": None,
        "tool_calls": [{"id": "call", "type": "custom", "custom": {"name": "exec", "input": "image(result);"}}]}
    assert "tool_calls" not in row["messages"][2]
    assert row["messages"][4]["role"] == "tool"
    assert row["messages"][4]["tool_call_id"] == "call"
    assert [b["type"] for b in row["messages"][4]["content"]] == ["text", "image_url", "text", "image_url"]
    assert row["messages"][4]["content"][1]["image_url"]["detail"] == "high"
    assert len(mappings[0]["message_sources"][2]) == 1
    assert len(mappings[0]["message_sources"][3]) == 1
    assert [s["target_kind"] for s in row["metadata"]["steps"]] == ["status", "tool_call", "final_reply"]
    assert all(s["input_provenance"] == "reconstructed"
               and s["context_indices"] == list(range(s["target_index"]))
               for s in row["metadata"]["steps"])
    assert any("推理" in e["reason"] for e in row["metadata"]["excluded_events"])

    context = {"recovery": {"static_tool_context": True}, "tools": [{
        "type": "namespace", "name": "functions", "description": "Execution tools",
        "tools": [{"type": "custom", "name": "exec", "description": "Complete runtime context",
                   "format": {"type": "text"}}]}]}
    source = {"path": "tool_context_recovery.json", "sha256": "a" * 64, "locator": "/tools"}
    recovered = recover(row, mappings[0], context, source)
    Trajectory.model_validate(recovered)
    assert recovered["messages"][0]["tools"] == context["tools"]
    assert recovered["messages"][0]["type"] == "additional_tools"
    assert len(mappings[0]["message_sources"]) == len(recovered["messages"])
    text, call, reply = recovered["metadata"]["steps"]
    assert text["context_indices"] == [0, 1, 2]
    assert call["context_indices"] == [0, 1, 2, 3]
    assert call["target_index"] == 4
    assert reply["context_indices"] == [0, 1, 2, 3, 4, 5]
    assert reply["target_index"] == 6
    assert "tool_context_index" not in call
    assert "input_provenance" not in reply
    assert "available_tool_indices" not in call
    assert recovered["tools"][0][0]["namespace_description"] == "Execution tools"
    assert mappings[0]["tools_provenance"][0]["status"] == "recovered"
    context["recovery"]["static_tool_context"] = False
    with pytest.raises(ValueError, match="per-request snapshots"):
        recover(row, mappings[0], context, source)


def test_redacted_image_requires_unique_evidence(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (tmp_path / "post_train_trajectory/images").mkdir(parents=True)
    images = ArchiveImages(raw, tmp_path / "stage/images")
    with pytest.raises(ValueError, match="one exact match"):
        images.save("data:image/png;base64,iVBORw0KGgo[REDACTED_IMAGE]AAAA")


@pytest.mark.parametrize("change", ["none", "version", "compaction", "second_turn", "search", "missing_tool"])
def test_runtime_recovery_checks_evidence_and_boundaries(tmp_path, change):
    runtime = {"cli_version": "0.153.4", "model": "gpt-6-astra", "effort": "high",
               "approval_policy": "never", "sandbox_policy": {"type": "danger-full-access"}}
    turn = {key: value for key, value in runtime.items() if key != "cli_version"}
    context = {"recovery": {"runtime": runtime, "static_tool_context": True}, "tools": [{
        "type": "namespace", "name": "functions", "description": "Execution tools",
        "tools": [{"type": "custom", "name": "exec", "description": "Runtime",
                   "format": {"type": "text"}}]}]}
    records = [
        {"type": "session_meta", "payload": {"session_id": "session", "cli_version": "0.153.4"}},
        {"type": "turn_context", "payload": turn},
        {"type": "response_item", "payload": {"type": "custom_tool_call", "name": "exec", "input": "text(1);"}},
    ]
    if change == "version":
        records[0]["payload"]["cli_version"] = "0.154.0"
    elif change == "compaction":
        records.append({"type": "compacted", "payload": {}})
    elif change == "second_turn":
        records.append({"type": "turn_context", "payload": turn})
    elif change == "search":
        records[2]["payload"]["input"] = "await tools.tool_search({query: 'render'});"
    elif change == "missing_tool":
        records[2]["payload"]["name"] = "other"
    path = tmp_path / "session.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    if change == "none":
        session, sources = runtime_sources(path, tmp_path, Evidence(), context)
        assert session == "session"
        assert [source["locator"] for source in sources] == ["line:1", "line:2"]
    else:
        with pytest.raises(ValueError):
            runtime_sources(path, tmp_path, Evidence(), context)


def test_environment_rendering_preserves_historical_values():
    state = {"environments": {"local": {"cwd": "/historical/workspace", "shell": "bash", "status": "available"}},
             "current_date": "2026-09-09", "timezone": "Etc/UTC",
             "filesystem": '<filesystem><permission_profile type="disabled" /></filesystem>'}
    item = environment_message(state)
    assert item["role"] == "user"
    assert item["content"] == ('<environment_context>\n  <cwd>/historical/workspace</cwd>\n'
                               '  <shell>bash</shell>\n  <current_date>2026-09-09</current_date>\n'
                               '  <timezone>Etc/UTC</timezone>\n'
                               '  <filesystem><permission_profile type="disabled" /></filesystem>\n'
                               '</environment_context>')
