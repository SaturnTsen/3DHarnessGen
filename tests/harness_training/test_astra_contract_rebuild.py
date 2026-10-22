import json

import pytest

from scripts.export_astra_chat_completions import ArchiveImages
from scripts.rebuild_astra_contract import Evidence, interactive
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
    Trajectory.model_validate(row)
    assert row["tools"] is None
    assert len(row["messages"]) == 5
    assert row["messages"][2]["content"] == [{"type": "text", "text": "Inspecting."}]
    assert row["messages"][2]["tool_calls"][0]["custom"]["input"] == "image(result);"
    assert row["messages"][3]["role"] == "tool"
    assert row["messages"][3]["tool_call_id"] == "call"
    assert [b["type"] for b in row["messages"][3]["content"]] == ["text", "image_url", "text", "image_url"]
    assert row["messages"][3]["content"][1]["image_url"]["detail"] == "high"
    assert len(mappings[0]["message_sources"][2]) == 2
    assert [s["target_kind"] for s in row["metadata"]["steps"]] == ["tool_call", "final_reply"]
    assert all(s["input_provenance"] == "unknown" and s["context_indices"] == []
               for s in row["metadata"]["steps"])
    assert any("推理" in e["reason"] for e in row["metadata"]["excluded_events"])

    context = {"recovery": {"static_tool_context": True}, "tools": [{
        "type": "namespace", "name": "functions", "description": "Execution tools",
        "tools": [{"type": "custom", "name": "exec", "description": "Complete runtime context",
                   "format": {"type": "text"}}]}]}
    source = {"path": "tool_context_recovery.json", "sha256": "a" * 64, "locator": "/tools"}
    recovered = recover(row, mappings[0], context, source)
    Trajectory.model_validate(recovered)
    call, reply = recovered["metadata"]["steps"]
    assert call["context_indices"] == [0, 1]
    assert call["target_index"] == 2
    assert reply["context_indices"] == [0, 1, 2, 3]
    assert reply["target_index"] == 4
    assert call["tool_context_index"] == 0
    assert call["available_tool_indices"] == reply["available_tool_indices"] == [0]
    assert recovered["tools"][0]["namespace_description"] == "Execution tools"
    assert recovered["metadata"]["tools_provenance"]["status"] == "recovered"
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
