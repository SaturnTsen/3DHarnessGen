#!/usr/bin/env python3
"""Export one complete multimodal SFT trajectory per provider-run instance."""
from __future__ import annotations

import argparse
import json
import tempfile
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

if __package__:
    from .sft_common import DATA_URI_RE, ImageStore, make_row, read_prompt, write_trajectory
else:
    from sft_common import DATA_URI_RE, ImageStore, make_row, read_prompt, write_trajectory

RUNS_ROOT = Path(__file__).resolve().parent.parent / "3DHarnessBench" / "runs"
INTERNAL_CHANNELS = {"analysis", "reasoning", "thinking", "internal"}


def json_records(path: Path) -> Iterable[Any]:
    """Read JSON, JSONL, or a JSON document with a trailing non-JSON error."""
    if not path.is_file():
        return
    if path.suffix.lower() == ".json" or path.name.startswith("mmx.request-"):
        try:
            value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(value, list):
                yield from value
            else:
                yield value
            return
        except (json.JSONDecodeError, OSError):
            pass
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Kimi session exports can end with a plain-text provider error.
                    continue
    except OSError:
        return


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))



def _channel_is_internal(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in INTERNAL_CHANNELS
    return False


def _image_value(part: dict[str, Any]) -> tuple[Any, str | None]:
    mime = part.get("mimeType") or part.get("mime_type") or part.get("media_type")
    for key in ("image_url", "imageUrl", "image", "source", "url", "data", "base64", "path", "blob"):
        if key in part:
            value = part[key]
            if key == "source" and isinstance(value, dict):
                mime = value.get("media_type") or value.get("mimeType") or mime
                value = value.get("data") or value.get("url") or value
            return value, mime
    return None, mime


def _content_blocks(content: Any, images: ImageStore) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            cursor = 0
            for match in DATA_URI_RE.finditer(value):
                if match.start() > cursor:
                    blocks.append({"type": "text", "text": value[cursor:match.start()]})
                path = images.save(match.group(0), match.group(1))
                if path:
                    blocks.append({"type": "image", "image": path})
                else:
                    blocks.append({"type": "text", "text": "[Image unavailable: source payload is missing or corrupt.]"})
                cursor = match.end()
            if cursor < len(value):
                blocks.append({"type": "text", "text": value[cursor:]})
            return
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            blocks.append({"type": "text", "text": _as_text(value)})
            return
        kind = str(value.get("type", "")).lower()
        if kind in {"thinking", "reasoning", "redacted_thinking", "analysis"}:
            return
        if kind in {"image", "input_image", "output_image", "image_url", "computer_image"} or "image_url" in value:
            image_value, mime = _image_value(value)
            path = images.save(image_value, mime)
            if path:
                blocks.append({"type": "image", "image": path})
            else:
                blocks.append({"type": "text", "text": "[Image unavailable: source payload is missing or corrupt.]"})
            return
        if kind in {"text", "input_text", "output_text"}:
            visit(value.get("text", value.get("content", "")))
            return
        if kind in {"tool_use", "tool_call", "function_call", "tool_result", "function_call_output"}:
            return
        # Some capture formats wrap content and images in a generic object.
        handled = False
        for key in ("content", "parts", "output", "result"):
            if key in value:
                visit(value[key])
                handled = True
        for key in ("images", "local_images", "generated_images"):
            for image in value.get(key, []) if isinstance(value.get(key), list) else []:
                visit({"type": "image", "image": image})
                handled = True
        if not handled:
            image_value, mime = _image_value(value)
            if image_value is not None and images._bytes_from_value(image_value, mime):
                path = images.save(image_value, mime)
                if path:
                    blocks.append({"type": "image", "image": path})
            elif "text" in value:
                visit(value["text"])
            else:
                blocks.append({"type": "text", "text": _as_text(value)})

    visit(content)
    return [block for block in blocks if block.get("type") != "text" or block.get("text")]


def _normalize_tool_call(call: Any, fallback_id: str) -> dict[str, Any] | None:
    if not isinstance(call, dict):
        return None
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    name = function.get("name") or call.get("name") or call.get("tool")
    if not name:
        return None
    arguments = function.get("arguments", function.get("input", call.get("arguments", call.get("input", {}))))
    # Qwen3.8's template iterates arguments.items(); API-style JSON strings
    # cannot be passed through unchanged. Keep freeform calls as an input arg.
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"input": arguments} if arguments else {}
    if not isinstance(arguments, dict):
        arguments = {"input": arguments}
    return {
        "id": str(call.get("id") or call.get("call_id") or fallback_id),
        "type": "function",
        "function": {"name": str(name), "arguments": arguments},
    }


def _append_message(messages: list[dict[str, Any]], role: str, content: Any,
                    images: ImageStore, tool_calls: list[Any] | None = None,
                    call_id: str | None = None, name: str | None = None,
                    event_id: str | None = None) -> None:
    role = {"agent": "assistant", "model": "assistant", "human": "user"}.get(role, role)
    if role not in {"system", "developer", "user", "assistant", "tool"}:
        return
    if _channel_is_internal(event_id):
        return
    blocks: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    tool_results: list[tuple[str | None, str | None, Any]] = []
    for part in content if isinstance(content, list) else [content]:
        if isinstance(part, dict):
            kind = str(part.get("type", "")).lower()
            if kind in {"thinking", "reasoning", "redacted_thinking", "analysis"}:
                continue
            if kind in {"tool_use", "tool_call", "function_call"}:
                call = _normalize_tool_call(part, f"call_{len(messages) + len(calls)}")
                if call:
                    calls.append(call)
                continue
            if kind in {"tool_result", "function_call_output"}:
                tool_results.append((part.get("tool_use_id") or part.get("tool_call_id") or part.get("call_id"),
                                     part.get("name"), part.get("content", part.get("output", ""))))
                continue
            if kind == "thinking_signature":
                continue
        blocks.extend(_content_blocks(part, images))
    if tool_calls:
        for index, call in enumerate(tool_calls):
            normalized = _normalize_tool_call(call, f"call_{len(messages) + index}")
            if normalized:
                calls.append(normalized)
    if blocks or calls or role == "tool":
        message: dict[str, Any] = {"role": role, "content": blocks}
        if calls:
            message["tool_calls"] = calls
        if role == "tool":
            if call_id:
                message["tool_call_id"] = str(call_id)
            if name:
                message["name"] = name
        messages.append(message)
    for result_id, result_name, result_content in tool_results:
        result_blocks = _content_blocks(result_content, images)
        messages.append({
            "role": "tool",
            "tool_call_id": str(result_id or ""),
            **({"name": str(result_name)} if result_name else {}),
            "content": result_blocks,
        })


def _parse_standard_message(record: dict[str, Any], messages: list[dict[str, Any]], images: ImageStore) -> bool:
    role = record.get("role")
    if not role:
        return False
    channel = record.get("channel") or record.get("phase") or record.get("subtype")
    if _channel_is_internal(channel):
        return True
    _append_message(
        messages,
        str(role),
        record.get("content", record.get("text", "")),
        images,
        tool_calls=record.get("tool_calls") or record.get("toolCalls"),
        call_id=record.get("tool_call_id"),
        name=record.get("name"),
        event_id=str(channel or ""),
    )
    return True


def parse_codex_record(record: Any, messages: list[dict[str, Any]], images: ImageStore) -> None:
    if not isinstance(record, dict):
        return
    if record.get("type") == "response_item":
        payload = record.get("payload", {})
        kind = payload.get("type")
        if kind == "message":
            role = payload.get("role", "assistant")
            _append_message(messages, role, payload.get("content", []), images,
                            event_id=str(payload.get("channel") or ""))
        elif kind in {"function_call", "custom_tool_call"}:
            call = {"id": payload.get("call_id") or payload.get("id"),
                    "name": payload.get("name"),
                    "arguments": payload.get("arguments", payload.get("input", {}))}
            _append_message(messages, "assistant", "", images, [call])
        elif kind in {"function_call_output", "custom_tool_call_output"}:
            _append_message(messages, "tool", payload.get("output", ""), images,
                            call_id=payload.get("call_id"), name=payload.get("name"))
        return
    if record.get("type") == "item.completed":
        item = record.get("item", {})
        kind = item.get("type")
        if kind == "agent_message":
            _append_message(messages, "assistant", item.get("text", ""), images,
                            event_id=str(item.get("phase") or ""))
        elif kind in {"mcp_tool_call", "function_call", "custom_tool_call"}:
            _append_message(messages, "assistant", "", images, [item])
        elif kind in {"mcp_tool_result", "function_call_output"}:
            _append_message(messages, "tool", item.get("result", item.get("output", "")), images,
                            call_id=item.get("call_id"), name=item.get("name"))
        elif kind == "command_execution":
            call_id = item.get("id") or f"command_{len(messages)}"
            _append_message(messages, "assistant", "", images, [{"id": call_id, "name": "command_execution", "arguments": {"command": item.get("command", "")}}])
            output = item.get("aggregated_output", item.get("output", ""))
            _append_message(messages, "tool", output, images, call_id=call_id, name="command_execution")
        return
    if _parse_standard_message(record, messages, images):
        return


def parse_claude_record(record: Any, messages: list[dict[str, Any]], images: ImageStore) -> None:
    if not isinstance(record, dict):
        return
    kind = record.get("type")
    if kind in {"assistant", "user"}:
        message = record.get("message", {})
        role = message.get("role", kind) if isinstance(message, dict) else kind
        _append_message(messages, role, message.get("content", record.get("content", "")) if isinstance(message, dict) else record.get("content", ""), images,
                        tool_calls=record.get("tool_calls"), call_id=record.get("tool_call_id"),
                        event_id=str(message.get("channel", "")) if isinstance(message, dict) else "")
        return
    if kind == "context.append_message":
        message = record.get("message", {})
        if isinstance(message, dict):
            _append_message(messages, str(message.get("role", "assistant")), message.get("content", ""), images,
                            tool_calls=message.get("toolCalls") or message.get("tool_calls"),
                            call_id=message.get("tool_call_id"), event_id=str(message.get("channel", "")))
        return
    if kind in {"response", "tool_result"}:
        body = record.get("response", record.get("result", {}))
        if isinstance(body, dict):
            role = body.get("role", "assistant" if kind == "response" else "tool")
            _append_message(messages, role, body.get("content", ""), images,
                            tool_calls=body.get("tool_calls"), call_id=body.get("tool_use_id"), name=body.get("name"))
        return
    _parse_standard_message(record, messages, images)


def parse_kimi_event(record: dict[str, Any], messages: list[dict[str, Any]], images: ImageStore) -> None:
    """Kimi wire archives put assistant turns and tool results in loop events."""
    event = record.get("event") or {}
    kind = event.get("type")
    if kind == "content.part":
        part = event.get("part") or {}
        if part.get("type") == "text":
            _append_message(messages, "assistant", [part], images)
    elif kind == "tool.call":
        call = {"id": event.get("toolCallId"), "name": event.get("name"),
                "arguments": event.get("args", {})}
        _append_message(messages, "assistant", "", images, [call])
    elif kind == "tool.result":
        _append_message(messages, "tool", event.get("result", {}), images,
                        call_id=event.get("toolCallId"))


def parse_gemini_transcript(path: Path, raw_root: Path, messages: list[dict[str, Any]], images: ImageStore) -> None:
    records = list(json_records(path))
    call_path = next((p for p in (path.parent / "tool_calls.json", raw_root / "session" / "tool_calls.json") if p.is_file()), None)
    calls = list(json_records(call_path)) if call_path else []
    call_index = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        role = record.get("role")
        event_type = str(record.get("type", "")).upper()
        content = record.get("content", "")
        if role == "user":
            _append_message(messages, "user", content, images)
        elif role in {"agent", "assistant"}:
            if event_type in {"EPHEMERAL_MESSAGE", "THINKING", "CHECKPOINT", "CONVERSATION_HISTORY"}:
                continue
            matched = None
            if call_index < len(calls):
                candidate = calls[call_index]
                if event_type == "MCP_TOOL" or str(candidate.get("tool", "")).upper() == event_type:
                    matched = candidate
                    call_index += 1
            if matched:
                call = {"id": f"gemini_{record.get('step_index', call_index)}",
                        "name": matched.get("tool"),
                        "arguments": matched.get("arguments", "{}")}
                _append_message(messages, "assistant", "", images, [call])
                _append_message(messages, "tool", content, images,
                                call_id=call["id"], name=str(matched.get("tool", "tool")))
            elif content:
                _append_message(messages, "assistant", content, images)


def parse_source(path: Path, messages: list[dict[str, Any]], images: ImageStore, warnings: list[str]) -> None:
    if not path.is_file():
        warnings.append(f"missing source: {path}")
        return
    for record in json_records(path):
        before = len(messages)
        if isinstance(record, dict) and record.get("type") == "context.append_loop_event":
            parse_kimi_event(record, messages, images)
        elif "codex.rollout" in path.name or "codex.session" in path.name:
            parse_codex_record(record, messages, images)
        elif "claude." in path.name:
            parse_claude_record(record, messages, images)
        else:
            if isinstance(record, dict) and record.get("type") in {"assistant", "user"} and isinstance(record.get("message"), dict):
                parse_claude_record(record, messages, images)
            elif isinstance(record, dict) and record.get("type") in {"response", "tool_result", "context.append_message"}:
                parse_claude_record(record, messages, images)
            elif isinstance(record, dict) and not _parse_standard_message(record, messages, images):
                if isinstance(record, dict) and "content" in record:
                    _append_message(messages, "assistant", record.get("content"), images)
        if len(messages) == before and isinstance(record, dict):
            # Non-conversation metadata and internal events are intentionally skipped.
            continue


def _all_source_paths(raw_root: Path, attempts: list[dict[str, Any]]) -> list[Path]:
    paths: list[Path] = []
    for attempt in attempts:
        for segment in attempt.get("segments", []):
            value = segment.get("path")
            if value:
                paths.append(raw_root / value)
        for source in attempt.get("supporting_sources", []):
            value = source.get("path")
            if value:
                candidate = raw_root / value
                if candidate.is_file():
                    paths.append(candidate)
    # Shared roots are intentionally scanned only for provider-native transcripts.
    for base in (raw_root / "session", raw_root / "attempts"):
        if base.exists():
            for name in ("messages.json", "wire.jsonl", "agy.transcript.json"):
                paths.extend(base.rglob(name))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _choose_native_transcripts(raw_root: Path, attempts: list[dict[str, Any]], model: str) -> list[Path]:
    all_paths = _all_source_paths(raw_root, attempts)
    explicit_paths = []
    for attempt in attempts:
        for item in attempt.get("segments", []) + attempt.get("supporting_sources", []):
            value = item.get("path")
            if value:
                candidate = raw_root / value
                if candidate.is_file():
                    explicit_paths.append(candidate)
    explicit_paths = list(dict.fromkeys(explicit_paths))
    if model == "gemini-3-1-pro":
        candidates = [p for p in explicit_paths if p.name == "agy.transcript.json"]
        if candidates:
            return candidates
        candidates = [p for p in all_paths if p.name == "agy.transcript.json"]
        session_ids = {str(sid) for a in attempts for sid in a.get("session_ids", [])}
        matched = [p for p in candidates if any(sid in str(p) for sid in session_ids)]
        return matched or (candidates[:1] if len(candidates) == 1 else [])
    if model == "minimax-m3":
        candidates = [p for p in all_paths if p.name == "messages.json"]
        session_ids = {str(sid) for a in attempts for sid in a.get("session_ids", [])}
        matched = [p for p in candidates if any(sid in str(p) for sid in session_ids)]
        return matched or (candidates[:1] if len(candidates) == 1 else [])
    if model == "kimi-k3":
        segment_paths = [raw_root / s.get("path", "") for a in attempts for s in a.get("segments", [])]
        useful = []
        for p in segment_paths:
            if p.name == "kimi.session.jsonl":
                records = list(json_records(p))
                if any(isinstance(x, dict) and x.get("role") in {"user", "assistant", "tool"} for x in records):
                    useful.append(p)
        candidates = [p for p in all_paths if p.name == "wire.jsonl" and "kimi_home" in str(p)]
        session_ids = {str(sid) for a in attempts for sid in a.get("session_ids", [])}
        matched = [p for p in candidates if any(sid in str(p) for sid in session_ids)]
        if matched or len(candidates) == 1:
            return matched or candidates
        return useful
    if model in {"gpt-5-6-sol", "gpt-6-astra"}:
        return [p for p in explicit_paths if "codex.rollout" in p.name]
    return []


def _parse_gemini(path: Path, raw_root: Path, messages: list[dict[str, Any]], images: ImageStore) -> None:
    parse_gemini_transcript(path, raw_root, messages, images)


def _extend_snapshot(messages: list[dict[str, Any]], snapshot: list[dict[str, Any]]) -> None:
    """Archives can contain the full resumed session again; append only its tail."""
    if not snapshot:
        return
    old = [json.dumps(m, sort_keys=True, ensure_ascii=False) for m in messages]
    new = [json.dumps(m, sort_keys=True, ensure_ascii=False) for m in snapshot]
    if old[:len(new)] == new:
        return
    # Only remove an exact overlap at archive boundaries. Repeated actions
    # inside one transcript remain part of the trajectory.
    for size in range(min(len(old), len(new)), 0, -1):
        if old[-size:] == new[:size]:
            messages.extend(snapshot[size:])
            return
    messages.extend(snapshot)


def _load_chain_messages(raw_root: Path, attempts: list[dict[str, Any]], model: str,
                         images: ImageStore) -> tuple[list[dict[str, Any]], list[str]]:
    messages: list[dict[str, Any]] = []
    warnings: list[str] = []
    image_warning_start = len(images.warnings)
    native_paths = _choose_native_transcripts(raw_root, attempts, model)
    if native_paths:
        for path in native_paths:
            snapshot: list[dict[str, Any]] = []
            if model == "gemini-3-1-pro":
                _parse_gemini(path, raw_root, snapshot, images)
            else:
                parse_source(path, snapshot, images, warnings)
            _extend_snapshot(messages, snapshot)
    elif model not in {"gemini-3-1-pro", "minimax-m3", "gpt-5-6-sol", "gpt-6-astra"}:
        for attempt in attempts:
            for segment in attempt.get("segments", []):
                path = raw_root / segment.get("path", "")
                parse_source(path, messages, images, warnings)
    else:
        # Fallback to all ordered segments if a richer transcript was not retained.
        for attempt in attempts:
            for segment in attempt.get("segments", []):
                parse_source(raw_root / segment.get("path", ""), messages, images, warnings)
    # For MiniMax, the standard message archive is authoritative; for unmatched chains,
    # request outputs still preserve visible assistant turns.
    if model == "minimax-m3" and not messages:
        for attempt in attempts:
            for segment in attempt.get("segments", []):
                parse_source(raw_root / segment.get("path", ""), messages, images, warnings)
    first_turn = next((m for m in messages if m["role"] not in {"system", "developer"}), None)
    if first_turn is None or first_turn["role"] != "user":
        for attempt in sorted(attempts, key=lambda a: (a.get("reading_position", 0), a.get("attempt", 0))):
            for source in attempt.get("supporting_sources", []):
                if Path(source.get("path", "")).name != "agent_input.txt":
                    continue
                prompt_path = raw_root / source["path"]
                try:
                    prompt = prompt_path.read_text(encoding="utf-8", errors="replace").strip()
                except OSError:
                    continue
                if prompt:
                    messages.insert(0, {"role": "user", "content": [{"type": "text", "text": prompt}]})
                    break
            if messages and messages[0]["role"] == "user":
                break
        if not messages or messages[0]["role"] != "user":
            fallback = raw_root / "session" / "agent_input.txt"
            if fallback.is_file():
                prompt = fallback.read_text(encoding="utf-8").strip()
                if prompt:
                    messages.insert(0, {"role": "user", "content": [{"type": "text", "text": prompt}]})
    warnings.extend(images.warnings[image_warning_start:])
    return messages, warnings


def export_instance(instance_dir: Path, model_dir: Path, setting: str, model: str,
                    dry_run: bool = False, prompt_file: Path | None = None) -> tuple[int, int, list[str]]:
    raw_root = instance_dir / "raw"
    order_path = raw_root / "session_order.json"
    order = json.loads(order_path.read_text(encoding="utf-8"))
    if order.get("kind") == "iterative":
        raise ValueError("iterative run: use build_trajectory.py")
    prompt, prompt_source = read_prompt(raw_root, prompt_file)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in order.get("attempts", []):
        chain = str(attempt.get("chain_group") or f"chain-{attempt.get('reading_position', attempt.get('attempt', 1)):03d}")
        grouped[chain].append(attempt)
    chains = sorted(grouped.items(), key=lambda pair: min(a.get("reading_position", a.get("attempt", 0)) for a in pair[1]))
    output_dir = instance_dir / "post_train_trajectory"
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".trajectory-images-",
                                     dir=None if dry_run else output_dir) as tmp:
        stage = Path(tmp)
        images = ImageStore(raw_root, None if dry_run else stage)
        messages: list[dict[str, Any]] = []
        chain_metadata = []
        all_warnings: list[str] = []
        for chain_name, attempts in chains:
            attempts = sorted(attempts, key=lambda a: (a.get("reading_position", 0), a.get("attempt", 0)))
            chain_messages, warnings = _load_chain_messages(raw_root, attempts, model, images)
            all_warnings.extend(f"{chain_name}: {warning}" for warning in warnings)
            # API failures with no model response have no conversation to train on.
            # Retain their provenance without inventing assistant messages.
            has_target = any(m["role"] == "assistant" for m in chain_messages)
            if not has_target:
                all_warnings.append(f"{chain_name}: no assistant training target parsed")
                chain_messages = []
            # Qwen allows system/developer context only in the first message;
            # offsets count conversation messages after that shared system prompt.
            start = 1 + sum(m["role"] not in {"system", "developer"} for m in messages)
            end = start + sum(m["role"] not in {"system", "developer"} for m in chain_messages)
            messages.extend(chain_messages)
            source_logs = [record["path"] for a in attempts
                           for key in ("segments", "supporting_sources")
                           for record in a.get(key, []) if record.get("path")]
            chain_metadata.append({
                "chain_group": chain_name, "message_start": start, "message_end": end,
                "session_ids": sorted({sid for a in attempts for sid in a.get("session_ids", [])}),
                "attempts": [a.get("attempt") for a in attempts],
                "source_logs": list(dict.fromkeys(source_logs)),
            })
        if not messages and model == "kimi-k3":
            # Some imported indexes only list failed invocations, while the
            # instance's native archive still contains its successful session.
            # Recover those source conversations and make the provenance explicit.
            candidates = [p for p in raw_root.rglob("wire.jsonl") if p.parent.name == "main"]
            def created_at(path: Path) -> int:
                first = next(iter(json_records(path)), {})
                return int(first.get("created_at", 0)) if isinstance(first, dict) else 0
            recovered_sources = []
            for path in sorted(candidates, key=lambda p: (created_at(p), str(p))):
                snapshot: list[dict[str, Any]] = []
                parse_source(path, snapshot, images, all_warnings)
                if any(m["role"] == "assistant" for m in snapshot):
                    _extend_snapshot(messages, snapshot)
                    recovered_sources.append(str(path.relative_to(raw_root)))
            if recovered_sources:
                all_warnings.append("session_order has no model responses; recovered native sessions in created_at order")
                all_warnings.extend(images.warnings)
                chain_metadata.append({
                    "chain_group": "recovered-native", "message_start": 1,
                    "message_end": 1 + sum(m["role"] not in {"system", "developer"} for m in messages),
                    "session_ids": [], "attempts": [], "source_logs": recovered_sources,
                })
        images_complete = not images.warnings
        if not images.digest_paths and any(m.get("tool_calls") for m in messages):
            images_complete = False
            all_warnings.append("source logs contain tool calls but no recoverable image payloads")
        if any("provided the following image data" in b.get("text", "")
               for m in messages for b in m.get("content", [])):
            images_complete = False
            all_warnings.append("source log replaced image payloads with text placeholders; those images are unavailable")
        row = make_row(messages, prompt, {
            "benchmark": setting, "model": model, "instance": instance_dir.name,
            "source_kind": "interactive", "prompt_source": prompt_source,
            "images_complete": images_complete,
            "chains": chain_metadata, "warnings": list(dict.fromkeys(all_warnings)),
        })
        if not dry_run:
            write_trajectory(output_dir, row, stage)
        return 1, len(set(row["images"])), all_warnings


def resolve_data_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = [(RUNS_ROOT / path).resolve(), (Path.cwd() / path).resolve()]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a model/setting directory of runs to multimodal SFT JSONL.")
    parser.add_argument("data_path", help="Setting/model directory, e.g. ActiveVisual/gpt-6-astra")
    parser.add_argument("--dry-run", action="store_true", help="Summarize outputs without writing files")
    parser.add_argument("--prompt-file", type=Path, help="Override the supplied task prompt for every instance")
    args = parser.parse_args()
    data_path = resolve_data_path(args.data_path)
    if not data_path.is_dir():
        parser.error(f"data_path is not a directory: {data_path}")
    if (data_path / "raw" / "session_order.json").is_file():
        instance_dirs = [data_path]
    else:
        instance_dirs = sorted(p.parent.parent for p in data_path.rglob("raw/session_order.json")
                               if not (p.parent / "process.json").is_file())
    if not instance_dirs:
        parser.error(f"no instance directories containing raw/session_order.json under: {data_path}")
    total_trajectories = total_images = 0
    failed = 0
    for instance_dir in instance_dirs:
        try:
            model_dir = instance_dir.parent
            setting, model = model_dir.parent.name, model_dir.name
            trajectories, images, warnings = export_instance(
                instance_dir, model_dir, setting, model, args.dry_run, args.prompt_file)
            total_trajectories += trajectories
            total_images += images
            label = "would write" if args.dry_run else "wrote"
            print(f"{label} {instance_dir}: {trajectories} trajectory, {images} images")
            for warning in dict.fromkeys(warnings):
                print(f"  warning: {warning}", file=sys.stderr)
        except Exception as exc:  # Continue to export other instances in the directory.
            failed += 1
            print(f"error: {instance_dir}: {exc}", file=sys.stderr)
    print(f"processed {len(instance_dirs)} instances: {total_trajectories} trajectories, {total_images} images, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
