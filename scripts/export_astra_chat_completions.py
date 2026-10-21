"""Rebuild one Astra instance in all four settings from archived session logs.

Run from the project root with Pixi:
    pixi run python -m scripts.export_astra_chat_completions AgaveMonocotFactory

Each JSONL row is one independent session in Chat Completions request format.
Local image URLs are relative to the JSONL; provenance is saved under temp/.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from scripts.build_trajectory import image_refs, prompts, resolve_recorded, retry_prompt
from scripts.sft_common import DATA_URI_RE, ImageStore


class ArchiveImages(ImageStore):
    """Recover redacted payloads only from a uniquely matching existing image."""

    def __init__(self, raw, output_stage):
        super().__init__(raw, output_stage)
        self.recoveries = {}

    def save(self, value, mime=None):
        if isinstance(value, str) and value.startswith("data:image/") and "[REDACTED_" in value:
            digest = hashlib.sha256(value.encode()).hexdigest()
            if digest not in self.recoveries:
                fragments = re.split(r"\[REDACTED_[A-Z0-9_]+\]", value.split(",", 1)[1])
                pattern = re.compile(".*".join(re.escape(fragment) for fragment in fragments))
                candidates = self.raw_root.parent / "post_train_trajectory" / "images"
                matches = [path for path in candidates.iterdir()
                           if path.is_file() and pattern.fullmatch(base64.b64encode(path.read_bytes()).decode())]
                if len(matches) != 1:
                    raise ValueError(f"Redacted image needs one exact match of all retained base64 fragments; got {len(matches)}")
                path = matches[0]
                if hashlib.sha256(path.read_bytes()).hexdigest() != path.stem:
                    raise ValueError("Recovered image filename does not match its SHA-256")
                relative = super().save(path.read_bytes(), mime)
                if relative is None:
                    raise ValueError("Recovered image failed integrity checks")
                self.recoveries[digest] = {"source_payload_sha256": digest, "image": relative,
                                          "source": "previous post_train_trajectory/images",
                                          "validation": "unique full match of every unredacted base64 fragment; SHA-256 and PNG CRC checked",
                                          "occurrences": 0}
            self.recoveries[digest]["occurrences"] += 1
            return self.recoveries[digest]["image"]
        return super().save(value, mime)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def content_parts(value, images):
    """Preserve text and image occurrence order, including images inside exec output."""
    if isinstance(value, str):
        parts = []
        cursor = 0
        for match in DATA_URI_RE.finditer(value):
            if match.start() > cursor:
                parts.append({"type": "text", "text": value[cursor:match.start()]})
            parts.extend(content_parts({"type": "input_image", "image_url": match.group(0)}, images))
            cursor = match.end()
        if cursor < len(value):
            parts.append({"type": "text", "text": value[cursor:]})
        return parts
    if isinstance(value, list):
        return [part for item in value for part in content_parts(item, images)]
    kind = value["type"]
    if kind in {"input_text", "output_text", "text"}:
        return content_parts(value["text"], images)
    if kind in {"input_image", "image_url"}:
        url = value["image_url"]
        path = images.save(url)
        if path is None:
            raise ValueError(f"Cannot recover image: {images.warnings[-1]}")
        image_url = {"url": path}
        detail = url.get("detail") if isinstance(url, dict) else value.get("detail")
        if detail is not None:
            image_url["detail"] = detail
        return [{"type": "image_url", "image_url": image_url}]
    raise ValueError(f"Unsupported content type: {kind}")


def append_tool_result(messages, call_id, parts):
    """Chat Completions tool content only permits text, so move the image suffix."""
    split = next((i for i, part in enumerate(parts) if part["type"] == "image_url"), len(parts))
    messages.append({"role": "tool", "tool_call_id": call_id, "content": parts[:split] or ""})
    if split < len(parts):
        messages.append({"role": "user", "name": "tool_output", "content": parts[split:]})


def interactive_sessions(raw, images):
    order = read_json(raw / "session_order.json")
    attempts = order["attempts"]
    if len(attempts) != 1:
        raise ValueError("This preview exporter requires one interactive attempt; resumed chains need explicit handling")
    paths = [s["path"] for s in attempts[0]["supporting_sources"] if "codex.rollout" in s["path"]]
    if len(paths) != 1:
        raise ValueError("Expected exactly one authoritative Codex rollout")
    source = paths[0]
    records = [json.loads(line) for line in (raw / source).read_text().splitlines() if line.strip()]
    messages, mapping, excluded = [], [], []
    for line, record in enumerate(records, 1):
        payload = record.get("payload", {})
        start = len(messages)
        if record["type"] == "session_meta":
            messages.append({"role": "system", "content": payload["base_instructions"]["text"]})
        elif record["type"] == "response_item":
            kind = payload["type"]
            if kind == "reasoning" or payload.get("channel") in {"analysis", "reasoning", "thinking"}:
                excluded.append({"source_line": line, "type": kind, "reason": "No standard Chat Completions reasoning message field"})
                continue
            if kind == "message":
                messages.append({"role": payload["role"], "content": content_parts(payload["content"], images)})
            elif kind in {"function_call", "custom_tool_call"}:
                call = {"id": payload["call_id"]}
                if kind == "custom_tool_call":
                    call.update(type="custom", custom={"name": payload["name"], "input": payload["input"]})
                else:
                    call.update(type="function", function={"name": payload["name"], "arguments": payload["arguments"]})
                messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
            elif kind in {"function_call_output", "custom_tool_call_output"}:
                append_tool_result(messages, payload["call_id"], content_parts(payload["output"], images))
            else:
                raise ValueError(f"Unsupported response item at {source}:{line}: {kind}")
        if len(messages) > start:
            mapping.append({"source_line": line, "source_type": payload.get("type", record["type"]),
                            "message_start": start, "message_end": len(messages),
                            **({"tool_call_id": payload["call_id"]} if "call_id" in payload else {})})
    metadata = {"source": f"../raw/{source}", "session_id": records[0]["payload"]["session_id"],
                "message_mapping": mapping, "excluded_events": excluded,
                "notes": ["Only response_item events are read for the conversation; event_msg mirrors are not duplicated.",
                          "Original system and developer instructions retain their roles and positions.",
                          "Tool image suffixes become user messages named tool_output; mapping preserves the originating call ID."]}
    return [{"model": "gpt-6-astra", "messages": messages}], [metadata]


def iterative_sessions(raw, images):
    checkpoint = read_json(raw / "process.json")
    prompt = (raw / "prompt.txt").read_text().rstrip()
    rows, sessions = [], []
    previous = None
    entries = list(checkpoint["iterations"])
    if checkpoint.get("pending_iteration"):
        entries.append(checkpoint["pending_iteration"])
    for entry in entries:
        iteration = entry["iteration"]
        prior_script = resolve_recorded(previous["code_path"], raw).read_text() if previous else None
        refs = image_refs(raw, checkpoint["settings"], iteration, previous)
        if not refs["gt_reference"] or (iteration and not refs["current_render"]):
            raise ValueError(f"Missing input images for iteration {iteration}")
        base_prompt = prompts(prompt, entry["phase"], prior_script, len(refs["gt_reference"]), len(refs["current_render"]))
        attempt_prompt = base_prompt
        for attempt in entry["attempts"]:
            source = resolve_recorded(attempt["raw_path"], raw)
            response = read_json(source)
            input_parts = [{"type": "text", "text": attempt_prompt}]
            for image_role in ("current_render", "gt_reference"):
                for path in refs[image_role]:
                    input_parts.extend(content_parts({"type": "input_image", "image_url": str(Path(path).resolve())}, images))
            messages = [{"role": "system", "content": prompt}, {"role": "user", "content": input_parts}]
            mapping, excluded, final_texts = [], [], []
            for line, text in enumerate(response["transcript"].splitlines(), 1):
                record = json.loads(text)
                if record["type"] != "item.completed":
                    continue
                item = record["item"]
                kind = item["type"]
                start = len(messages)
                if kind in {"agent_message", "reasoning"}:
                    messages.append({"role": "assistant", "content": item["text"]})
                    if kind == "agent_message":
                        final_texts.append(item["text"])
                elif kind in {"command_execution", "file_change", "mcp_tool_call", "web_search", "tool_call"}:
                    excluded.append({"source_line": line, "source_item_id": item["id"], "source_type": kind,
                                     "reason": "Tool activity excluded from iterative trajectories by user request"})
                else:
                    raise ValueError(f"Unsupported transcript item: {kind}")
                if len(messages) > start:
                    mapping.append({"source_line": line, "source_item_id": item["id"], "source_type": kind,
                                    "message_start": start, "message_end": len(messages)})
            if not final_texts or final_texts[-1] != response["text"]:
                raise ValueError(f"Transcript final reply differs from the archived response: {source}")
            rows.append({"model": "gpt-6-astra", "messages": messages})
            sessions.append({"source": f"../raw/{source.name}", "source_field": "transcript",
                             "iteration": iteration, "attempt": attempt["attempt"], "status": attempt["status"],
                             "message_mapping": mapping, "excluded_events": excluded,
                             "reconstructed_input_messages": [0, 1],
                             "input_image_sources": refs,
                             "notes": ["Each iteration/attempt was a fresh session and is stored in its own JSONL row.",
                                       "Input requests were not archived: prompts and image order are reconstructed using process.json, prompt.txt and the existing iterative exporter.",
                                       "Only input messages, recorded assistant analysis/text and the final output text are retained, in original order.",
                                       "All tool calls, tool results and file change events are excluded; the temporary audit report records source locations only."]})
            attempt_prompt = retry_prompt(base_prompt, attempt, response["text"], raw)
        previous = entry
    return rows, sessions


def validate(rows, output):
    """Validate message roles, tool sequencing and every ordered image reference."""
    stats = Counter(sessions=len(rows))
    image_paths = []
    for row in rows:
        if set(row) != {"model", "messages"} or row["model"] != "gpt-6-astra":
            raise ValueError("Unexpected request fields")
        pending, seen = set(), set()
        for message in row["messages"]:
            role = message["role"]
            stats["messages"] += 1
            if role not in {"system", "developer", "user", "assistant", "tool"}:
                raise ValueError(f"Invalid role: {role}")
            allowed = {"role", "content", "tool_call_id"} if role == "tool" else {"role", "content", "name", "tool_calls"} if role == "assistant" else {"role", "content", "name"}
            if set(message) - allowed:
                raise ValueError("Unexpected message fields")
            if pending and role != "tool":
                raise ValueError("A tool result must precede the next non-tool message")
            if role == "tool":
                pending.remove(message["tool_call_id"])
                stats["tool_results"] += 1
            for call in message.get("tool_calls", []):
                if role != "assistant" or call["id"] in seen:
                    raise ValueError("Invalid or repeated tool call")
                seen.add(call["id"])
                pending.add(call["id"])
                kind = call["type"]
                if kind not in {"custom", "function"} or set(call) != {"id", "type", kind}:
                    raise ValueError("Invalid tool call structure")
                body = call[kind]
                field = "input" if kind == "custom" else "arguments"
                if set(body) != {"name", field} or not isinstance(body[field], str):
                    raise ValueError("Tool inputs must be strings")
                if kind == "function":
                    json.loads(body["arguments"])
                stats[f"{kind}_calls"] += 1
            content = message["content"]
            if content is None:
                if role != "assistant" or not message.get("tool_calls"):
                    raise ValueError("Null content without an assistant tool call")
            elif isinstance(content, list):
                if not content:
                    raise ValueError("Empty content parts")
                for part in content:
                    if part["type"] == "text":
                        if set(part) != {"type", "text"} or not isinstance(part["text"], str):
                            raise ValueError("Invalid text part")
                    elif part["type"] == "image_url" and role == "user":
                        path = part["image_url"]["url"]
                        if Path(path).is_absolute() or Path(path).parts[0] != "images" or ".." in Path(path).parts:
                            raise ValueError(f"Invalid relative image URL: {path}")
                        data = (output / path).read_bytes()
                        if hashlib.sha256(data).hexdigest() != Path(path).stem:
                            raise ValueError(f"Image hash mismatch: {path}")
                        image_paths.append(path)
                    else:
                        raise ValueError(f"Invalid content part for role {role}")
            elif not isinstance(content, str):
                raise ValueError("Invalid message content")
        if pending:
            raise ValueError(f"Missing tool results: {pending}")
    stats.update(image_references=len(image_paths), unique_images=len(set(image_paths)))
    files = {p.relative_to(output).as_posix() for p in (output / "images").iterdir()}
    if files != set(image_paths):
        raise ValueError("Image directory differs from referenced images")
    return dict(stats)


def build_preview(instance, stage, report_dir):
    raw = instance / "raw"
    stage.mkdir(parents=True)
    images = ArchiveImages(raw, stage / "images")
    kind = read_json(raw / "session_order.json")["kind"]
    rows, sessions = iterative_sessions(raw, images) if kind == "iterative" else interactive_sessions(raw, images)
    if images.warnings:
        raise ValueError(images.warnings)
    stats = validate(rows, stage)
    for index, session in enumerate(sessions, 1):
        session["jsonl_line"] = index
        source = (instance / "post_train_trajectory" / session["source"]).resolve()
        session["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = {"format": "openai_chat_completions", "setting": instance.parent.parent.name,
                "instance": instance.name, "image_url_storage": "relative_to_trajectory_jsonl",
                "message_indexing": "zero-based, message_end exclusive", "statistics": stats, "sessions": sessions,
                "image_recoveries": list(images.recoveries.values())}
    (stage / "trajectory.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows))
    report_dir.mkdir(parents=True)
    (report_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    reloaded = [json.loads(line) for line in (stage / "trajectory.jsonl").read_text().splitlines()]
    if reloaded != rows:
        raise ValueError("JSONL round trip mismatch")
    validate(reloaded, stage)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", help="One instance directory name")
    parser.add_argument("--settings", nargs="+", choices=("Single-view", "Multi-view", "ActiveVisual", "Full3DInteraction"),
                        default=("Single-view", "Multi-view", "ActiveVisual", "Full3DInteraction"),
                        help="Settings to regenerate; defaults to all four")
    args = parser.parse_args()
    if Path(args.instance).name != args.instance:
        parser.error("instance must be a single directory name")
    bench = Path(__file__).resolve().parents[1] / "3DHarnessBench"
    temp = bench / "temp"
    temp.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="astra-chat-completions-", dir=temp))
    pending = []
    for setting in dict.fromkeys(args.settings):
        instance = bench / "runs" / setting / "gpt-6-astra" / args.instance
        stage = work / "stage" / setting
        stats = build_preview(instance, stage, work / "reports" / setting)
        pending.append((instance / "post_train_trajectory", stage, setting))
        print(json.dumps({"setting": setting, **stats}), flush=True)
    # Validate all selected settings before replacing any output, and preserve backups.
    for output, stage, setting in pending:
        backup = work / "backup" / setting
        backup.parent.mkdir(exist_ok=True)
        if output.exists():
            output.rename(backup)
        shutil.move(str(stage), str(output))
    print(f"Published {len(pending)} instances. Previous outputs: {work / 'backup'}")


if __name__ == "__main__":
    main()
