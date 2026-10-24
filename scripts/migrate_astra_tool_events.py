"""Restore Astra tool event boundaries from source logs, staging before replacement.

Run with Pixi: python -m scripts.migrate_astra_tool_events --tool-context PATH [--apply].
Original sessions and images are read only. Backups and staging live in temp/.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from harness_training.data.contract import Trajectory
from scripts.rebuild_astra_contract import Evidence, SETTINGS, audit, native_tool_event

CALLS = {"function_call", "custom_tool_call"}
OUTPUTS = {"function_call_output", "custom_tool_call_output"}


def restore_output(payload, previous):
    output = payload["output"]
    expected = previous["content"]
    if isinstance(output, str):
        if expected not in (output, [{"type": "text", "text": output}], "" if not output else output):
            raise ValueError("Tool output text differs from the source log")
        return output
    if len(output) != len(expected):
        raise ValueError("Tool output block count differs from the source log")
    restored = copy.deepcopy(output)
    for block, old in zip(restored, expected, strict=True):
        if block["type"] == "input_text":
            if old != {"type": "text", "text": block["text"]}:
                raise ValueError("Tool output text differs from the source log")
        elif block["type"] == "input_image":
            if old["type"] != "image_url" or old["image_url"].get("detail") != block.get("detail"):
                raise ValueError("Tool image position or detail differs from the source log")
            block["image_url"] = old["image_url"]["url"]
        else:
            raise ValueError(f"Unsupported native output block: {block['type']}")
    return restored


def restore_session(row, mapping, output, evidence):
    old_messages = row["messages"]
    origins = mapping["message_sources"]
    records = {}
    expanded, messages, sources = {}, [], []
    for old, refs in zip(old_messages, origins, strict=True):
        start = len(messages)
        if old.get("tool_calls") or old.get("role") == "tool":
            for ref in refs:
                path = (output / ref["path"]).resolve()
                if evidence.source(path, output, ref["locator"])["sha256"] != ref["sha256"]:
                    raise ValueError(f"Original source changed: {path}")
                if path not in records:
                    records[path] = [json.loads(line) for line in path.read_text().splitlines()]
                number = int(ref["locator"].removeprefix("line:"))
                record = records[path][number - 1]
                if record["type"] != "response_item":
                    raise ValueError("Tool source must be a response_item")
                payload = record["payload"]
                kind = payload["type"]
                if kind == "message":
                    if payload["role"] != "assistant":
                        raise ValueError("Merged text source is not assistant")
                    content = [{"type": "text", "text": b["text"]} for b in payload["content"]]
                    if content != old["content"]:
                        raise ValueError("Assistant text differs from source")
                    item = {"role": "assistant", "content": content}
                elif kind in CALLS:
                    legacy = {"id": payload["call_id"], "type": "custom" if kind == "custom_tool_call" else "function"}
                    legacy[legacy["type"]] = {"name": payload["name"],
                        **({"input": payload["input"]} if kind == "custom_tool_call" else {"arguments": payload["arguments"]})}
                    if legacy not in old["tool_calls"]:
                        raise ValueError("Tool call differs from source")
                    item = native_tool_event(payload, None)
                elif kind in OUTPUTS:
                    if old["tool_call_id"] != payload["call_id"]:
                        raise ValueError("Tool output call ID differs from source")
                    # Supply the already verified local image references, in order.
                    item = {k: copy.deepcopy(v) for k, v in payload.items()
                            if k in {"type", "call_id", "id", "status"}}
                    item["output"] = restore_output(payload, old)
                else:
                    raise ValueError(f"Unexpected source event: {kind}")
                item["index"] = len(messages)
                messages.append(item)
                sources.append([ref])
        else:
            item = copy.deepcopy(old)
            item["index"] = len(messages)
            messages.append(item)
            sources.append(refs)
        expanded[old["index"]] = list(range(start, len(messages)))
    steps = []
    for previous in row["metadata"]["steps"]:
        context = [i for old_index in previous["context_indices"] for i in expanded[old_index]]
        for index in expanded[previous["target_index"]]:
            item = messages[index]
            is_call = item.get("type") in CALLS
            if not is_call and item.get("role") != "assistant":
                raise ValueError("Invalid expanded training target")
            step = copy.deepcopy(previous)
            step.update(id=f"step_{index}", target_index=index, context_indices=context.copy(),
                        target_kind="tool_call" if is_call else
                        "status" if previous["target_kind"] == "tool_call" else previous["target_kind"],
                        source=sources[index][0])
            steps.append(step)
            context.append(index)
    row["messages"] = messages
    row["metadata"]["steps"] = steps
    mapping["message_sources"] = sources
    return len(messages) - len(old_messages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--tool-context", type=Path, required=True, help="Archived original tool context, checked by SHA-256")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    work = Path(tempfile.mkdtemp(prefix="native-tool-events-", dir=root / "temp"))
    evidence = Evidence()
    context_archive = args.tool_context.resolve()
    context_digest = hashlib.sha256(context_archive.read_bytes()).hexdigest()
    context_destination = root / "3DHarnessBench/metadata/trajectory_contract_v3/tool_contexts" / f"{context_digest}.json"
    pending, summaries = [], []
    edited_sources = {root / path for path in (
        "scripts/rebuild_astra_contract.py", "scripts/recover_interactive_contract.py",
        "scripts/audit_astra_contract.py", "docs/POST_TRAIN_TRAJECTORY_CONTRACT.md")}
    for scene in SETTINGS:
        base = root / "3DHarnessBench/runs" / scene / "gpt-6-astra"
        for path in sorted(base.glob("*/post_train_trajectory/trajectory.jsonl")):
            output = path.parent
            stage = work / "stage" / scene / output.parent.name
            stage.mkdir(parents=True)
            (stage / "images").symlink_to((output / "images").resolve(), target_is_directory=True)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            report_path = root / "3DHarnessBench/metadata/trajectory_contract_v3" / scene / f"{output.parent.name}.json"
            report = json.loads(report_path.read_text())
            added = 0
            context_relative = Path(os.path.relpath(context_destination, output)).as_posix()
            def relocate(value):
                if isinstance(value, dict):
                    if value.get("path") == "tool_context_recovery.json":
                        if value["sha256"] != context_digest:
                            raise ValueError("Archived tool context digest differs")
                        value["path"] = context_relative
                    for child in value.values():
                        relocate(child)
                elif isinstance(value, list):
                    for child in value:
                        relocate(child)
            relocate(rows)
            relocate(report)
            for row, mapping in zip(rows, report["sessions"], strict=True):
                if row["metadata"]["view"] == "agent":
                    added += restore_session(row, mapping, output, evidence)
                for source in row["metadata"]["sources"]:
                    source_path = (output / source["path"]).resolve()
                    if source_path in edited_sources:
                        source["sha256"] = evidence.source(source_path, output, source["locator"])["sha256"]
                Trajectory.model_validate(row)
            stats = audit(rows, stage, output.parent, evidence,
                          local_sources={context_relative: context_archive})
            report["statistics"] = stats
            report["tool_events"] = "native_session_events"
            report["added_messages"] = added
            (stage / "trajectory.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows))
            (stage / "recovery_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            summaries.append({"scene": scene, "instance": output.parent.name, "added_messages": added, **stats})
            pending.append((output, stage))
            if len(pending) % 25 == 0:
                print(json.dumps({"validated_instances": len(pending), "scene": scene}), flush=True)
    evidence.verify()
    summary = {"instances": len(pending), "added_messages": sum(r["added_messages"] for r in summaries),
               "statistics": dict(sum((Counter({k: v for k, v in r.items() if isinstance(v, int)}) for r in summaries), Counter())),
               "applied": False, "work_directory": str(work.relative_to(root)), "details": summaries}
    if args.apply:
        context_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(context_archive, context_destination)
        for output, stage in pending:
            backup = work / "backup" / output.parent.parent.parent.name / output.parent.name
            backup.mkdir(parents=True)
            for filename in ("trajectory.jsonl", "recovery_report.json", "audit_report.json"):
                if (output / filename).exists():
                    shutil.copy2(output / filename, backup / filename)
            (stage / "trajectory.jsonl").replace(output / "trajectory.jsonl")
            destination = root / "3DHarnessBench/metadata/trajectory_contract_v3" / output.parent.parent.parent.name
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination / f"{output.parent.name}.json", backup / "recovery_report.json")
            shutil.copy2(stage / "recovery_report.json", destination / f"{output.parent.name}.json")
        summary["applied"] = True
    (work / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "details"}), flush=True)


if __name__ == "__main__":
    main()
