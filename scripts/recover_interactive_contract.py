"""Recover one uncompacted interactive delivery from an evidenced tool context.

The context JSON contains the original namespace-shaped tool specifications in
``tools`` and a ``recovery`` object documenting how they were recovered. This
command never executes recorded calls or loads model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from scripts.export_astra_chat_completions import ArchiveImages
from scripts.rebuild_astra_contract import Evidence, attach_additional_tools, audit, indexed_tools, interactive, step
from tools.validate_trajectory import Trajectory


def recover(row, mapping, context, source):
    if not context["recovery"]["static_tool_context"]:
        raise ValueError("Changing tool contexts require per-request snapshots")
    row["schema_version"] = 3
    row["tools"] = [indexed_tools(context["tools"])]
    metadata = row["metadata"]
    metadata["tools_provenance"] = [{"status": "recovered", "sources": [source]}]
    metadata["sources"].append(source)
    metadata["excluded_events"] = [event for event in metadata["excluded_events"]
                                   if "推理事件" in event["reason"]]
    metadata["steps"] = []
    for message in row["messages"]:
        if message["role"] != "assistant":
            continue
        index = message["index"]
        # This recovery is restricted to a single, fresh, uncompacted turn.
        # Tool discovery responses remain ordinary tool messages in the prefix.
        kind = "tool_call" if message.get("tool_calls") else "status"
        if index == len(row["messages"]) - 1 and kind == "status":
            kind = "final_reply"
        target = step(index, kind, mapping["message_sources"][index][0],
                      context=list(range(index)), provenance="reconstructed", status="UNKNOWN")
        metadata["steps"].append(target)
    attach_additional_tools(row, mapping, context["tools"])
    Trajectory.model_validate(row)
    mapping.update(tool_definitions="recovered", request_boundaries="reconstructed")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--tool-context", type=Path, required=True)
    args = parser.parse_args()
    instance = args.instance.resolve()
    if instance.parent.parent.name not in {"ActiveVisual", "Full3DInteraction"}:
        parser.error("Only interactive scenes are supported")
    raw, output = instance / "raw", instance / "post_train_trajectory"
    order = json.loads((raw / "session_order.json").read_text())
    for attempt in order["attempts"]:
        path = raw / next(s["path"] for s in attempt["supporting_sources"] if "codex.rollout" in s["path"])
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if sum(record["type"] == "turn_context" for record in records) != 1:
            raise ValueError("Recovery requires one fresh, uncompacted turn per session")
        if any(record["type"] == "compacted" for record in records):
            raise ValueError("Compaction requires explicit visible-context reconstruction")
    root = Path(__file__).resolve().parents[1]
    work = Path(tempfile.mkdtemp(prefix="interactive-recovery-", dir=root / "temp"))
    stage = work / "stage"
    stage.mkdir()
    evidence = Evidence()
    images = ArchiveImages(raw, stage / "images")
    rows, mappings = interactive(instance, images, evidence)
    context = json.loads(args.tool_context.read_text())
    for origin in context["recovery"]["sources"]:
        if hashlib.sha256(Path(origin["path"]).read_bytes()).hexdigest() != origin["sha256"]:
            raise ValueError(f"Tool recovery source digest mismatch: {origin['path']}")
    context_path = stage / "tool_context_recovery.json"
    context_path.write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n")
    # Both the staged and published evidence paths refer to the final delivery.
    source = {"path": "tool_context_recovery.json",
              "sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(), "locator": "/tools"}
    for row, mapping in zip(rows, mappings, strict=True):
        recover(row, mapping, context, source)
        row["metadata"]["sources"].append(evidence.source(Path(__file__), output, "/"))
    path = stage / "trajectory.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows))
    reloaded = [json.loads(line) for line in path.read_text().splitlines()]
    if reloaded != rows or images.warnings:
        raise ValueError("Delivery roundtrip or image recovery failed")
    digest = source["sha256"]
    statistics = audit(reloaded, stage, instance, evidence,
                       local_sources={source["path"]: context_path})
    report = {"scene": instance.parent.parent.name, "instance": instance.name,
              "sessions": mappings, "image_recoveries": list(images.recoveries.values()),
              "recovery": context["recovery"], "tool_context_sha256": digest,
              "statistics": statistics}
    (stage / "recovery_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    # Check source immutability before moving any existing delivery.
    evidence.verify()
    if output.exists():
        output.rename(work / "backup")
    stage.rename(output)
    # Evidence stores the staged path: replace that single key after publication.
    del evidence.hashes[context_path.resolve()]
    evidence.source(output / "tool_context_recovery.json", output, "/tools")
    report["statistics"] = audit(reloaded, output, instance, evidence)
    evidence.verify()
    (output / "recovery_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"backup": str(work / "backup"), **report["statistics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
