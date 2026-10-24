"""Audit every Astra delivery against the contract, source hashes and images.

Run with ``pixi run python -m scripts.audit_astra_contract``. This command only
reads trajectories, original logs and images; it writes audit reports after all
instances have passed. It never executes recorded tools or loads model weights.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scripts.rebuild_astra_contract import SETTINGS, Evidence, audit, environment_message, indexed_tools
from tools.validate_trajectory import load_json


def main():
    root = Path(__file__).resolve().parents[1]
    contract = root / "docs/POST_TRAIN_TRAJECTORY_CONTRACT.md"
    evidence = Evidence()
    contract_source = evidence.source(contract, root, "/")
    checked_at = datetime.now(timezone.utc).isoformat()
    reports = []
    coverage = {}
    for setting in SETTINGS:
        base = root / "3DHarnessBench/runs" / setting / "gpt-6-astra"
        expected = {path.parent.parent for path in base.glob("*/raw/session_order.json")}
        delivered = {path.parent.parent for path in base.glob("*/post_train_trajectory/trajectory.jsonl")}
        if not expected or expected != delivered:
            raise ValueError(f"Incomplete coverage in {setting}: missing={expected - delivered}, extra={delivered - expected}")
        coverage[setting] = len(expected)
        for instance in sorted(expected):
            output = instance / "post_train_trajectory"
            path = output / "trajectory.jsonl"
            rows = [load_json(line) for line in path.read_text(encoding="utf-8").splitlines()]
            if not rows:
                raise ValueError(f"Empty JSONL: {path}")
            statistics = audit(rows, output, instance, evidence)
            recovery = load_json((output / "tool_context_recovery.json").read_text())
            origins = load_json((output / "recovery_report.json").read_text())
            if [row["metadata"]["session_id"] for row in rows] != [s["session_id"] for s in origins["sessions"]]:
                raise ValueError(f"Recovery report session mismatch: {instance}")
            for source in recovery["recovery"]["sources"]:
                actual = evidence.source(Path(source["path"]), output, "/")
                if source["sha256"] != actual["sha256"]:
                    raise ValueError(f"Tool recovery evidence changed: {source['path']}")
            additional, environment_count, statuses = 0, 0, Counter()
            for row, mapping in zip(rows, origins["sessions"], strict=True):
                if row["model"] != "gpt-6-astra":
                    raise ValueError(f"Teacher model mismatch: {path}")
                native = [m for m in row["messages"] if m.get("type") == "additional_tools"]
                if len(native) != 1 or native[0]["index"] != 0 or native[0]["tools"] != recovery["tools"]:
                    raise ValueError(f"Recovered native tools mismatch: {path}")
                if row["tools"] != [indexed_tools(recovery["tools"])]:
                    raise ValueError(f"Recovered tool contexts mismatch: {path}")
                additional += len(native)
                environments = [m for m in row["messages"] if m["role"] == "user"
                                and isinstance(m.get("content"), str)
                                and m["content"].startswith("<environment_context>\n")]
                environment_count += len(environments)
                if row["metadata"]["view"] == "agent":
                    source = row["metadata"]["sources"][1]
                    records = [load_json(line) for line in (output / source["path"]).read_text().splitlines()]
                    states = [(i, r) for i, r in enumerate(records, 1) if r["type"] == "world_state"]
                    expected_environments = [environment_message(r["payload"]["state"]["environments"]) for _, r in states]
                    actual_environments = [{k: v for k, v in m.items() if k != "index"} for m in environments]
                    if actual_environments != expected_environments:
                        raise ValueError(f"Historical environment mismatch: {path}")
                    if len(mapping["message_sources"]) != len(row["messages"]):
                        raise ValueError(f"Message source mapping mismatch: {path}")
                    for message, (line, _) in zip(environments, states, strict=True):
                        if not any(s["path"] == source["path"] and s["locator"] == f"line:{line}"
                                   for s in mapping["message_sources"][message["index"]]):
                            raise ValueError(f"Environment source location mismatch: {path}")
                for step in row["metadata"]["steps"]:
                    statuses[step["status"]] += 1
            statistics.update(additional_tools_inputs=additional, environment_inputs=environment_count)
            reports.append((output, {
                "result": "PASS", "checked_at": checked_at,
                "verification": contract_source,
                "scene": setting, "instance": instance.name,
                "trajectory": evidence.source(path, output, "/"),
                "statistics": statistics, "execution_status_counts": dict(statuses),
                "checks": ["strict_jsonl_and_contract_v3", "directory_and_session_identity",
                           "message_and_step_indices", "visible_additional_tools_match_context",
                           "native_tools_match_recovery_evidence", "call_result_identity_and_visibility",
                           "all_source_digests", "historical_environment_content_and_source_location",
                           "all_image_occurrences_and_files", "image_digest_and_decode"],
                "training_verified": False,
            }))
            if len(reports) % 25 == 0:
                print(json.dumps({"audited_instances": len(reports), "scene": setting}), flush=True)
    # Include every delivery and all sources in the final immutability check.
    evidence.verify()
    destination = root / "3DHarnessBench/metadata/trajectory_contract_v3/audit"
    for output, report in reports:
        report["verification"] = {**contract_source, "path": "docs/POST_TRAIN_TRAJECTORY_CONTRACT.md",
                                  "path_base": "project_root"}
        raw = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        (output / "audit_report.json").write_text(raw)
        path = destination / report["scene"] / f"{report['instance']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(raw)
    summary = {
        "result": "PASS", "schema_version": 3, "checked_at": checked_at,
        "verification": {**contract_source, "path_base": "project_root"},
        "instances": len(reports), "failed_instances": 0, "coverage": coverage,
        "statistics": dict(sum((Counter(r["statistics"]) for _, r in reports), Counter())),
        "execution_status_counts": dict(sum((Counter(r["execution_status_counts"]) for _, r in reports), Counter())),
        "source_files_verified": len(evidence.hashes),
        "limitations": ["工具来源仍为 recovered，输入来源仍为 reconstructed；未取得原会话的逐次 HTTP 请求。",
                        "UNKNOWN 只表示执行结果未知，不因审计通过而改为 OK。",
                        "未验证训练读取器、对象划分、模型标签或服务上传，未执行轨迹代码或训练。"],
    }
    (destination / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
