"""Rebuild Astra archives using POST_TRAIN_TRAJECTORY_CONTRACT.md.

Run with Pixi: python -m scripts.rebuild_astra_contract [--instances NAME ...]
All outputs are staged and audited before publication. Original archives are
read only; replaced delivery directories are backed up under project temp/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image

from scripts.build_trajectory import image_refs, prompts, resolve_recorded, retry_prompt
from scripts.export_astra_chat_completions import ArchiveImages, content_parts, read_json
from tools.validate_trajectory import Trajectory, load_json


SETTINGS = ("Single-view", "Multi-view", "ActiveVisual", "Full3DInteraction")


class Evidence:
    def __init__(self):
        self.hashes = {}

    def source(self, path, output, locator):
        path = path.resolve()
        if path not in self.hashes:
            self.hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        return {"path": Path(os.path.relpath(path, output)).as_posix(),
                "sha256": self.hashes[path], "locator": locator}

    def verify(self):
        for path, expected in self.hashes.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Source changed during rebuilding: {path}")


def message(role, content, **fields):
    return {"role": role, "content": content, **fields}


def step(target, kind, source, *, context, provenance, status):
    return {"id": f"step_{target}", "context_indices": context,
            "tool_context_index": None,
            "target_index": target, "target_kind": kind, "status": status,
            "input_provenance": provenance, "available_tools": None,
            "tools_provenance": None, "source": source}


def trajectory(instance, session_id, messages, sources, steps, excluded, *, script):
    for index, item in enumerate(messages):
        item["index"] = index
    return {"schema_version": 1, "model": "gpt-6-astra", "messages": messages,
            "tools": [] if script else None,
            "metadata": {"scene": instance.parent.parent.name,
                         "instance": instance.name, "session_id": session_id,
                         "view": "script" if script else "agent", "sources": sources,
                         "tools_provenance": {"status": "not_applicable" if script else "unknown",
                                              "sources": []},
                         "steps": steps, "excluded_events": excluded}}


def iterative(instance, images, evidence):
    raw, output = instance / "raw", instance / "post_train_trajectory"
    checkpoint = read_json(raw / "process.json")
    prompt = (raw / "prompt.txt").read_text()
    if hashlib.sha256(prompt.encode()).hexdigest() != checkpoint["settings"]["prompt_sha256"]:
        raise ValueError(f"Task prompt differs from checkpoint: {instance}")
    rows, mappings = [], []
    previous = None
    entries = list(checkpoint["iterations"])
    if checkpoint.get("pending_iteration"):
        entries.append(checkpoint["pending_iteration"])
    for ei, entry in enumerate(entries):
        pointer = f"/iterations/{ei}" if ei < len(checkpoint["iterations"]) else "/pending_iteration"
        iteration = entry["iteration"]
        # restore_previous() uses the accepted script with trailing whitespace removed.
        prior_path = resolve_recorded(previous["code_path"], raw) if previous else None
        prior_script = prior_path.read_text().rstrip() if prior_path else None
        refs = image_refs(raw, checkpoint["settings"], iteration, previous)
        expected_gt = 1 if checkpoint["settings"]["view_mode"] == "single" else 4
        if len(refs["gt_reference"]) != expected_gt or (iteration and len(refs["current_render"]) != expected_gt):
            raise ValueError(f"Incomplete input images: {instance}, iteration {iteration}")
        # The script task view exposes images directly, rather than recreating
        # the teacher's workspace file manifest and image-opening tool calls.
        base_prompt = prompts(prompt.rstrip(), entry["phase"], prior_script,
                              len(refs["gt_reference"]), len(refs["current_render"]))
        attempt_prompt = base_prompt
        for ai, attempt in enumerate(entry["attempts"]):
            response_path = resolve_recorded(attempt["raw_path"], raw)
            response = read_json(response_path)
            script_path = resolve_recorded(attempt.get("code_path"), raw)
            text = response["text"]
            if script_path is None or script_path.read_text().strip() not in text:
                raise ValueError(f"Response does not contain the full saved script: {response_path}")
            sources = [evidence.source(raw / "process.json", output, f"{pointer}/attempts/{ai}"),
                       evidence.source(raw / "prompt.txt", output, "/"),
                       evidence.source(response_path, output, "/text"),
                       evidence.source(script_path, output, "/"),
                       evidence.source(Path(__file__), output, "line:1"),
                       evidence.source(Path(__file__).with_name("build_trajectory.py"), output,
                                       "line:1")]
            if prior_path:
                sources.append(evidence.source(prior_path, output, "/"))
            blocks = [{"type": "text", "text": attempt_prompt}]
            for image_role in ("current_render", "gt_reference"):
                for path in refs[image_role]:
                    sources.append(evidence.source(Path(path), output, "/"))
                    blocks.extend(content_parts({"type": "input_image", "image_url": str(Path(path).resolve())}, images))
            excluded, thread_ids = [], []
            for li, line in enumerate(response["transcript"].splitlines(), 1):
                record = load_json(line)
                if record["type"] == "thread.started":
                    thread_ids.append(record["thread_id"])
                if record["type"] == "item.completed":
                    item = record["item"]
                    if item["type"] == "agent_message" and item.get("text") == text:
                        continue
                    reason = ("推理事件仅保留在原始日志，不作为脚本答案。" if item["type"] == "reasoning"
                              else "脚本任务视图仅交付独立请求和最终完整脚本；教师内部过程保留在原始日志。")
                    excluded.append({"source": evidence.source(response_path, output,
                                                               f"/transcript#line:{li}"), "reason": reason})
            if len(thread_ids) != 1:
                raise ValueError(f"Expected one independent session: {response_path}")
            status = "OK" if attempt["status"] == "OK" else "ERROR"
            messages = [message("system", prompt.rstrip()), message("user", blocks), message("assistant", text)]
            rows.append(trajectory(instance, thread_ids[0], messages, sources,
                                   [step(2, "script", sources[0], context=[0, 1],
                                         provenance="reconstructed", status=status)], excluded, script=True))
            mappings.append({"session_id": thread_ids[0], "iteration": iteration,
                             "attempt": attempt["attempt"], "recorded_status": attempt["status"],
                             "input_image_sources": refs})
            # Match the runner's extracted script, which is stripped before
            # render_feedback(), rather than the newline-terminated saved file.
            if attempt["status"] == "ERR_PARSE":
                attempt_prompt = retry_prompt(base_prompt, attempt, text, raw)
            else:
                script = script_path.read_text().strip()
                error = (attempt.get("error") or "(no Blender error was recorded)").strip()
                attempt_prompt = (f"{base_prompt}\n\n--- PREVIOUS SCRIPT FAILED TO RENDER ---\n"
                                  f"The previous script failed in Blender with status {attempt['status']}.\n\n"
                                  f"Failed script:\n```python\n{script}\n```\n\nBlender error:\n{error}\n\n"
                                  "Fix the failure and return the COMPLETE corrected script. Output no prose\noutside the script.\n")
        previous = entry
    return rows, mappings


def interactive(instance, images, evidence):
    raw, output = instance / "raw", instance / "post_train_trajectory"
    order_path = raw / "session_order.json"
    order = read_json(order_path)
    rows, mappings, seen_paths = [], [], set()
    for attempt in order["attempts"]:
        if attempt["resume_mode"] != "fresh":
            raise ValueError(f"Resumed session requires explicit boundary recovery: {instance}")
        paths = [raw / s["path"] for s in attempt["supporting_sources"] if "codex.rollout" in s["path"]]
        if len(paths) != 1 or paths[0] in seen_paths:
            raise ValueError(f"Ambiguous authoritative rollout: {instance}")
        path = paths[0]
        seen_paths.add(path)
        records = [load_json(line) for line in path.read_text().splitlines()]
        messages, origins, excluded, generation_keys, phases = [], [], [], [], []
        sources = [evidence.source(order_path, output, "/attempts"), evidence.source(path, output, "/")]
        session_id = None
        for li, record in enumerate(records, 1):
            payload = record.get("payload", {})
            source = evidence.source(path, output, f"line:{li}")
            if record["type"] == "compacted":
                raise ValueError(f"Compaction requires explicit context recovery: {path}:{li}")
            if record["type"] == "session_meta":
                session_id = payload["session_id"]
                item = message("system", payload["base_instructions"]["text"])
                key, phase = None, None
            elif record["type"] == "response_item":
                kind = payload["type"]
                if kind == "reasoning" or payload.get("channel") in {"analysis", "reasoning", "thinking"}:
                    excluded.append({"source": source, "reason": "推理事件仅保留在原始日志，不混入普通 assistant 答案。"})
                    continue
                meta = payload.get("internal_chat_message_metadata_passthrough", {})
                key = (meta.get("turn_id"), meta.get("create_time"))
                if None in key:
                    key = None
                phase = payload.get("phase")
                if kind == "message":
                    item = message(payload["role"], content_parts(payload["content"], images))
                elif kind in {"function_call", "custom_tool_call"}:
                    call = {"id": payload["call_id"]}
                    if kind == "custom_tool_call":
                        call.update(type="custom", custom={"name": payload["name"], "input": payload["input"]})
                    else:
                        call.update(type="function", function={"name": payload["name"], "arguments": payload["arguments"]})
                    if messages and messages[-1]["role"] == "assistant" and key is not None and generation_keys[-1] == key:
                        messages[-1].setdefault("tool_calls", []).append(call)
                        origins[-1].append(source)
                        continue
                    item = message("assistant", None, tool_calls=[call])
                elif kind in {"function_call_output", "custom_tool_call_output"}:
                    item = message("tool", content_parts(payload["output"], images) or "",
                                   tool_call_id=payload["call_id"])
                else:
                    raise ValueError(f"Unsupported response item: {path}:{li}: {kind}")
            else:
                continue  # Runtime state and event_msg mirrors are not model messages.
            messages.append(item)
            origins.append([source])
            generation_keys.append(key)
            phases.append(phase)
        if session_id is None:
            raise ValueError(f"No session ID: {path}")
        steps = []
        for index, item in enumerate(messages):
            if item["role"] != "assistant":
                continue
            if item.get("tool_calls"):
                kind = "tool_call"
            elif phases[index] == "final_answer":
                kind = "final_reply"
            else:
                excluded.append({"source": origins[index][0],
                                 "reason": "独立 commentary 消息按原文保留；没有可靠的纯状态目标分类，不提供训练 step。"})
                continue
            # A rollout records emitted items, not complete per-request inputs.
            # [] makes no unsupported claim about a visible prefix. Unknown
            # inputs and unknown tool definitions make these audit-only steps.
            steps.append(step(index, kind, origins[index][0], context=[], provenance="unknown", status="UNKNOWN"))
        excluded.append({"source": sources[1],
                         "reason": "未归档完整顶层工具定义、exec 内部初始接口说明及逐次模型请求输入；tools=null，step.input_provenance=unknown；context_indices=[] 表示未确认任何上下文索引，仅供审计，禁止训练。"})
        rows.append(trajectory(instance, session_id, messages, sources, steps, excluded, script=False))
        mappings.append({"session_id": session_id, "message_sources": origins,
                         "tool_definitions": "unknown", "request_boundaries": "unknown"})
    return rows, mappings


def audit(rows, stage, instance, evidence, *, local_sources=None):
    stats = Counter(records=len(rows))
    session_ids, referenced = set(), set()
    output = instance / "post_train_trajectory"
    for row in rows:
        parsed = Trajectory.model_validate(row)
        metadata = row["metadata"]
        if metadata["scene"] != instance.parent.parent.name or metadata["instance"] != instance.name:
            raise ValueError("Directory metadata mismatch")
        if metadata["session_id"] in session_ids:
            raise ValueError("Duplicate session ID")
        session_ids.add(metadata["session_id"])
        stats["messages"] += len(parsed.messages)
        stats["excluded_events"] += len(metadata["excluded_events"])
        stats["audit_only_records"] += row["tools"] is None
        for s in metadata["steps"]:
            stats[f"steps_{s['target_kind']}_{s['status']}"] += 1
            stats[f"inputs_{s['input_provenance']}"] += 1
        calls, results = set(), set()
        sources = list(metadata["sources"])
        sources.extend(s["source"] for s in metadata["steps"])
        sources.extend(e["source"] for e in metadata["excluded_events"])
        sources.extend(metadata["tools_provenance"]["sources"])
        for s in metadata["steps"]:
            if s["tools_provenance"] is not None:
                sources.extend(s["tools_provenance"]["sources"])
        for source in sources:
            path = (local_sources or {}).get(source["path"], output / source["path"])
            actual = evidence.source(path, output, source["locator"])
            if actual["sha256"] != source["sha256"]:
                raise ValueError("Source digest mismatch")
        for item in row["messages"]:
            calls.update(c["id"] for c in item.get("tool_calls", []))
            if item["role"] == "tool":
                results.add(item["tool_call_id"])
            for block in item["content"] if isinstance(item["content"], list) else []:
                if block["type"] == "image_url":
                    stats["image_occurrences"] += 1
                    referenced.add(block["image_url"]["url"])
        if calls != results:
            raise ValueError(f"Incomplete tool results: {calls ^ results}")
        stats["tool_calls"] += len(calls)
        stats["tool_results"] += len(results)
    for relative in sorted(referenced):
        path = stage / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != path.stem:
            raise ValueError(f"Image hash mismatch: {path}")
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
    if referenced != {p.relative_to(stage).as_posix() for p in (stage / "images").glob("*")}:
        raise ValueError("Images differ from references")
    stats["unique_images"] = len(referenced)
    return dict(stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="+")
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bench = root / "3DHarnessBench"
    work = Path(tempfile.mkdtemp(prefix="astra-contract-", dir=root / "temp"))
    evidence, pending, reports = Evidence(), [], []
    for setting in SETTINGS:
        instances = sorted(p.parent.parent for p in (bench / "runs" / setting / "gpt-6-astra").glob("*/raw/session_order.json"))
        if args.instances:
            instances = [p for p in instances if p.name in args.instances]
            if {p.name for p in instances} != set(args.instances):
                raise ValueError(f"Requested instances missing in {setting}")
        if not instances:
            raise ValueError(f"No instances in {setting}")
        for instance in instances:
            stage = work / "stage" / setting / instance.name
            stage.mkdir(parents=True)
            images = ArchiveImages(instance / "raw", stage / "images")
            builder = iterative if setting in SETTINGS[:2] else interactive
            rows, mappings = builder(instance, images, evidence)
            if images.warnings:
                raise ValueError(images.warnings)
            path = stage / "trajectory.jsonl"
            path.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows), encoding="utf-8")
            reloaded = [load_json(line) for line in path.read_text().splitlines()]
            if reloaded != rows:
                raise ValueError("JSONL roundtrip mismatch")
            stats = audit(reloaded, stage, instance, evidence)
            report = {"scene": setting, "instance": instance.name, "statistics": stats,
                      "sessions": mappings, "image_recoveries": list(images.recoveries.values())}
            report_path = work / "reports" / setting / f"{instance.name}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            reports.append(report)
            pending.append((instance / "post_train_trajectory", stage))
            if len(pending) % 10 == 0 or args.instances:
                print(json.dumps({"staged_instances": len(pending), "scene": setting,
                                  "instance": instance.name, **stats}), flush=True)
    evidence.verify()
    summary = {"instances": len(pending), "statistics": dict(sum((Counter(r["statistics"]) for r in reports), Counter())),
               "scenes": {s: dict(sum((Counter(r["statistics"]) for r in reports if r["scene"] == s), Counter())) for s in SETTINGS},
               "published": False, "work_directory": str(work.relative_to(root)),
               "limitations": ["交互记录缺少完整工具定义、exec 初始内部接口说明和逐次请求输入，仅供审计。",
                               "单图、多图是按运行检查点重建的脚本任务视图，不声称复原教师内部工具会话。",
                               "当前训练数据读取器仍采用旧 ShareGPT 协议；本次仅整理数据，不修改训练代码，未运行训练 audit/preview。"]}
    if not args.stage_only:
        for output, stage in pending:
            backup = work / "backup" / output.parent.parent.parent.name / output.parent.name
            backup.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                output.rename(backup)
            stage.rename(output)
        # Check the published bytes and every image using the same contract audit.
        for output, _ in pending:
            rows = [load_json(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
            audit(rows, output, output.parent, evidence)
        evidence.verify()
        summary["published"] = True
        destination = bench / "metadata" / "trajectory_contract_v1"
        destination.mkdir(parents=True, exist_ok=True)
        for setting in SETTINGS:
            shutil.copytree(work / "reports" / setting, destination / setting, dirs_exist_ok=True)
        (destination / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (work / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
