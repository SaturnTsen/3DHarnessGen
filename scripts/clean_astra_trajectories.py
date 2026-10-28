"""Clean GPT-6-Astra trajectories without changing their source archives.

Run from the repository root with:
    pixi run python -m scripts.clean_astra_trajectories
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from PIL import Image

from harness_training.data.contract import AdditionalTools, Trajectory, load_json


SETTINGS = ("Single-view", "Multi-view", "ActiveVisual", "Full3DInteraction")
UNUSED_TOOLS = frozenset({
    "web__run", "image_gen__imagegen", "create_goal", "get_goal", "update_goal",
    "request_user_input", "request_user_input_async", "list_mcp_resource_templates",
    "list_mcp_resources", "read_mcp_resource", "followup_task", "interrupt_agent",
    "list_agents", "send_message", "spawn_agent", "wait_agent",
})


def result_failed(value: object) -> bool:
    if isinstance(value, str):
        return failure_text(value)
    if isinstance(value, list):
        return any(result_failed(item) for item in value)
    if not isinstance(value, dict):
        return False
    if value.get("isError") is True or value.get("status") in ("error", "failed", "failure", "rejected"):
        return True
    if type(value.get("exit_code")) is int and value["exit_code"] != 0:
        return True
    return any(result_failed(value[key]) for key in (
        "structuredContent", "content", "text", "result", "value", "output",
    ) if key in value)


def failure_text(text: str) -> bool:
    """Inspect actual result envelopes, not arbitrary mentions of an error."""
    if re.search(r"(?m)^(?:Script failed\b|Script error:|Traceback \(most recent call last\):)", text):
        return True
    body = text.split("Output:\n", 1)[-1].strip()
    body = re.sub(r"\AWarning: truncated output[^\n]*\nTotal output lines:[^\n]*\n\s*", "", body)
    decoder = json.JSONDecoder()
    while body:
        # Numeric labels are emitted before parallel result objects by exec.
        label = re.match(r"\d+\n", body)
        if label:
            body = body[label.end():].lstrip()
        if not body or body[0] not in '{["':
            return False
        try:
            value, end = decoder.raw_decode(body)
        except json.JSONDecodeError:
            return False
        if result_failed(value):
            return True
        body = body[end:].lstrip()
    return False


def failure_response(message: dict) -> bool:
    content = message["content"]
    if isinstance(content, str):
        return failure_text(content)
    # Separate text blocks may be viewport labels, JSON results, or logs.
    # Joining them would hide JSON results behind arbitrary label text.
    return any(failure_text(block["text"]) for block in content or [] if block["type"] == "text")


def used_tool_names(paths: list[Path]) -> set[str]:
    used: set[str] = set()
    pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(UNUSED_TOOLS))) + r")\b")
    for path in paths:
        for line in path.read_text().splitlines():
            row = cast(dict, load_json(line))
            for message in row["messages"]:
                for call in message.get("tool_calls") or []:
                    payload = call.get("custom") or call["function"]
                    used.add(payload["name"])
                    code = payload.get("input", payload.get("arguments", ""))
                    used.update(pattern.findall(code))
    return used


def prune_description(description: str, removed: set[str]) -> str:
    headings = list(re.finditer(r"(?m)^### `([^`]+)`[^\n]*\n", description))
    if not headings:
        return description
    parts = [description[:headings[0].start()]]
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(description)
        if heading.group(1) not in removed:
            parts.append(description[heading.start():end])
    cleaned = "".join(parts)
    for namespace, name in (("web", "web__run"), ("image_gen", "image_gen__imagegen")):
        if name in removed:
            cleaned = re.sub(rf"(?m)^## {namespace}\s*\n", "", cleaned)
    return cleaned


def prune_declaration(message: dict, removed: set[str]) -> dict:
    cleaned = copy.deepcopy(message)
    namespaces = []
    for namespace in cleaned["tools"]:
        tools = []
        for tool in namespace["tools"]:
            if tool["name"] in removed:
                continue
            tool["description"] = prune_description(tool["description"], removed)
            tools.append(tool)
        if tools:
            namespace["tools"] = tools
            namespaces.append(namespace)
    cleaned["tools"] = namespaces
    AdditionalTools.model_validate(cleaned)
    return cleaned


@dataclass
class ToolRound:
    call_index: int
    indices: list[int]
    failed: bool = False


def tool_rounds(row: dict) -> tuple[int, list[ToolRound], dict[int, int], set[int]]:
    messages = row["messages"]
    first_agent = next((i for i, m in enumerate(messages) if m["role"] == "assistant"), len(messages))
    failed_targets = {step["target_index"] for step in row["metadata"]["steps"] if step["status"] == "ERROR"}
    rounds: list[ToolRound] = []
    calls: dict[str, int] = {}
    owners: dict[int, int] = {}
    pending: list[int] = []
    for index in range(first_agent, len(messages)):
        message = messages[index]
        if message.get("tool_calls"):
            owner = len(rounds)
            rounds.append(ToolRound(index, [*pending, index], index in failed_targets))
            for member in rounds[-1].indices:
                owners[member] = owner
            pending.clear()
            for call in message["tool_calls"]:
                calls[call["id"]] = owner
        elif message["role"] == "tool":
            owner = calls[message["tool_call_id"]]
            owners[index] = owner
            rounds[owner].indices.append(index)
            if failure_response(message):
                rounds[owner].failed = True
        else:
            pending.append(index)
    failed_targets.update(turn.call_index for turn in rounds if turn.failed)
    return first_agent, rounds, owners, failed_targets


def clean_targets(row: dict, keep_rounds: int, removed: set[str]):
    """Yield one v3 record per target; current results never enter its input."""
    first_agent, rounds, owners, failed_targets = tool_rounds(row)
    declarations = {
        i: prune_declaration(message, removed)
        for i, message in enumerate(row["messages"])
        if message.get("type") == "additional_tools"
    }
    for step in row["metadata"]["steps"]:
        target = step["target_index"]
        if target in failed_targets:
            yield None, {"source_step_id": step["id"], "original_target_index": target,
                         "reason": "failed_target"}
            continue
        visible = set(step["context_indices"])
        history = [i for i, turn in enumerate(rounds) if turn.call_index in visible and not turn.failed]
        selected_rounds = history[-keep_rounds:]
        kept = {i for i in visible if i < first_agent}
        for owner in selected_rounds:
            kept.update(i for i in rounds[owner].indices if i in visible)
        # Keep the current agent work preceding the target, including status text.
        kept.update(i for i in visible if i >= first_agent and row["messages"][i]["role"] != "tool"
                    and (i not in owners or rounds[owners[i]].call_index >= target))
        kept.difference_update(failed_targets)
        selected = sorted(kept) + [target]
        cleaned = {key: copy.deepcopy(value) for key, value in row.items() if key not in {"messages", "tools"}}
        cleaned_messages: list[dict] = [
            copy.deepcopy(declarations[i] if i in declarations else row["messages"][i]) for i in selected
        ]
        cleaned["messages"] = cleaned_messages
        cleaned["tools"] = []
        for index, message in enumerate(cleaned_messages):
            message["index"] = index
            if message.get("type") == "additional_tools":
                context = [tool.model_dump(exclude_unset=True) for tool in
                           AdditionalTools.model_validate(message).tool_context()]
                if context not in cleaned["tools"]:
                    cleaned["tools"].append(context)
        cleaned_step = copy.deepcopy(step)
        cleaned_step["context_indices"] = list(range(len(selected) - 1))
        cleaned_step["target_index"] = len(selected) - 1
        cleaned["metadata"]["steps"] = [cleaned_step]
        cleaned["metadata"]["session_id"] = f'{row["metadata"]["session_id"]}:cleaned:{step["id"]}'
        yield cleaned, {
            "source_step_id": step["id"], "original_target_index": target,
            "selected_original_indices": selected, "new_target_index": len(selected) - 1,
            "history_tool_rounds": [rounds[i].call_index for i in selected_rounds],
        }


class ImageCopier:
    def __init__(self, source_dir: Path, stage_dir: Path):
        self.source_dir, self.stage_dir = source_dir, stage_dir
        self.references: dict[str, str] = {}
        self.digests: dict[str, str] = {}

    def copy(self, reference: str) -> str:
        if reference not in self.references:
            source = self.source_dir / reference
            payload = source.read_bytes()
            with Image.open(source) as image:
                if image.format != "PNG":
                    raise ValueError(f"Expected an original PNG image: {source}")
                image.verify()
            digest = hashlib.sha256(payload).hexdigest()
            destination = f"imgs/{digest}.png"
            path = self.stage_dir / destination
            if digest not in self.digests:
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, path)
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Image copy does not match source: {source}")
                self.digests[digest] = destination
            self.references[reference] = destination
        return self.references[reference]


def image_blocks(row: dict):
    for message in row["messages"]:
        content = message.get("content")
        if isinstance(content, list):
            yield from (block for block in content if block["type"] == "image_url")


def relocate_sources(row: dict, source: Path, destination: Path, digest: str, line: int) -> None:
    for item in [*row["metadata"]["sources"], row["metadata"]["steps"][0]["source"]]:
        item["path"] = Path(os.path.relpath((source.parent / item["path"]).resolve(), destination.parent)).as_posix()
    row["metadata"]["sources"].append({
        "path": Path(os.path.relpath(source, destination.parent)).as_posix(),
        "sha256": digest, "locator": f"line:{line}",
    })


def clean_dataset(runs_root: Path, output_root: Path, settings: list[str], keep_rounds: int) -> dict:
    runs_root, output_root = runs_root.resolve(), output_root.resolve()
    if keep_rounds < 1:
        raise ValueError("keep_tool_rounds must be positive")
    if output_root.is_relative_to(runs_root) or runs_root.is_relative_to(output_root):
        raise ValueError("Input and output trees must not overlap")
    if output_root.exists():
        raise FileExistsError(f"Output already exists; select a new output directory: {output_root}")
    paths = []
    for setting in settings:
        selected = sorted((runs_root / setting / "gpt-6-astra").glob("*/post_train_trajectory/trajectory.jsonl"))
        if not selected:
            raise ValueError(f"No GPT-6-Astra trajectories for {setting}")
        paths.extend(selected)
    removed = set(UNUSED_TOOLS) - used_tool_names(paths)
    report = {"keep_history_tool_rounds": keep_rounds, "removed_unused_tools": sorted(removed),
              "settings": {}, "source_sha256": {}}
    temp_root = Path.cwd() / "temp"
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="clean-astra-", dir=temp_root) as work:
        stage = Path(work) / "trajectory_cleaned"
        stage.mkdir()
        with (stage / "cleaning_manifest.jsonl").open("w", encoding="utf-8") as manifest:
            for source in paths:
                setting, instance = source.parents[3].name, source.parents[1].name
                stats = report["settings"].setdefault(setting, Counter())
                relative = Path(setting) / instance / "trajectory.jsonl"
                destination = output_root / relative
                stage_file = stage / relative
                stage_file.parent.mkdir(parents=True)
                copier = ImageCopier(source.parent, stage_file.parent)
                payload = source.read_bytes()
                digest = hashlib.sha256(payload).hexdigest()
                report["source_sha256"][str(source)] = digest
                stats["source_files"] += 1
                source_images: set[str] = set()
                output_line = 0
                with stage_file.open("w", encoding="utf-8") as output:
                    for line, raw in enumerate(payload.decode("utf-8").splitlines(), 1):
                        row = cast(dict, load_json(raw))
                        Trajectory.model_validate(row)
                        if row["metadata"]["scene"] != setting or row["metadata"]["instance"] != instance:
                            raise ValueError(f"Source metadata does not match path: {source}:{line}")
                        stats["input_records"] += 1
                        stats["input_targets"] += len(row["metadata"]["steps"])
                        source_images.update(block["image_url"]["url"] for block in image_blocks(row))
                        for cleaned, info in clean_targets(row, keep_rounds, removed):
                            info.update(source_file=str(source), source_line=line,
                                        source_session_id=row["metadata"]["session_id"])
                            if cleaned is None:
                                stats["removed_failed_targets"] += 1
                            else:
                                for block in image_blocks(cleaned):
                                    block["image_url"]["url"] = copier.copy(block["image_url"]["url"])
                                    stats["output_image_references"] += 1
                                relocate_sources(cleaned, source, destination, digest, line)
                                Trajectory.model_validate(cleaned)
                                output.write(json.dumps(cleaned, ensure_ascii=False, separators=(",", ":")) + "\n")
                                output_line += 1
                                info.update(output_file=relative.as_posix(), output_line=output_line)
                                stats["output_records"] += 1
                            manifest.write(json.dumps(info, ensure_ascii=False) + "\n")
                stats["source_unique_images"] += len(source_images)
                stats["used_source_images"] += len(copier.references)
                stats["copied_images"] += len(copier.digests)
                stats["unreferenced_source_images"] += len(source_images - copier.references.keys())
                if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Source changed during cleaning: {source}")
                print(f"[clean] {setting}/{instance}: {output_line} targets", flush=True)
        (stage / "cleaning_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        output_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(stage, output_root)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("3DHarnessBench/runs"))
    parser.add_argument("--output-root", type=Path, default=Path("data/trajectory_cleaned"))
    parser.add_argument("--settings", choices=SETTINGS, nargs="+", default=list(SETTINGS))
    parser.add_argument("--keep-tool-rounds", type=int, default=5)
    args = parser.parse_args()
    report = clean_dataset(args.runs_root, args.output_root, args.settings, args.keep_tool_rounds)
    print(json.dumps(report["settings"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
