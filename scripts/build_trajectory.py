#!/usr/bin/env python3
"""Export one complete multimodal SFT trajectory per iterative-run instance.

Usage: python scripts/build_trajectory.py PATH/TO/run-root
The root may be an agent directory or a higher-level Single-view/Multi-view
directory. Existing trajectories are replaced from process.json and attempt
JSON files. Every trajectory starts with the supplied prompt.txt.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

if __package__:
    from .sft_common import ImageStore, make_row, read_prompt, write_trajectory
else:
    from sft_common import ImageStore, make_row, read_prompt, write_trajectory


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def text_file(path):
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def resolve_recorded(path: str | None, instance_dir: Path):
    if not path:
        return None
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate
    relative = instance_dir / candidate
    if relative.exists():
        return relative
    local = instance_dir / candidate.name
    if local.exists():
        return local
    # Run artifacts are sometimes moved between machines while checkpoints
    # retain their original absolute paths. Recover paths below the instance.
    normalized = candidate.as_posix()
    for marker in ("/iteration_renders/",):
        if marker in normalized:
            suffix = normalized.split(marker, 1)[1]
            recovered = instance_dir / "iteration_renders" / suffix
            if recovered.exists():
                return recovered
    return None


def image_refs(instance_dir: Path, settings: dict, iteration: int,
               previous_entry: dict | None):
    """Find the exact GT/render files supplied to this iteration where possible."""
    result = {"current_render": [], "gt_reference": []}
    if iteration and previous_entry:
        render_dir = resolve_recorded(previous_entry.get("renders_dir"), instance_dir)
        if render_dir and render_dir.is_dir():
            # 3DHarnessBench exposes rendered views in this fixed camera order.
            # Keep that order instead of relying on arbitrary directory order.
            view_order = ("Image_005.png", "Image_015.png",
                          "Image_025.png", "Image_035.png")
            render_paths = [render_dir / name for name in view_order
                            if (render_dir / name).is_file()]
            if settings.get("view_mode") == "single":
                wanted = settings.get("single_view", "Image_005.png")
                render_paths = [p for p in render_paths if p.name == wanted] or render_paths[:1]
            result["current_render"] = [str(p) for p in render_paths]
    # Locate the benchmark tree in common checkout/data layouts.
    mode = settings.get("view_mode")
    image_dir = settings.get("image_dir", "color_renders")
    single = settings.get("single_view", "Image_005.png")
    instance = (instance_dir.parent.name
                if instance_dir.name == "raw" else instance_dir.name)
    roots = []
    for parent in (instance_dir, *instance_dir.parents):
        # Current HarnessBench checkouts keep benchmark/ beside runs/; older
        # layouts used data/benchmark/. Support both from the run directory.
        roots.extend((parent / "benchmark", parent / "data" / "benchmark"))
    for root in roots:
        candidates = []
        for base in (root / instance / image_dir,
                     root / instance / "images",
                     root / instance):
            if mode == "single":
                p = base / single
                if p.is_file():
                    candidates.append(p)
                    break
            elif base.is_dir():
                # Match iterative_runner.discover_tasks(): all supported image
                # formats in lexical order, not only Image_*.png.
                extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
                candidates.extend(sorted(
                    p for p in base.iterdir()
                    if p.is_file() and p.suffix.lower() in extensions))
                if candidates:
                    break
        if candidates:
            result["gt_reference"] = [str(p) for p in candidates]
            break
    return result


def prompts(base_prompt: str | None, phase: str, previous_script: str | None,
            n_gt: int, n_current: int):
    if not base_prompt:
        return None
    if phase == "generation":
        return (f"{n_gt} GT reference image(s) of one target object are attached. "
                "Inspect all of them before writing the reconstruction.")
    return f"""The attached images are ordered as follows: first {n_current} image(s)
are renders of the CURRENT generated script, followed by {n_gt} GT reference
image(s) of the TARGET object. They depict the current reconstruction and the
desired object respectively.

Here is the complete current Blender script:
```python
{previous_script or ''}
```

Compare the current renders against every GT view. Identify the largest visible
geometry, proportion, placement, silhouette, or missing-part errors, then edit
the script to correct them. Preserve useful detail already present. Return the
COMPLETE corrected self-contained Blender Python script, not a diff. Do not use
ellipsis and do not omit unchanged code. Output no prose outside the script.
"""


def retry_prompt(base_prompt: str, failed_attempt: dict, raw_text: str | None,
                 instance_dir: Path):
    if failed_attempt.get("status") == "ERR_PARSE":
        reply = raw_text or ""
        return f"""{base_prompt}

--- PREVIOUS REPLY WAS NOT A PARSABLE PYTHON SCRIPT ---
The previous reply could not be extracted as Python. Return one complete,
self-contained Blender Python script and no prose. Do not abbreviate it.

Previous reply:
{reply[-4000:]}
"""
    code_path = failed_attempt.get("code_path")
    script_path = resolve_recorded(code_path, instance_dir) if code_path else None
    script = text_file(script_path) if script_path else ""
    error = (failed_attempt.get("error") or "(no Blender error was recorded)").strip()
    return f"""{base_prompt}

--- PREVIOUS SCRIPT FAILED TO RENDER ---
The previous script failed in Blender with status {failed_attempt.get('status')}.

Failed script:
```python
{script or ''}
```

Blender error:
{error}

Fix the failure and return the COMPLETE corrected script. Output no prose
outside the script.
"""


def build_instance(instance_dir: Path, dry_run: bool = False,
                   prompt_file: Path | None = None):
    checkpoint_path = instance_dir / "process.json"
    checkpoint = read_json(checkpoint_path)
    if not isinstance(checkpoint, dict):
        return False
    settings = checkpoint.get("settings") or {}
    # Some exported runs place process.json under <instance>/raw/.
    instance = (instance_dir.parent.name
                if instance_dir.name == "raw" else instance_dir.name)
    prompt, prompt_source = read_prompt(instance_dir, prompt_file)
    items = []
    iteration_entries = list(checkpoint.get("iterations") or [])
    pending = checkpoint.get("pending_iteration")
    if isinstance(pending, dict):
        iteration_entries.append(pending)

    previous_entry = None
    for entry in iteration_entries:
        iteration = entry.get("iteration")
        if not isinstance(iteration, int):
            continue
        phase = entry.get("phase", "generation" if iteration == 0 else "editing")
        prior_code = None
        if iteration and previous_entry:
            code_path = resolve_recorded(previous_entry.get("code_path"), instance_dir)
            prior_code = text_file(code_path) if code_path else None
        # The benchmark prompt template is stored as prompt.txt. The full
        # runtime prompt can differ (generation/editing wrapper); record both
        # template and the available per-turn prompt context transparently.
        refs = image_refs(instance_dir, settings, iteration, previous_entry)
        if not refs["gt_reference"]:
            raise ValueError(f"{instance}: missing GT reference images")
        if iteration and not refs["current_render"]:
            raise ValueError(f"{instance}: iteration {iteration} is missing previous renders")
        base_prompt = prompts(prompt, phase, prior_code,
                              len(refs["gt_reference"]),
                              len(refs["current_render"]))
        turn_input = {"phase": phase, "prompt_template": prompt,
                      "prompt": base_prompt,
                      "prior_script": prior_code,
                      "images": refs}
        attempts = []
        attempt_prompt = base_prompt
        for attempt in entry.get("attempts", []):
            raw_path = resolve_recorded(attempt.get("raw_path"), instance_dir)
            raw = read_json(raw_path) if raw_path else None
            raw = raw if isinstance(raw, dict) else {}
            code_path = resolve_recorded(attempt.get("code_path"), instance_dir)
            output_text = raw.get("text")
            if output_text is None and code_path:
                output_text = text_file(code_path)
            actual_input = {**turn_input, "attempt_prompt": attempt_prompt}
            attempts.append({
                "attempt": attempt.get("attempt"),
                "input": actual_input,
                "output": {"text": output_text, "raw_response": raw,
                           "script": text_file(code_path) if code_path else None,
                           "status": attempt.get("status", raw.get("status")),
                           "error": attempt.get("error")},
            })
            attempt_prompt = retry_prompt(
                base_prompt or "", attempt, output_text or "", instance_dir)
        items.append({"iteration": iteration, "phase": phase,
                      "input": turn_input, "attempts": attempts,
                      "output": (attempts[-1]["output"] if attempts else None),
                      "status": entry.get("status")})
        previous_entry = entry

    # Keep the post-training trajectory beside `raw/`, matching the per-task
    # run layout users inspect (<instance>/post_train_trajectory/).
    trajectory_root = (instance_dir.parent
                       if instance_dir.name == "raw" else instance_dir)
    trajectory_dir = trajectory_root / "post_train_trajectory"
    if not dry_run:
        trajectory_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".trajectory-images-",
                                     dir=None if dry_run else trajectory_dir) as tmp:
        stage = Path(tmp)
        images = ImageStore(instance_dir, None if dry_run else stage)
        messages = []
        steps = []
        for item in items:
            for attempt in item["attempts"]:
                input_data = attempt["input"]
                output_text = attempt["output"]["text"]
                if not isinstance(output_text, str) or not output_text.strip():
                    raise ValueError(f"{instance}: missing response for iteration "
                                     f"{item['iteration']} attempt {attempt['attempt']}")
                blocks = [{"type": "text", "text": input_data["attempt_prompt"]}]
                # The observed image order is current render, then GT.
                for role in ("current_render", "gt_reference"):
                    for source in input_data["images"][role]:
                        path = images.save(str(Path(source).resolve()))
                        if path is None:
                            raise ValueError(f"{instance}: unreadable image {source}")
                        blocks.append({"type": "image", "image": path})
                steps.append({"iteration": item["iteration"], "phase": item["phase"],
                              "attempt": attempt["attempt"],
                              "status": attempt["output"]["status"],
                              "message_start": len(messages) + 1,
                              "message_end": len(messages) + 3})
                messages.extend([
                    {"role": "user", "content": blocks},
                    {"role": "assistant", "content": [{"type": "text", "text": output_text}]},
                ])
        row = make_row(messages, prompt, {
            "benchmark": trajectory_root.parent.parent.name,
            "model": settings.get("agent", trajectory_root.parent.name),
            "instance": instance, "source_kind": "iterative",
            "images_complete": True,
            "view_mode": settings.get("view_mode"), "prompt_source": prompt_source,
            "steps": steps,
        })
        if not dry_run:
            write_trajectory(trajectory_dir, row, stage)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="agent run root or mode run root")
    parser.add_argument("--dry-run", action="store_true", help="Parse without writing files")
    parser.add_argument("--prompt-file", type=Path, help="Override the supplied task prompt for every instance")
    args = parser.parse_args()
    root = args.path.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    dirs = [root] if (root / "process.json").is_file() else sorted(
        p.parent for p in root.rglob("process.json"))
    if not dirs:
        parser.error(f"no process.json checkpoints found under {root}")
    count = failed = 0
    for directory in dirs:
        try:
            if not build_instance(directory, args.dry_run, args.prompt_file):
                raise ValueError("cannot read process.json")
            count += 1
        except (OSError, ValueError) as exc:
            failed += 1
            print(f"error: {directory}: {exc}", file=sys.stderr)
    label = "Would write" if args.dry_run else "Wrote"
    print(f"{label} {count} trajectory.jsonl file(s) under {root}; {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
