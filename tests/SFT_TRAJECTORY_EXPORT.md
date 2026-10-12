# Qwen3.8-27B trajectory export

Both exporters use the same multimodal conversation format, intended for
Qwen3.8-27B / Transformers and subsequent TRL data preparation. This change
exports data only; it does not implement or launch SFT training.

## Commands

```bash
# Iterative runs: one instance, a model directory, or the entire runs tree.
python scripts/build_trajectory.py 3DHarnessBench/runs/Single-view
python scripts/build_trajectory.py 3DHarnessBench/runs/Multi-view

# Interactive runs: one instance, a model/setting directory, or the runs tree.
python scripts/export_sft_trajectories.py ActiveVisual/gpt-6-astra
python scripts/export_sft_trajectories.py Full3DInteraction
```

Both commands accept `--dry-run` and `--prompt-file PATH`. The latter overrides
the supplied task prompt for every instance in that invocation. Otherwise the
prompt comes from `raw/prompt.txt` (or `prompt.txt` in an unpacked iterative run).
Full3DInteraction uses its archived `raw/SKILL.md` when `prompt.txt` is absent.
Missing or empty prompts are reported as errors.

The interactive exporter's relative input paths resolve first under
`3DHarnessBench/runs/`, then under the current directory. Absolute paths work
in both commands. Recursive interactive export skips iterative runs.

## Output contract

```text
<instance>/post_train_trajectory/
├── trajectory.jsonl       # exactly ONE line for the entire instance
└── images/
    ├── <sha256>.png
    └── ...
```

Every row has `messages`, `images`, and `metadata`:

- `messages[0]` is a `system` message whose first text block is the supplied
  prompt. Existing system/developer context follows in that same system message,
  because the Qwen template permits only one system message at the beginning.
- Remaining messages use `user`, `assistant`, and `tool` roles. All `content`
  values are lists of text blocks and image placeholders (`{"type":"image"}`).
- `images` lists paths relative to the directory containing **that JSONL file**,
  in the exact order of image placeholders across all messages, including tool
  results. Repeated occurrences repeat the path; identical bytes are stored
  only once. The list length always equals the number of image placeholders.
- Tool calls use `assistant.tool_calls`, with `function.name` and a JSON object
  in `function.arguments`. Tool responses retain `tool_call_id` where recorded.
  Provider freeform arguments are preserved as `{"input": <original value>}`.
- `metadata` records the instance, benchmark, source model, prompt source, and
  source provenance. It is not conversation content or a training target.

Illustrative row (formatted over multiple lines here; the actual JSONL is one line):

```json
{
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "The supplied task prompt."}]},
    {"role": "user", "content": [{"type": "text", "text": "Reconstruct this object."}, {"type": "image"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "The initial Blender script."}]},
    {"role": "user", "content": [{"type": "text", "text": "Compare the current render with the reference and correct the script."}, {"type": "image"}, {"type": "image"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "The complete corrected Blender script."}]}
  ],
  "images": ["images/<reference-sha256>.png", "images/<render-sha256>.png", "images/<reference-sha256>.png"],
  "metadata": {"benchmark": "Single-view", "model": "gpt-6-astra", "instance": "ExampleFactory", "source_kind": "iterative"}
}
```

Iterative runs retain every recorded generation, edit, and render retry in
`process.json` order. The first user turn introduces the reference images;
the task prompt is already in the system message. Later user turns contain the
edit/retry instructions, prior scripts, and current-render-then-GT image order.
`metadata.steps` maps attempts to message spans (zero-based, end exclusive).

Interactive runs combine all recorded chains for the instance into one sample,
ordered by `session_order.json`. Continuations remain within their chain;
`metadata.chains` records chain boundaries and original session/attempt IDs.
Exact overlapping history in native continuation snapshots is included once.
Repeated actions within a transcript are retained. Attempts without any model
response are recorded in provenance and warnings, without fabricated targets.
Explicitly marked internal reasoning is omitted, as in the original exporter.
If a Kimi index contains only failed invocations, the exporter can recover
conversations from the instance's native main-session archives, ordered by
their recorded creation times. These appear as `recovered-native` provenance
with an explicit warning.

Images and JSONL are staged before publication. Successful exports replace the
old `trajectories.jsonl` and remove obsolete generated image files. Raw logs are
not modified. Failed instances keep their previous export and the command
returns a nonzero exit status while processing the other instances.

## Source limitations and TRL preparation

Some provider logs omit screenshots or replace them with text placeholders.
Some captured base64 images have been damaged by source-log redaction. The
exporter preserves the conversation and all recoverable images, marks an
unreadable image at its original position with a text notice, and records
`metadata.images_complete=false` plus warnings for known missing images.
No replacement image is invented. This flag can be used to exclude incomplete
visual samples during future training preparation. It does not certify that
an upstream provider captured every original event.

Before using these rows in TRL, resolve image paths against each JSONL's parent
and decode them as images. Preserve all occurrences and their order when
combining instances into a dataset. Tool argument objects can have different
schemas; use a JSON-capable dataset feature rather than inferring a single
fixed Arrow struct for all arguments.

The installed TRL multimodal helper skips image placeholders in `tool` messages.
For those trajectories, future data preparation must either handle tool images
explicitly or render `messages` with the model's chat template first and feed
TRL the resulting `text` plus the complete `images` list. Do not drop tool
screenshots or shift them into the initial user request. Full-trajectory training
also needs a length policy that does not silently truncate the conversation.

Format references:
[Qwen3.8-27B model](https://huggingface.co/Qwen/Qwen3.8-27B),
[official chat template](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/chat_template.jinja),
[TRL dataset formats](https://huggingface.co/docs/trl/dataset_formats),
[TRL SFT trainer](https://huggingface.co/docs/trl/sft_trainer).
