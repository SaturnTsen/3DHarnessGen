"""Command arguments and process launch for checkpoint serving."""

import argparse
import json
import os
from pathlib import Path

from harness_training.serving import build_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a checkpoint with vLLM", allow_abbrev=False)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path)
    parser.add_argument("--adapter-name", "--model-name", dest="adapter_name")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print argv without starting vLLM")

    args, passthrough = parser.parse_known_args()
    if "--" in passthrough:
        passthrough.remove("--")
    command, name = build_command(args, passthrough)
    print(f"Client request model: {name}", flush=True)
    if args.dry_run:
        display = list(command)
        redact = False
        for index, argument in enumerate(display):
            if redact and not argument.startswith("--"):
                display[index] = "<redacted>"
                continue
            redact = False
            if argument == "--api-key":
                redact = True
            if argument.startswith("--api-key="):
                display[index] = "--api-key=<redacted>"
        print(json.dumps(display, ensure_ascii=False, indent=2))
        return
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
