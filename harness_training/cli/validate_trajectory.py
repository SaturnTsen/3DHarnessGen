"""Validate trajectory documents using the package's v3 contract."""

import argparse
from pathlib import Path

from harness_training.data.contract import validate_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate v3 JSON/JSONL", allow_abbrev=False)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    passed = failed = 0
    sessions = set()
    for path in args.paths:
        text = path.read_text(encoding="utf-8")
        records = [text] if path.suffix == ".json" else text.splitlines()
        for line, raw in enumerate(records, 1):
            try:
                row = validate_record(raw)
                identity = (row.metadata.scene, row.metadata.instance, row.metadata.session_id)
                if identity in sessions:
                    raise ValueError("重复的 session_id")
                sessions.add(identity)
            except ValueError as exc:
                failed += 1
                print(f"{path}: 记录/行 {line}: {exc}")
            else:
                passed += 1
    print(f"{passed} 条通过，{failed} 条失败")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
