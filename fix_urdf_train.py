"""修复 Articraft URDF train.parquet 的任务指令和代码块类型。"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import pyarrow as pa
import pyarrow.parquet as pq


OLD_SYSTEM = (
    "You are an expert in parametric CAD with CadQuery. Write a complete, "
    "standalone CadQuery script that builds the requested part. "
    "Output ONLY the code in one ```python block."
)
NEW_SYSTEM = (
    "You are an expert in articulated 3D modeling with URDF. Write a complete, "
    "standalone URDF XML document that describes the requested object, "
    "including its links, joints, geometry, and materials where applicable. "
    "Output ONLY the URDF XML in one ```xml block."
)
OLD_REQUEST = "Write the CadQuery script that builds it."
NEW_REQUEST = "Write the URDF XML document that describes it."
SUBSETS = ("urdf_geo_only_llamafactory_img", "urdf_tex_llamafactory_img")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def repair_table(table: pa.Table) -> tuple[pa.Table, dict[str, int]]:
    systems = table.column("system").to_pylist()
    conversations = table.column("conversations").to_pylist()
    changes = Counter(system=0, user_request=0, code_fence=0)
    for index, (system, turns) in enumerate(zip(systems, conversations, strict=True)):
        if system not in (OLD_SYSTEM, NEW_SYSTEM):
            raise ValueError(f"第 {index} 行包含未识别的 system 指令")
        if system == OLD_SYSTEM:
            systems[index] = NEW_SYSTEM
            changes["system"] += 1
        for turn in turns:
            value = turn["value"]
            if turn["from"] == "human":
                if OLD_REQUEST in value:
                    turn["value"] = value.replace(OLD_REQUEST, NEW_REQUEST)
                    changes["user_request"] += 1
            elif turn["from"] == "gpt":
                lines = value.splitlines(keepends=True)
                if lines[0].strip() not in ("```python", "```xml") or lines[-1].strip() != "```":
                    raise ValueError(f"第 {index} 行的目标代码块格式不符合预期")
                xml = "".join(lines[1:-1])
                if ET.fromstring(xml).tag != "robot":
                    raise ValueError(f"第 {index} 行的 XML 根元素不是 robot")
                if lines[0].strip() == "```python":
                    turn["value"] = value.replace("```python", "```xml", 1)
                    changes["code_fence"] += 1
    result = table.set_column(
        table.schema.get_field_index("system"), table.schema.field("system"),
        pa.array(systems, type=table.schema.field("system").type),
    )
    result = result.set_column(
        table.schema.get_field_index("conversations"), table.schema.field("conversations"),
        pa.array(conversations, type=table.schema.field("conversations").type),
    )
    return result, dict(changes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parent
    parser.add_argument("--dataset-root", type=Path, default=project / "3dcodeverse")
    parser.add_argument("--backup-dir", type=Path, default=project / "temp/urdf_train_repair")
    parser.add_argument("--apply", action="store_true", help="备份并写入；不指定时只检查")
    args = parser.parse_args()
    prepared = []
    for subset in SUBSETS:
        path = args.dataset_root / "articraft" / subset / "train.parquet"
        original = pq.read_table(path)
        repaired, changes = repair_table(original)
        untouched = [name for name in original.column_names if name not in ("system", "conversations")]
        assert original.select(untouched).equals(repaired.select(untouched))
        assert original.schema.equals(repaired.schema, check_metadata=True)
        prepared.append((path, repaired, changes))

    backup = args.backup_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report = {"applied": args.apply, "files": []}
    for path, repaired, changes in prepared:
        item = {"path": str(path), "rows": repaired.num_rows, "changes": changes}
        if args.apply and any(changes.values()):
            folder = backup / path.parent.name
            folder.mkdir(parents=True)
            saved = folder / "train.parquet"
            shutil.copy2(path, saved)
            item["before_sha256"] = sha256(saved)
            staged = folder / "repaired.parquet"
            pq.write_table(repaired, staged, compression="snappy")
            if not pq.read_table(staged).equals(repaired, check_metadata=True):
                raise ValueError(f"写入后校验失败：{staged}")
            os.replace(staged, path)
            item["after_sha256"] = sha256(path)
            item["backup"] = str(saved)
        report["files"].append(item)
    if args.apply and backup.exists():
        (backup / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
