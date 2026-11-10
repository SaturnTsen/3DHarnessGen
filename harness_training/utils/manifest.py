"""Record training provenance and fingerprint its inputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import cast

from harness_training.config import TrainingConfig
from harness_training.data.base import DatasetBundle
from harness_training.types import Manifest


def backend_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in ("torch", "transformers", "trl", "pydantic")}


def code_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {str(path.relative_to(root)): hash_file(path) for path in sorted(root.rglob("*.py"))}


def hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_sources(config: TrainingConfig) -> list[Path]:
    paths = sorted(config.model.path.glob("*.json")) + sorted(config.model.path.glob("*.jinja"))
    initialization = config.model.init_from
    if initialization is not None:
        paths += [initialization / name for name in (
            "adapter_config.json", "adapter_model.safetensors", "adapter_model.bin", "pytorch_model_fsdp_0/.metadata",
        ) if (initialization / name).is_file()]
        shards = initialization / "pytorch_model_fsdp_0"
        if (shards / ".metadata").is_file():
            run_dir = initialization.parent.parent if initialization.parent.name == "checkpoints" else initialization.parent
            paths.append(run_dir / "resolved_config.yaml")
            paths += sorted(shards.glob("*.distcp"))
    return paths


def hash_source(path: Path) -> str:
    # Fingerprint large FSDP shards by metadata without reading every tensor.
    if path.suffix == ".distcp":
        stat = path.stat()
        return hashlib.sha256(f"{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()
    return hash_file(path)


def build_manifest(config: TrainingConfig, data: DatasetBundle) -> Manifest:
    sources = dict(data.source_hashes)
    images = {path for split in (data.train, data.validation, data.test) for example in split
              for index in example.indices for path in example.record.images_by_message[index]}
    for path in sorted(images):
        if str(path) not in sources:
            sources[str(path)] = hash_file(path)
    for path in model_sources(config):
        if str(path) not in sources:
            sources[str(path)] = hash_source(path)
    payload = {
        "config": config.model_dump(mode="json"), "sources": sources,
        "versions": backend_versions(), "code": code_identity(),
        "targets": {split: [e.id for e in data.examples(split)] for split in ("train", "validation", "test")},
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return cast(Manifest, {**payload, "fingerprint": fingerprint})

