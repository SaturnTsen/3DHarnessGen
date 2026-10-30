"""Build and compare resume identities without a Processor or model weights."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from harness_training.config import TrainingConfig
from harness_training.data.base import PreparationWorkers, SerialPreparationWorkers


def config_identity(config: TrainingConfig) -> dict:
    result = config.model_dump(mode="json")
    result["experiment"].pop("output_dir")
    for field in ("max_steps", "epochs", "logging_steps", "save_steps", "eval_steps", "save_total_limit",
                  "dataloader_workers", "resume_from"):
        result["method"].pop(field)
    return result


def backend_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in ("torch", "transformers", "trl", "pydantic-yaml")}


def code_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
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
        if (initialization / "pytorch_model_fsdp_0/.metadata").is_file():
            paths.append(initialization.parent.parent / "resolved_config.yaml")
            paths += sorted((initialization / "pytorch_model_fsdp_0").glob("*.distcp"))
    return paths


def hash_source(path: Path) -> str:
    # Preserve the legacy fingerprint scheme for very large FSDP base-model shards.
    if path.suffix == ".distcp":
        stat = path.stat()
        return hashlib.sha256(f"{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()
    return hash_file(path)


def saved_source_inventory(prior: dict, data_paths: list[Path], config: TrainingConfig,
                           *, workers: PreparationWorkers | None = None) -> dict[str, str]:
    """Rehash saved images plus today's selected trajectories and model metadata."""
    old_model = prior["config"]["model"]
    old_root = Path(old_model["path"])
    old_init = Path(old_model["init_from"]) if old_model.get("init_from") is not None else None
    old_data_files = prior["config"]["data"]["files"]
    old_data = set(old_data_files) if old_data_files is not None else {
        name for name in prior["sources"] if Path(name).name == "trajectory.jsonl"}
    images = []
    for name in prior["sources"]:
        path = Path(name)
        model_file = path.parent == old_root or (old_init is not None and (
            path.is_relative_to(old_init) or path == old_init.parent.parent / "resolved_config.yaml"))
        if name not in old_data and not model_file:
            images.append(path)
    paths = sorted(set(images + data_paths + model_sources(config)))
    workers = workers if workers is not None else SerialPreparationWorkers()
    with workers.progress("核对恢复源文件摘要", len(paths)) as advance:
        local: dict[str, str] = {}
        errors: dict[str, str] = {}
        for path in paths[workers.rank::workers.world_size]:
            try:
                local[str(path)] = hash_source(path) if path.is_file() else "<missing>"
            except OSError as exc:
                errors[str(path)] = str(exc)
            advance()
        shards = workers.gather("resume_sources", (local, errors))
    sources = {}
    failures = {}
    for selected, failed in shards:
        sources.update(selected)
        failures.update(failed)
    if failures:
        path = sorted(failures)[0]
        raise OSError(f"Resume source check failed for {path}: {failures[path]}")
    return sources


def differences(previous: Any, current: Any, prefix: str) -> list[dict]:
    """Report exact fields; lists get counts and the first differing position."""
    if previous == current:
        return []
    if isinstance(previous, Mapping) and isinstance(current, Mapping):
        result = []
        for key in sorted(set(previous) | set(current)):
            name = f"{prefix}.{key}"
            if key not in previous:
                result.append({"field": name, "change": "added", "current": display(current[key], name)})
            elif key not in current:
                result.append({"field": name, "change": "removed", "previous": display(previous[key], name)})
            else:
                result.extend(differences(previous[key], current[key], name))
        return result
    if isinstance(previous, list) and isinstance(current, list):
        index = next((index for index, (a, b) in enumerate(zip(previous, current)) if a != b),
                     min(len(previous), len(current)))
        return [{"field": prefix, "change": "changed", "previous_count": len(previous),
                 "current_count": len(current), "first_difference": index,
                 "previous": display(previous[index], prefix) if index < len(previous) else "<end>",
                 "current": display(current[index], prefix) if index < len(current) else "<end>"}]
    return [{"field": prefix, "change": "changed", "previous": display(previous, prefix),
             "current": display(current, prefix)}]


def display(value: Any, field: str) -> Any:
    if any(secret in field.lower() for secret in ("hub_token", "api_key", "password", "secret")):
        return "<redacted>"
    if isinstance(value, (Mapping, list)):
        # Added containers may themselves contain credentials; expand their fields instead.
        return f"<{type(value).__name__}, {len(value)} entries>"
    return value


class ResumeMismatch(ValueError):
    def __init__(self, details: list[dict]):
        self.details = details
        lines = [f"{item['field']}: {json.dumps({key: value for key, value in item.items() if key != 'field'}, ensure_ascii=False)}"
                 for item in details[:10]]
        super().__init__("Resume compatibility check failed; files or settings changed since this checkpoint:\n" + "\n".join(lines) +
                         (f"\n... {len(details)} differences total" if len(details) > 10 else ""))
