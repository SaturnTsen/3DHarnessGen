"""Validate data identity and migrate explicitly supported legacy resume states."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from harness_training.config import TrainingConfig
from .manifest import ResumeMismatch, differences


class DataProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1]
    global_step: int = Field(ge=0)
    epoch: int = Field(ge=0)
    next_sample: int = Field(ge=0)
    sample_count: int = Field(gt=0)
    global_batch_size: int = Field(gt=0)
    world_size: int = Field(gt=0)
    seed: int
    strategy: Literal["random", "sequential"]
    order: list[int]
    target_hash: str


def target_hash(targets: list[str]) -> str:
    return hashlib.sha256(json.dumps(targets).encode()).hexdigest()


def normalized_config(config: dict) -> dict:
    result = deepcopy(config)
    result.pop("distributed")
    result["model"].pop("max_length")
    result["template"].pop("max_length")
    method = result["method"]
    for key in ("resume_policy", "batch_size", "gradient_accumulation_steps"):
        method.pop(key, None)
    for key in ("fsdp", "fsdp_config"):
        method["backend"].pop(key)
    return result


def validate_manifest(prior: dict, current: dict) -> None:
    details = []
    for key in ("sources", "versions", "targets"):
        details.extend(differences(prior[key], current[key], key))
    details.extend(differences(normalized_config(prior["config"]), normalized_config(current["config"]), "config"))
    for section in ("model", "template"):
        previous = prior["config"][section]["max_length"]
        selected = current["config"][section]["max_length"]
        if selected < previous:
            details.append({"field": f"config.{section}.max_length", "change": "decreased",
                            "previous": previous, "current": selected,
                            "reason": "Restore the original limit or increase it; lowering may remove checkpoint samples"})
    known_legacy = json.loads(Path(__file__).with_name("legacy_code.json").read_text())
    if prior["code"] != current["code"] and prior["code"] not in known_legacy.values():
        details.append({"field": "code", "change": "not a supported version"})
        details.extend(differences(prior["code"], current["code"], "code"))
    if details:
        raise ResumeMismatch(details)


def load_progress(checkpoint: Path, prior: dict, config: TrainingConfig, world_size: int) -> DataProgress:
    from harness_training.data.elastic import epoch_order

    state = json.loads((checkpoint / "trainer_state.json").read_text())
    count = len(prior["targets"]["train"])
    global_batch = config.method.batch_size * config.method.gradient_accumulation_steps * world_size
    progress_file = checkpoint / "data_progress.json"
    if progress_file.is_file():
        if not (checkpoint / "portable_state.pt").is_file() or not (checkpoint / "resume_manifest.json").is_file():
            raise ValueError("Elastic checkpoint save did not finish: missing portable state or manifest")
        progress = DataProgress.model_validate_json(progress_file.read_text())
        if progress.version != 1 or progress.global_step != state["global_step"]:
            raise ValueError("Elastic resume: checkpoint data position and model step do not match")
    else:
        if prior["config"]["method"].get("resume_policy") == "elastic":
            raise ValueError("Elastic checkpoint save did not finish: missing data_progress.json")
        backend = prior["config"]["method"]["backend"]
        accelerator = backend["accelerator_config"]
        if backend["train_sampling_strategy"] == "random" and accelerator is not None and not accelerator.get("use_seedable_sampler", True):
            raise ValueError("Cannot reconstruct legacy data order without a seedable sampler")
        if backend["train_sampling_strategy"] not in {"random", "sequential"} or backend["ignore_data_skip"] or backend["dataloader_drop_last"]:
            raise ValueError("Unsupported legacy data sampling configuration")
        rng = list(checkpoint.glob("rng_state*.pth"))
        old_world = len(rng)
        expected = ({"rng_state.pth"} if old_world == 1 else
                    {f"rng_state_{rank}.pth" for rank in range(old_world)})
        if old_world == 0 or {path.name for path in rng} != expected:
            raise ValueError("Cannot determine legacy checkpoint process count from RNG states")
        method = prior["config"]["method"]
        old_batch = method["batch_size"] * method["gradient_accumulation_steps"] * old_world
        epochs, updates = divmod(state["global_step"], math.ceil(count / old_batch))
        seed = backend["data_seed"]
        if seed is None:
            seed = prior["config"]["experiment"]["seed"]
        progress = DataProgress(version=1, global_step=state["global_step"], epoch=epochs,
                                next_sample=updates * old_batch, sample_count=count,
                                global_batch_size=old_batch, world_size=old_world, seed=seed,
                                strategy=backend["train_sampling_strategy"],
                                order=epoch_order(count, seed, epochs, backend["train_sampling_strategy"]),
                                target_hash=target_hash(prior["targets"]["train"]))
    if progress.global_batch_size != global_batch:
        raise ValueError(f"Keep global batch size {progress.global_batch_size}: world_size * batch_size * gradient_accumulation_steps; got {global_batch}")
    expected_seed = config.method.backend.data_seed
    if expected_seed is None:
        expected_seed = config.experiment.seed
    if progress.seed != expected_seed or progress.strategy != config.method.backend.train_sampling_strategy:
        raise ValueError("Elastic resume: sampling seed or strategy changed")
    if progress.sample_count != count or progress.target_hash != target_hash(prior["targets"]["train"]):
        raise ValueError("Elastic resume: sample identity changed")
    if progress.order != epoch_order(count, progress.seed, progress.epoch, progress.strategy):
        raise ValueError("Elastic resume: saved sample order is incompatible")
    if not 0 <= progress.next_sample < count or progress.next_sample % global_batch:
        raise ValueError("Elastic resume: checkpoint is not at a global update boundary")
    if state["global_step"] != progress.epoch * math.ceil(count / global_batch) + progress.next_sample // global_batch:
        raise ValueError("Elastic resume: global step and saved cursor disagree")
    return progress
