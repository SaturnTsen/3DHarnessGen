import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness_training.config import DataConfig, TrainingConfig, load_config, save_config
from harness_training.data.trajectory import DataError, TrajectoryDataset
from harness_training.models.qwen import QwenModel
from harness_training.training.methods.sft import SFTMethod
from harness_training.training.trainer import Trainer
from conftest import write_trajectory


def test_config_roundtrip_paths_and_unknown_keys(config, tmp_path):
    path = tmp_path / "config.yaml"
    save_config(config, path)
    restored = load_config(path)
    assert restored == config
    path.write_text(path.read_text() + "typo_option: true\n")
    with pytest.raises(ValidationError):
        load_config(path)


@pytest.mark.parametrize("section", [
    (), ("experiment",), ("data",), ("template",), ("model",), ("method",),
    ("model", "lora"), ("model", "adapter"), ("model", "lora", "backend"),
    ("model", "lora", "backend", "runtime_config"), ("method", "backend"),
    ("method", "trainer"),
    ("method", "backend", "gradient_checkpointing_kwargs"), ("method", "backend", "dataset_kwargs"),
])
def test_config_requires_explicit_fields(config, section):
    fields = config.model_dump(mode="json")
    for name in section:
        fields = fields[name]
    for field in fields:
        incomplete = config.model_dump(mode="json")
        target = incomplete
        for name in section:
            target = target[name]
        del target[field]
        with pytest.raises(ValidationError) as caught:
            TrainingConfig.model_validate(incomplete)
        assert any(error["loc"] == section + (field,) and error["type"] == "missing"
                   for error in caught.value.errors())


def test_trainer_audit_uses_config_without_loading_processor(config, tmp_path):
    config.model.path = tmp_path / "unavailable_model"
    trainer = Trainer(config)
    assert isinstance(trainer.model, QwenModel)
    assert isinstance(trainer.method, SFTMethod)
    report = trainer.audit()
    assert report["samples"]["train"] == 2
    assert trainer.dataset.config is config.data
    assert trainer.model.config is config.model
    assert trainer.method.config is config.method
    assert trainer.dataset.seed == trainer.method.seed == config.experiment.seed
    assert trainer._template is None
    assert trainer.model._processor is None
    assert trainer.model.model is None


@pytest.mark.parametrize("section, message", [("model", "Unsupported model"), ("method", "Unsupported training method")])
def test_trainer_rejects_unimplemented_component_names(config, section, message):
    payload = config.model_dump(mode="json")
    payload[section]["name"] = "unimplemented"
    selected = TrainingConfig.model_validate(payload)
    with pytest.raises(ValueError, match=f"{message}: unimplemented"):
        Trainer(selected)


@pytest.mark.parametrize("section", ["model", "method"])
def test_component_names_cannot_be_empty(config, section):
    payload = config.model_dump(mode="json")
    payload[section]["name"] = ""
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(payload)


@pytest.mark.parametrize("field, value", [
    ("remove_unused_columns", True), ("packing", True), ("padding_free", True),
    ("assistant_only_loss", True), ("completion_only_loss", True), ("max_length", 1024),
])
def test_backend_rejects_settings_incompatible_with_indexed_samples(config, field, value):
    payload = config.model_dump(mode="json")
    payload["method"]["backend"][field] = value
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(payload)


def test_resume_path_resolves_from_yaml(config, tmp_path):
    config.method.resume_from = Path("checkpoints/checkpoint-1")
    path = tmp_path / "config.yaml"
    save_config(config, path)
    assert load_config(path).method.resume_from == tmp_path / "checkpoints/checkpoint-1"


def test_split_rejects_invalid_ratios(config):
    with pytest.raises(ValidationError, match="must be less than 1"):
        DataConfig.model_validate({**config.data.model_dump(), "validation_fraction": .6, "test_fraction": .4})


def test_step_selection_uses_source_mapping(tmp_path, config):
    write_trajectory(tmp_path, failed_first=True)
    cfg = config.data.model_copy(update={"runs_root": tmp_path})
    data = TrajectoryDataset(cfg, "Single-view", config.experiment.seed).prepare()
    assert [example.target_index for example in data.train] == [3]
    assert data.train[0].indices == (2, 3)
    assert data.train[0].record.images_by_conversation[2][0].name == "reference.png"


def test_missing_image_only_excludes_affected_step(tmp_path, config):
    write_trajectory(tmp_path, missing=True)
    cfg = config.data.model_copy(update={"runs_root": tmp_path})
    data = TrajectoryDataset(cfg, "Single-view", config.experiment.seed).prepare()
    assert [example.target_index for example in data.train] == [3]
    assert len(data.rejected) == 1 and "absent.png" in data.rejected[0]["reason"]
    cfg.missing_images = "error"
    with pytest.raises(DataError, match="absent.png"):
        TrajectoryDataset(cfg, "Single-view", config.experiment.seed).prepare()


def test_split_stays_same_across_scenes(tmp_path, config):
    for scene in ["Single-view", "Multi-view"]:
        for i in range(20):
            write_trajectory(tmp_path, scene, f"Object{i}Factory")
    cfg = config.data.model_copy(update={"runs_root": tmp_path, "validation_fraction": .1, "test_fraction": .1})
    a = TrajectoryDataset(cfg, "Single-view", config.experiment.seed).prepare()
    b = TrajectoryDataset(cfg, "Multi-view", config.experiment.seed).prepare()
    assert a.assignments == b.assignments
    assert a.assignments == TrajectoryDataset(cfg, "Single-view", config.experiment.seed).prepare().assignments
    assert a.assignments != TrajectoryDataset(cfg, "Single-view", config.experiment.seed + 1).prepare().assignments
    assert {name: list(a.assignments.values()).count(name) for name in ("train", "validation", "test")} == {
        "train": 16, "validation": 2, "test": 2,
    }
    for split in [a.train, a.validation, a.test]:
        for example in split:
            assert a.assignments[example.record.instance] == next(name for name in ["train", "validation", "test"]
                                                                 if split is getattr(a, name))


def test_split_uses_only_selected_scene_paths_and_instances(tmp_path, config):
    for index in range(20):
        write_trajectory(tmp_path, "Single-view", f"Object{index}Factory")
    for index in range(10):
        write_trajectory(tmp_path, "Multi-view", f"Object{index}Factory")
    selected_instances = [f"Object{index}Factory" for index in range(5)]
    cfg = config.data.model_copy(update={
        "runs_root": tmp_path, "validation_fraction": .2, "test_fraction": .2,
        "instances": selected_instances,
    })
    selected = TrajectoryDataset(cfg, "Single-view", config.experiment.seed).prepare()
    assert set(selected.assignments) == set(selected_instances)
    assert {name: list(selected.assignments.values()).count(name) for name in ("train", "validation", "test")} == {
        "train": 3, "validation": 1, "test": 1,
    }
    for index in range(20, 25):
        write_trajectory(tmp_path, "Single-view", f"Object{index}Factory")
        write_trajectory(tmp_path, "Multi-view", f"Object{index}Factory")
    repeated = TrajectoryDataset(cfg, "Single-view", config.experiment.seed).prepare()
    assert repeated.assignments == selected.assignments


def test_split_objects_uses_supplied_paths_without_changing_their_order(tmp_path, config):
    paths = [write_trajectory(tmp_path, "Single-view", f"Object{index}Factory") for index in range(10)]
    original_paths = paths.copy()
    cfg = config.data.model_copy(update={"validation_fraction": .2, "test_fraction": .2})
    dataset = TrajectoryDataset(cfg, "Single-view", config.experiment.seed)
    assignments = dataset._split_objects(paths)
    assert paths == original_paths
    assert set(assignments) == {f"Object{index}Factory" for index in range(10)}
    assert {name: list(assignments.values()).count(name) for name in ("train", "validation", "test")} == {
        "train": 6, "validation": 2, "test": 2,
    }
    assert dataset._split_objects(paths) == assignments


def test_bad_image_alignment_is_rejected(tmp_path, config):
    path = write_trajectory(tmp_path)
    row = json.loads(path.read_text())
    row["images"].pop()
    path.write_text(json.dumps(row) + "\n")
    cfg = config.data.model_copy(update={"runs_root": tmp_path})
    data = TrajectoryDataset(cfg, "Single-view", config.experiment.seed).prepare()
    assert not data.train and "placeholders" in data.rejected[0]["reason"]
