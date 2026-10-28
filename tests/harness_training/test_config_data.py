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
    ("distributed",),
    ("tracking",), ("tracking", "settings"),
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


def test_initialization_path_resolves_from_yaml(config, tmp_path):
    config.model.init_from = Path("previous/checkpoints/checkpoint-40")
    path = tmp_path / "config.yaml"
    save_config(config, path)
    assert load_config(path).model.init_from == tmp_path / "previous/checkpoints/checkpoint-40"


def test_initialization_requires_lora(config):
    config.model.init_from = Path("checkpoint")
    config.model.tuning = "full"
    with pytest.raises(ValidationError, match="requires tuning: lora"):
        TrainingConfig.model_validate(config.model_dump())


def test_initialization_rejects_missing_source_before_processor(config, tmp_path):
    config.model.init_from = tmp_path / "missing"
    config.method.use_cpu = True
    trainer = Trainer(config)
    with pytest.raises(ValueError, match="existing checkpoint"):
        trainer.run()
    assert isinstance(trainer.model, QwenModel)
    assert trainer.model.model is None
    assert trainer.model._processor is None


def test_split_rejects_invalid_ratios(config):
    with pytest.raises(ValidationError, match="must be less than 1"):
        DataConfig.model_validate({**config.data.model_dump(), "validation_fraction": .6, "test_fraction": .4})


def test_mix_files_split_by_metadata_object_and_resolve_images(config, tmp_path):
    from PIL import Image

    image_root = tmp_path / "project"
    (image_root / "renders").mkdir(parents=True)
    Image.new("RGB", (32, 32), "green").save(image_root / "renders/first.png")
    Image.new("RGB", (32, 32), "red").save(image_root / "renders/second.png")
    original = json.loads(write_trajectory(tmp_path / "original", "Multi-view").read_text())
    paths = [tmp_path / "source_a.jsonl", tmp_path / "source_b.jsonl"]
    for source_index, path in enumerate(paths):
        rows = []
        for index in range(10):
            row = json.loads(json.dumps(original))
            row["metadata"]["instance"] = f"dataset/object_{index}"
            row["metadata"]["session_id"] = f"source_{source_index}"
            images = [{"type": "image_url", "image_url": {"url": name}} for name in
                      ("renders/second.png", "renders/first.png", "renders/second.png")]
            row["messages"][1]["content"] = images
            row["messages"][3]["content"] = images
            rows.append(json.dumps(row))
        path.write_text("\n".join(rows) + "\n")
    config.data.files = paths
    config.data.image_root = image_root
    config.data.validation_fraction = .2
    config.data.test_fraction = .2
    data = TrajectoryDataset(config.data, "Multi-view", config.experiment.seed).prepare()
    assert data.report()["samples"] == {"train": 24, "validation": 8, "test": 8}
    assert not data.rejected
    assert set(data.source_hashes) == {str(path) for path in paths}
    for split in ("train", "validation", "test"):
        for example in data.examples(split):
            assert data.assignments[example.record.instance] == split
            assert example.record.images_by_message[1] == (
                image_root / "renders/second.png", image_root / "renders/first.png",
                image_root / "renders/second.png")
    first_assignments = data.assignments
    config.data.files = list(reversed(paths))
    assert TrajectoryDataset(config.data, "Multi-view", config.experiment.seed).prepare().assignments == first_assignments


def test_blender_config_sources_paths_and_training_settings(tmp_path):
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/sft_blender_9b_fsdp.yaml")
    baseline = load_config(root / "configs/sft_multi_view_9b_fsdp.yaml")
    assert config.data.image_root == root
    assert len(config.data.files) == 2
    assert all(path.is_absolute() for path in config.data.files)
    assert config.experiment.scene == "Multi-view"
    assert config.model == baseline.model
    assert config.template == baseline.template
    assert config.method == baseline.method.model_copy(update={
        "save_steps": 20,
        "backend": baseline.method.backend.model_copy(update={"run_name": config.experiment.name})})
    assert config.method.resume_from is None
    saved = tmp_path / "blender.yaml"
    save_config(config, saved)
    assert load_config(saved) == config


@pytest.mark.parametrize("url", ["../renders/image.png", "/renders/image.png", "renders/../image.png"])
def test_explicit_image_root_rejects_absolute_paths_and_traversal(tmp_path, config, url):
    path = write_trajectory(tmp_path / "runs")
    row = json.loads(path.read_text())
    row["messages"][1]["content"][1]["image_url"]["url"] = url
    path.write_text(json.dumps(row) + "\n")
    config.data.files = [path]
    config.data.image_root = tmp_path
    data = TrajectoryDataset(config.data, "Single-view", config.experiment.seed).prepare()
    assert not data.train
    assert "image URL" in data.rejected[0]["reason"]


def test_step_selection_uses_explicit_indices(tmp_path, config):
    write_trajectory(tmp_path, failed_first=True)
    cfg = config.data.model_copy(update={"runs_root": tmp_path})
    data = TrajectoryDataset(cfg, "Single-view", config.experiment.seed).prepare()
    assert [example.target_index for example in data.train] == [4]
    assert data.train[0].indices == (0, 3, 4)
    assert data.train[0].record.images_by_message[3][0].name == "reference.png"


def test_missing_image_only_excludes_affected_step(tmp_path, config):
    write_trajectory(tmp_path, missing=True)
    cfg = config.data.model_copy(update={"runs_root": tmp_path})
    data = TrajectoryDataset(cfg, "Single-view", config.experiment.seed).prepare()
    assert [example.target_index for example in data.train] == [4]
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


@pytest.mark.parametrize("scenes", [
    ["Single-view", "Multi-view"], ["ActiveVisual", "Full3DInteraction"],
])
@pytest.mark.parametrize("explicit_files", [False, True])
def test_mix_scenes_share_object_splits_and_keep_image_bindings(tmp_path, config, scenes, explicit_files):
    from test_templates import interactive_record

    paths = []
    for scene, count in zip(scenes, [10, 15], strict=True):
        for index in range(count):
            instance = f"Object{index}Factory"
            if scene in ("Single-view", "Multi-view"):
                path = write_trajectory(tmp_path, scene, instance)
            else:
                source = interactive_record(tmp_path / f"source-{scene}-{index}", instance=instance)
                row = json.loads(source.read_text())
                row["metadata"]["scene"] = scene
                source.write_text(json.dumps(row) + "\n")
                path = tmp_path / scene / "gpt-6-astra" / instance / "post_train_trajectory/trajectory.jsonl"
                path.parent.parent.mkdir(parents=True)
                source.parent.rename(path.parent)
            paths.append(path)
    config.data.runs_root = tmp_path
    config.data.validation_fraction = .2
    config.data.test_fraction = .2
    config.data.files = paths if explicit_files else None
    data = TrajectoryDataset(config.data, scenes, config.experiment.seed).prepare()
    assert not data.rejected
    assert len(data.assignments) == 15
    assert len(data.source_hashes) == 25
    examples = [example for split in ("train", "validation", "test") for example in data.examples(split)]
    assert len(examples) == (50 if scenes[0] == "Single-view" else 75)
    assert len({example.id for example in examples}) == len(examples)
    assert {name: list(data.assignments.values()).count(name) for name in ("train", "validation", "test")} == {
        "train": 9, "validation": 3, "test": 3,
    }
    for split in ("train", "validation", "test"):
        for example in data.examples(split):
            assert data.assignments[example.record.instance] == split
            if scenes[0] == "Single-view":
                assert example.record.images_by_message[1] == (example.record.path.parent / "images/reference.png",)
                assert example.target_index in (2, 4)
            else:
                image = example.record.path.parent / "images/shot.png"
                assert example.record.images_by_message[4] == (image, image)
                assert example.target_index in (3, 5, 7)
    reversed_data = TrajectoryDataset(config.data, list(reversed(scenes)), config.experiment.seed).prepare()
    assert reversed_data.assignments == data.assignments


def test_mix_rejects_scene_metadata_in_wrong_directory(tmp_path, config):
    path = write_trajectory(tmp_path)
    write_trajectory(tmp_path, "Multi-view")
    row = json.loads(path.read_text())
    row["metadata"]["scene"] = "Multi-view"
    path.write_text(json.dumps(row) + "\n")
    config.data.runs_root = tmp_path
    data = TrajectoryDataset(config.data, ["Single-view", "Multi-view"], config.experiment.seed).prepare()
    assert len(data.train) == 2
    assert "Scene metadata does not match input path" in data.rejected[0]["reason"]


@pytest.mark.parametrize("name, scenes", [
    ("view", ["Single-view", "Multi-view"]),
    ("interaction", ["ActiveVisual", "Full3DInteraction"]),
])
def test_mix_configs_and_roundtrip(tmp_path, name, scenes):
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / f"configs/sft_{name}_mix_9b_fsdp.yaml")
    baseline = load_config(root / "configs/sft_single_view_9b_fsdp.yaml")
    assert config.experiment.scene == scenes
    assert config.experiment.output_dir != baseline.experiment.output_dir
    assert config.model == baseline.model
    assert config.distributed == baseline.distributed
    assert config.method.epochs == 1
    assert config.method.resume_from is None
    snapshot = tmp_path / "mixed.yaml"
    save_config(config, snapshot)
    assert load_config(snapshot) == config


@pytest.mark.parametrize("scenes", [[], ["Single-view", "Single-view"], ["invalid"]])
def test_invalid_scene_lists_rejected(config, scenes):
    payload = config.model_dump()
    payload["experiment"]["scene"] = scenes
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(payload)


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
    row["messages"][1]["content"][1]["image_url"]["url"] = "../outside.png"
    path.write_text(json.dumps(row) + "\n")
    cfg = config.data.model_copy(update={"runs_root": tmp_path})
    data = TrajectoryDataset(cfg, "Single-view", config.experiment.seed).prepare()
    assert not data.train and "image URL" in data.rejected[0]["reason"]


def test_sessions_share_object_split_and_have_unique_sample_ids(tmp_path, config):
    path = write_trajectory(tmp_path)
    first = json.loads(path.read_text())
    second = json.loads(path.read_text())
    second["metadata"]["session_id"] = "retry"
    path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n" + json.dumps(first) + "\n")
    cfg = config.data.model_copy(update={"runs_root": tmp_path})
    data = TrajectoryDataset(cfg, "Single-view", config.experiment.seed).prepare()
    assert len(data.train) == 4
    assert len({e.id for e in data.train}) == 4
    assert len(data.assignments) == 1
    assert len(data.rejected) == 1 and "Duplicate session" in data.rejected[0]["reason"]


def test_old_sharegpt_is_rejected(tmp_path, config):
    path = write_trajectory(tmp_path)
    path.write_text(json.dumps({"conversations": [], "metadata": {}}))
    cfg = config.data.model_copy(update={"runs_root": tmp_path})
    data = TrajectoryDataset(cfg, "Single-view", config.experiment.seed).prepare()
    assert not data.train
    assert "schema_version" in data.rejected[0]["reason"]
