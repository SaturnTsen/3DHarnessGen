import json

import pytest
from pydantic import ValidationError

from harness_training.config import DataConfig, load_config, save_config
from harness_training.data.trajectory import TrajectoryDataset
from harness_training.data.validation import DataError
from conftest import write_trajectory


def test_config_roundtrip_paths_and_unknown_keys(config, tmp_path):
    path = tmp_path / "config.yaml"
    save_config(config, path)
    restored = load_config(path)
    assert restored == config
    path.write_text(path.read_text() + "typo_option: true\n")
    with pytest.raises(ValidationError):
        load_config(path)


def test_split_rejects_invalid_ratios():
    with pytest.raises(ValidationError):
        DataConfig(runs_root=".", validation_fraction=.6, test_fraction=.4)


def test_step_selection_uses_source_mapping(tmp_path):
    write_trajectory(tmp_path, failed_first=True)
    data = TrajectoryDataset(DataConfig(runs_root=tmp_path, validation_fraction=0, test_fraction=0), "Single-view").prepare()
    assert [example.target_index for example in data.train] == [3]
    assert data.train[0].indices == (2, 3)
    assert data.train[0].record.images_by_conversation[2][0].name == "reference.png"


def test_missing_image_only_excludes_affected_step(tmp_path):
    write_trajectory(tmp_path, missing=True)
    cfg = DataConfig(runs_root=tmp_path, validation_fraction=0, test_fraction=0)
    data = TrajectoryDataset(cfg, "Single-view").prepare()
    assert [example.target_index for example in data.train] == [3]
    assert len(data.rejected) == 1 and "absent.png" in data.rejected[0]["reason"]
    cfg.missing_images = "error"
    with pytest.raises(DataError, match="absent.png"):
        TrajectoryDataset(cfg, "Single-view").prepare()


def test_split_stays_same_across_scenes(tmp_path):
    for scene in ["Single-view", "Multi-view"]:
        for i in range(20):
            write_trajectory(tmp_path, scene, f"Object{i}Factory")
    cfg = DataConfig(runs_root=tmp_path)
    a = TrajectoryDataset(cfg, "Single-view").prepare()
    b = TrajectoryDataset(cfg, "Multi-view").prepare()
    assert a.assignments == b.assignments
    for split in [a.train, a.validation, a.test]:
        for example in split:
            assert a.assignments[example.record.instance] == next(name for name in ["train", "validation", "test"]
                                                                 if split is getattr(a, name))


def test_bad_image_alignment_is_rejected(tmp_path):
    path = write_trajectory(tmp_path)
    row = json.loads(path.read_text())
    row["images"].pop()
    path.write_text(json.dumps(row) + "\n")
    data = TrajectoryDataset(DataConfig(runs_root=tmp_path), "Single-view").prepare()
    assert not data.train and "placeholders" in data.rejected[0]["reason"]
