"""Exercise training's real preflight with torchrun, without loading model weights."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from harness_training.config import load_config, save_config
from harness_training.data.trajectory import DataError, TrajectoryDataset
from harness_training.templates.qwen import QwenTemplate
from harness_training.training.preflight import PreparationProgress, PreflightWorkers
from harness_training.training.trainer import Trainer
from conftest import write_trajectory


ROOT = Path(__file__).resolve().parents[2]


class RecordingTemplate(QwenTemplate):
    def __init__(self, config, processor):
        super().__init__(config, processor)
        self.calls = []
        self.encoded = {}

    def encode(self, example):
        self.calls.append(example.id)
        result = super().encode(example)
        self.encoded[example.id] = {
            name: hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
            for name, tensor in result.items()
        }
        return result


class RecordingTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.dataset = RecordingDataset(config.data, config.experiment.scene, config.experiment.seed)

    @property
    def template(self):
        if self._template is None:
            self._template = RecordingTemplate(self.config.template, self.model.processor)
        return self._template


class RecordingDataset(TrajectoryDataset):
    def __init__(self, config, scene, seed):
        super().__init__(config, scene, seed)
        self.read_paths = []
        self.checked_images = []

    def _read_path(self, index, path):
        self.read_paths.append(str(path))
        return super()._read_path(index, path)

    def _check_image(self, path):
        self.checked_images.append(str(path))
        return super()._check_image(path)


def worker(config_path, report_dir):
    config = load_config(Path(config_path))
    trainer = RecordingTrainer(config)
    try:
        bundle = trainer.prepare(encode=True)
    except DataError as exc:
        report = {"error": str(exc), "read_paths": trainer.dataset.read_paths,
                  "checked_images": trainer.dataset.checked_images}
        write_report(report_dir, report)
        return
    read_paths = list(trainer.dataset.read_paths)
    checked_images = list(trainer.dataset.checked_images)
    calls = list(trainer._template.calls) if trainer._template is not None else []
    # Repeated preparation in the same launch must not read stale collective keys
    # or encode twice; it also exercises workers with no assigned examples.
    repeated = trainer.prepare(encode=True)
    assert [example.id for example in repeated.train] == [example.id for example in bundle.train]
    assert (trainer._template.calls if trainer._template is not None else []) == calls
    report = {"targets": {split: [example.id for example in bundle.examples(split)]
                          for split in ("train", "validation", "test")},
              "rejected": bundle.rejected, "manifest": trainer._manifest(bundle),
              "read_paths": read_paths, "checked_images": checked_images,
              "calls": trainer._template.calls if trainer._template is not None else [],
              "encoded": trainer._template.encoded if trainer._template is not None else {},
              "processor_loaded": trainer.model._processor is not None,
              "weights_loaded": trainer.model.model is not None}
    write_report(report_dir, report)


def write_report(report_dir, report):
    path = Path(report_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"rank-{os.environ.get('RANK', '0')}.json").write_text(json.dumps(report))


def launch(config, tmp_path, name, world):
    config_path = tmp_path / f"{name}.yaml"
    save_config(config, config_path)
    report_dir = tmp_path / name
    command = [sys.executable]
    if world > 1:
        command += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc-per-node={world}"]
    command += [str(Path(__file__)), str(config_path), str(report_dir)]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    (tmp_path / f"{name}.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return [json.loads((report_dir / f"rank-{rank}.json").read_text()) for rank in range(world)], result.stdout


def fixture_config(config, processor, tmp_path):
    config.model.path = tmp_path / "processor"
    processor.save_pretrained(config.model.path)
    config.data.validation_fraction = .2
    config.data.test_fraction = .2
    for index in range(4):
        write_trajectory(config.data.runs_root, instance=f"Object{index}Factory")
    # Include a real length rejection on a worker other than rank zero.
    source = next(config.data.runs_root.rglob("trajectory.jsonl"))
    row = json.loads(source.read_text())
    row["messages"][4]["content"] = "print('long')\n" * 1000
    source.write_text(json.dumps(row) + "\n")
    config.template.max_length = 512
    return config


def test_sharded_checks_match_single_rank_including_pixels_labels_and_rejections(config, processor, tmp_path):
    config = fixture_config(config, processor, tmp_path)
    config.experiment.output_dir = tmp_path / "single-output"
    single, _ = launch(config, tmp_path, "single", 1)
    config.experiment.output_dir = tmp_path / "four-output"
    distributed, log = launch(config, tmp_path, "four", 4)
    reference = single[0]
    assert reference["rejected"] and all(reference["targets"].values())
    calls = [call for rank in distributed for call in rank["calls"]]
    assert len(calls) == len(set(calls)) == len(reference["calls"])
    assert set(calls) == set(reference["calls"])
    files_read = [path for rank in distributed for path in rank["read_paths"]]
    images_checked = [path for rank in distributed for path in rank["checked_images"]]
    assert len(files_read) == len(set(files_read)) == 5
    assert set(files_read) == set(reference["read_paths"])
    assert len(images_checked) == len(set(images_checked)) == 5
    assert set(images_checked) == set(reference["checked_images"])
    rejected_ids = {rejection["id"] for rejection in reference["rejected"]}
    assert any(rejected_ids.intersection(rank["calls"]) for rank in distributed[1:])
    encoded = {key: value for rank in distributed for key, value in rank["encoded"].items()}
    assert encoded == reference["encoded"]
    for rank in distributed:
        assert rank["targets"] == reference["targets"]
        assert rank["rejected"] == reference["rejected"]
        assert rank["manifest"]["sources"] == reference["manifest"]["sources"]
        assert not rank["weights_loaded"]
    assert "4 个进程分摊检查" in log
    assert "读取和校验轨迹文件：5/5 (100.0%)" in log
    assert "检查去重后的图片：5/5 (100.0%)" in log
    # A different launch geometry reuses the same validation, without a Processor.
    config.method.batch_size = 2
    config.method.gradient_accumulation_steps = 2
    warm, log = launch(config, tmp_path, "two-warm", 2)
    assert "命中预检查缓存" in log
    assert all(not rank["calls"] and not rank["processor_loaded"] and not rank["weights_loaded"] for rank in warm)
    assert all(rank["targets"] == reference["targets"] and rank["rejected"] == reference["rejected"] for rank in warm)


def test_more_workers_than_samples(config, processor, tmp_path):
    config.model.path = tmp_path / "processor"
    processor.save_pretrained(config.model.path)
    ranks, _ = launch(config, tmp_path, "sparse-four", 4)
    assert sorted(len(rank["calls"]) for rank in ranks) == [0, 0, 1, 1]
    assert sum(len(rank["read_paths"]) for rank in ranks) == 1
    assert sum(len(rank["checked_images"]) for rank in ranks) == 1
    assert all(rank["targets"] == ranks[0]["targets"] and not rank["rejected"] for rank in ranks)


@pytest.mark.parametrize("explicit", [False, True])
def test_sharded_data_preserves_bad_records_missing_and_corrupt_images(config, processor, tmp_path, explicit):
    config.model.path = tmp_path / "processor"
    processor.save_pretrained(config.model.path)
    sources = [write_trajectory(config.data.runs_root, instance=f"Object{index}Factory") for index in range(4)]
    # One missing reference, one corrupt image, and a duplicate/invalid row.
    (sources[0].parent / "images/reference.png").unlink()
    (sources[1].parent / "images/reference.png").write_bytes(b"corrupt-image")
    sources[2].write_text(sources[2].read_text() * 2 + "{invalid json}\n")
    if explicit:
        config.data.files = list(reversed(sources))
        # The same session appears in files assigned to different ranks.
        sources[3].write_text(sources[3].read_text() + sources[2].read_text().splitlines()[0] + "\n")
    config.experiment.output_dir = tmp_path / "serial-output"
    serial, _ = launch(config, tmp_path, "bad-serial", 1)
    config.experiment.output_dir = tmp_path / "distributed-output"
    ranks, _ = launch(config, tmp_path, "bad-four", 4)
    assert serial[0]["rejected"]
    assert any("Duplicate session_id" in item["reason"] for item in serial[0]["rejected"])
    assert any("missing image" in item["reason"] for item in serial[0]["rejected"])
    assert any("unreadable image" in item["reason"] for item in serial[0]["rejected"])
    assert all(rank["targets"] == serial[0]["targets"] and rank["rejected"] == serial[0]["rejected"] for rank in ranks)
    assert all(rank["manifest"]["sources"] == serial[0]["manifest"]["sources"] for rank in ranks)
    config.data.missing_images = "error"
    errors, _ = launch(config, tmp_path, "bad-error-four", 4)
    assert all(rank["error"] == errors[0]["error"] for rank in errors)
    assert "missing image" in errors[0]["error"] or "unreadable image" in errors[0]["error"]


def test_repeated_images_across_files_are_checked_once(config, processor, tmp_path):
    config.model.path = tmp_path / "processor"
    processor.save_pretrained(config.model.path)
    paths = [write_trajectory(config.data.runs_root, instance=f"Object{index}Factory") for index in range(4)]
    config.data.files = paths
    config.data.image_root = paths[0].parent
    ranks, _ = launch(config, tmp_path, "shared-image-four", 4)
    assert sum(len(rank["read_paths"]) for rank in ranks) == 4
    assert sum(len(rank["checked_images"]) for rank in ranks) == 1


class FrequentProgress(PreparationProgress):
    def monitor(self):
        while not self.stopped.wait(.02):
            self.log()


def progress_worker(report_dir):
    workers = PreflightWorkers()
    workers.gather("progress_start", workers.rank)
    with FrequentProgress(workers, "测试等待其他进程", workers.world_size) as advance:
        if workers.rank == 1:
            time.sleep(.3)
        advance()
        workers.gather("finished", workers.rank)
    write_report(report_dir, {"done": True})


def test_global_progress_continues_while_rank_zero_waits(tmp_path):
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2",
               str(Path(__file__)), "progress", str(tmp_path / "progress")]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "测试等待其他进程：1/2 (50.0%)" in result.stdout
    assert "测试等待其他进程：2/2 (100.0%)" in result.stdout


@pytest.mark.parametrize("change", ["image", "trajectory", "length", "template", "tokenizer"])
def test_cache_invalidated_when_encoding_inputs_change(config, processor, tmp_path, change):
    config = fixture_config(config, processor, tmp_path)
    trainer = RecordingTrainer(config)
    trainer.prepare(encode=True)
    key = json.loads((config.experiment.output_dir / "preflight_cache.json").read_text())["key"]
    if change == "image":
        from PIL import Image

        Image.new("RGB", (32, 32), "blue").save(next(config.data.runs_root.rglob("reference.png")))
    elif change == "trajectory":
        source = next(config.data.runs_root.rglob("trajectory.jsonl"))
        source.write_text(source.read_text().replace("print('first')", "print('updated')"))
    elif change == "length":
        config.template.max_length *= 2
    elif change == "template":
        config.template.enable_thinking = not config.template.enable_thinking
    else:
        path = config.model.path / "tokenizer_config.json"
        path.write_text(path.read_text() + "\n")
    trainer = RecordingTrainer(config)
    trainer.prepare(encode=True)
    assert trainer._template is not None and trainer._template.calls
    assert json.loads((config.experiment.output_dir / "preflight_cache.json").read_text())["key"] != key


if __name__ == "__main__":
    if sys.argv[1] == "progress":
        progress_worker(sys.argv[2])
    else:
        worker(*sys.argv[1:])
