"""Real libuv TCPStore transfers above its single-message size limit."""

import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

from harness_training.config import load_config, save_config
from harness_training.data.trajectory import TrajectoryDataset
from harness_training.training.preflight import PreparationProgress, PreflightWorkers
from conftest import write_trajectory


ROOT = Path(__file__).resolve().parents[2]


def save_result(directory, rank, value):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"rank-{rank}.json").write_text(json.dumps(value))


def transport_worker(directory):
    import torch

    workers = PreflightWorkers()
    # Empty, exact chunk boundary, above 8 MiB, and the production failure size.
    sizes = [0, 4 * 1024**2, 8 * 1024**2 + 1, 154546773]
    data = bytes([workers.rank]) * sizes[workers.rank]
    value = {"rank": workers.rank, "data": data, "alias": data}
    encoded_sizes = workers.gather("encoded_sizes", len(pickle.dumps(value, protocol=5)))
    results = workers.gather("large", value)
    summaries = []
    for rank, result in enumerate(results):
        assert result["rank"] == rank
        assert len(result["data"]) == sizes[rank]
        assert result["alias"] is result["data"]
        summaries.append(hashlib.sha256(result["data"]).hexdigest())
    workers.gather("large_read_complete", workers.rank)
    for rank, size in enumerate(encoded_sizes):
        assert not workers.store.check([f"large/{rank}/size"])
        for index, _ in enumerate(range(0, size, workers.chunk_bytes)):
            assert not workers.store.check([f"large/{rank}/chunk/{index}"])
    assert workers.gather("still_alive", workers.rank) == list(range(workers.world_size))
    save_result(directory, workers.rank, {"sizes": sizes, "encoded_sizes": encoded_sizes, "hashes": summaries,
                                       "cuda_initialized": torch.cuda.is_initialized(),
                                       "distributed_initialized": torch.distributed.is_initialized()})


def records_worker(config_path, directory):
    config = load_config(Path(config_path))
    workers = PreflightWorkers()
    data = TrajectoryDataset(config.data, config.experiment.scene, config.experiment.seed).prepare(workers=workers)
    save_result(directory, workers.rank, describe(data))


def describe(data):
    return {"targets": {split: [{"id": example.id,
                                  "system_hash": hashlib.sha256(example.record.row.messages[0].content.encode()).hexdigest(),
                                  "images": [[str(path) for path in example.record.images_by_message[index]]
                                             for index in example.indices]}
                                 for example in data.examples(split)]
                        for split in ("train", "validation", "test")},
            "assignments": data.assignments, "rejected": data.rejected, "sources": data.source_hashes}


def launch(tmp_path, mode, *arguments):
    directory = tmp_path / mode
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=4",
               str(Path(__file__)), mode, *map(str, arguments), str(directory)]
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT)}
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120)
    (tmp_path / f"{mode}.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, (result.stdout + result.stderr)[-10000:]
    assert "Invalid payload size" not in result.stderr and "Broken pipe" not in result.stderr
    return [json.loads((directory / f"rank-{rank}.json").read_text()) for rank in range(4)]


def test_four_process_transfer_exceeding_154_mb_preserves_content_and_releases_chunks(tmp_path):
    ranks = launch(tmp_path, "transport")
    reference = ranks[0]
    assert reference["encoded_sizes"][3] > 154546773
    assert reference["hashes"] == [hashlib.sha256(bytes([rank]) * size).hexdigest()
                                   for rank, size in enumerate(reference["sizes"])]
    assert all(rank == reference for rank in ranks)
    assert not reference["cuda_initialized"] and not reference["distributed_initialized"]


def test_real_large_trajectory_records_preserve_splits_order_and_images(config, tmp_path):
    # Keep every rank's serialized record shard above the original 8 MiB limit.
    config.data.validation_fraction = config.data.test_fraction = .25
    for index in range(3):
        write_trajectory(config.data.runs_root, instance=f"Object{index}Factory")
    for index, path in enumerate(sorted(config.data.runs_root.rglob("trajectory.jsonl"))):
        row = json.loads(path.read_text())
        row["messages"][0]["content"] = chr(ord("A") + index) * (9 * 1024**2)
        path.write_text(json.dumps(row) + "\n")
    data = TrajectoryDataset(config.data, config.experiment.scene, config.experiment.seed).prepare()
    expected = describe(data)
    path = tmp_path / "records.yaml"
    save_config(config, path)
    ranks = launch(tmp_path, "records", path)
    assert all(rank == expected for rank in ranks)
    assert all(expected["targets"].values())


class PublishFailsAfterStart(PreparationProgress):
    def __init__(self, workers):
        super().__init__(workers, "失败清理测试", 1)
        self.publish_calls = 0

    def publish(self):
        self.publish_calls += 1
        if self.publish_calls > 1:
            raise RuntimeError("transport is already closed")
        super().publish()


def test_failed_transfer_keeps_original_error_and_stops_progress_thread():
    progress = PublishFailsAfterStart(PreflightWorkers())
    with pytest.raises(ValueError, match="original transfer failure"):
        with progress:
            raise ValueError("original transfer failure")
    assert progress.publish_calls == 1
    assert progress.stopped.is_set()
    assert progress.thread is not None and not progress.thread.is_alive()


if __name__ == "__main__":
    if sys.argv[1] == "transport":
        transport_worker(sys.argv[2])
    else:
        records_worker(sys.argv[2], sys.argv[3])
