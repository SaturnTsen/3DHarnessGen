"""Training's early resume checks share file hashing without loading model tensors."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from harness_training.config import load_config, save_config
from harness_training.training.preflight import PreflightWorkers
from harness_training.training.resume import check_resume
from test_resume_precheck import elastic_qwen_path, saved_resume


ROOT = Path(__file__).resolve().parents[2]


class RecordingWorkers(PreflightWorkers):
    def __init__(self):
        super().__init__(label="resume")
        self.checked = []

    def gather(self, phase, value):
        if phase == "resume_sources":
            self.checked = list(value[0]) + list(value[1])
        return super().gather(phase, value)


def worker(config_path, report_dir):
    import torch

    config = load_config(Path(config_path))
    workers = RecordingWorkers()
    try:
        _, progress, report = check_resume(config, workers.world_size, workers=workers)
    except ValueError as exc:
        result = {"error": str(exc)}
    else:
        result = {"report": report, "step": progress.global_step}
    result.update({"checked": workers.checked, "cuda_initialized": torch.cuda.is_initialized(),
                   "distributed_initialized": torch.distributed.is_initialized()})
    path = Path(report_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"rank-{workers.rank}.json").write_text(json.dumps(result))


def launch(config, tmp_path, name="parallel", *, cli=False):
    config.distributed.strategy = "ddp"
    config.method.gradient_accumulation_steps = 2
    path = tmp_path / f"{name}.yaml"
    save_config(config, path)
    reports = tmp_path / name
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=4"]
    command += (["-m", "scripts.post_train", "train", "--config", str(path)] if cli else
                [str(Path(__file__)), str(path), str(reports)])
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT)}
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60)
    (tmp_path / f"{name}.log").write_text(result.stdout + result.stderr)
    if cli:
        return result
    assert result.returncode == 0, result.stdout + result.stderr
    return [json.loads((reports / f"rank-{rank}.json").read_text()) for rank in range(4)], result.stdout


def test_parallel_resume_hashes_each_source_once_and_preserves_cursor(saved_resume, tmp_path):
    config, prior = saved_resume
    _, _, serial = check_resume(config, 1)
    ranks, log = launch(config, tmp_path)
    checked = [path for rank in ranks for path in rank["checked"]]
    assert len(checked) == len(set(checked)) == len(prior["sources"])
    assert set(checked) == set(prior["sources"])
    for rank in ranks:
        assert rank["report"]["status"] == "passed" and rank["report"]["stage"] == "complete"
        assert rank["report"]["data_position"] == serial["data_position"]
        assert rank["step"] == 1
        assert not rank["report"]["processor_loaded"] and not rank["report"]["weights_loaded"]
        assert not rank["cuda_initialized"] and not rank["distributed_initialized"]
    assert "[resume] 核对恢复源文件摘要：0/" in log
    assert "(100.0%)" in log and "恢复检查通过" in log
    assert "[preflight]" not in log


@pytest.mark.parametrize("change", ["image", "missing", "batch"])
def test_parallel_failures_reach_all_ranks_and_write_root_report(saved_resume, tmp_path, change):
    config, _ = saved_resume
    original = (config.experiment.output_dir / "manifest.json").read_bytes()
    if change == "batch":
        config.method.batch_size = 2
        expected = "Keep global batch size"
    else:
        path = next(config.data.runs_root.rglob("reference.png"))
        expected = str(path)
        if change == "missing":
            path.unlink()
        else:
            path.write_bytes(path.read_bytes() + b"changed")
    ranks, _ = launch(config, tmp_path)
    assert all(expected in rank["error"] for rank in ranks)
    if change == "batch":
        assert all(not rank["checked"] for rank in ranks)
    else:
        checked = [path for rank in ranks for path in rank["checked"]]
        assert len(checked) == len(set(checked))
    report = json.loads((ROOT / "temp/resume_check.json").read_text())
    assert report["status"] == "failed"
    assert report["stage"] == ("training_state" if change == "batch" else "source_files")
    assert (config.experiment.output_dir / "manifest.json").read_bytes() == original


def test_read_error_on_one_rank_is_reported_on_every_rank(saved_resume, tmp_path):
    if os.geteuid() == 0:
        pytest.skip("Root bypasses file read permissions")
    config, _ = saved_resume
    path = next(config.data.runs_root.rglob("reference.png"))
    path.chmod(0)
    try:
        ranks, _ = launch(config, tmp_path)
    finally:
        path.chmod(0o644)
    assert all("Resume source check failed" in rank["error"] and str(path) in rank["error"] for rank in ranks)
    report = json.loads((ROOT / "temp/resume_check.json").read_text())
    assert report["status"] == "failed" and "Permission denied" in report["error"]


def test_actual_ddp_train_rejects_changed_source_before_preflight(saved_resume, tmp_path):
    config, _ = saved_resume
    path = next(config.data.runs_root.rglob("reference.png"))
    path.write_bytes(path.read_bytes() + b"changed")
    result = launch(config, tmp_path, cli=True)
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "[resume] 核对恢复源文件摘要" in output
    assert str(path) in output
    assert "[preflight]" not in output and "Loading weights" not in output


def test_fsdp_to_ddp_training_uses_parallel_resume_checks(elastic_qwen_path, tmp_path):
    import torch
    from harness_training.config import LongContextConfig
    from test_elastic_resume import run_training, setup_config

    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA GPUs")
    config = setup_config(tmp_path, elastic_qwen_path, "fsdp", "4bit")
    config.method.long_context = LongContextConfig(
        require_optimized_kernels=False, activation_dtype="bfloat16", checkpoint_group_size=2,
        mlp_chunk_size=16, vision_batch_size=1, loss_chunk_size=8,
    )
    config.method.backend.loss_type = "nll"
    config.method.resume_policy = "elastic"
    config.method.gradient_accumulation_steps = 2
    run_training(config, tmp_path / "fsdp.yaml", 2)
    output = config.experiment.output_dir
    identity = json.loads((output / "wandb_run.json").read_text())["id"]
    config.distributed.strategy = "ddp"
    config.method.backend.fsdp = None
    config.method.backend.fsdp_config = None
    config.method.resume_from = output / "checkpoints/checkpoint-1"
    config.method.max_steps = 2
    log = run_training(config, tmp_path / "ddp.yaml", 2)
    assert "[resume] 核对恢复源文件摘要" in log and "恢复检查通过" in log
    assert "下一样本位置 4" in log
    checkpoint = output / "checkpoints/checkpoint-2"
    progress = json.loads((checkpoint / "data_progress.json").read_text())
    assert progress["global_step"] == 2 and progress["next_sample"] == 8
    state = torch.load(checkpoint / "portable_state.pt", map_location="cpu", weights_only=True)
    assert all(int(value["step"]) == 2 for value in state["optimizer"]["state"].values())
    assert json.loads((output / "wandb_run.json").read_text())["id"] == identity


if __name__ == "__main__":
    worker(*sys.argv[1:])
