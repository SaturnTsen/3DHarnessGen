"""Share training-time encoding checks and reuse content-verified validation results."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import timedelta
import hashlib
import importlib.metadata
import json
import os
import pickle
from threading import Event, Thread
import time
from types import TracebackType
from typing import Literal

from pydantic import BaseModel

from harness_training.config import TrainingConfig
from harness_training.data.base import DatasetBundle
from harness_training.templates.base import BaseTemplate, SampleTooLong, TemplateError
from harness_training.types import Rejection
from .manifest import backend_versions, code_identity, hash_source, model_sources


def preflight_log(message: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[preflight] {message}", flush=True)


class PreflightWorkers:
    """Exchange validated in-memory results without initializing NCCL.

    The training backend still owns GPU assignment and process-group initialization.
    Each launch/restart gets its own namespace; no shared scratch files or stale ranks.
    """

    def __init__(self, label: Literal["preflight", "resume"] = "preflight") -> None:
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.label = label
        self.timeout = timedelta(hours=2)
        self.store = None
        if self.world_size > 1:
            # These are public PyTorch APIs, omitted from its explicit type exports.
            from torch.distributed import PrefixStore, TCPStore  # pyright: ignore[reportPrivateImportUsage]

            store = TCPStore(os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]),
                             is_master=False, timeout=self.timeout)
            namespace = f"training-preflight/{os.environ['TORCHELASTIC_RUN_ID']}/{os.environ.get('TORCHELASTIC_RESTART_COUNT', '0')}/"
            calls = PrefixStore(namespace, store)
            sequence = calls.add(f"calls/{self.rank}", 1)
            self.store = PrefixStore(f"call-{sequence}/", calls)

    def gather[T](self, phase: str, value: T) -> list[T]:
        if self.store is None:
            return [value]
        # Only objects created by this launch are exchanged. No dataset/checkpoint
        # pickle files are read, and no conversations are written to disk.
        self.store.set(f"{phase}/{self.rank}", pickle.dumps(value, protocol=5))  # pyright: ignore[reportArgumentType]
        keys = [f"{phase}/{rank}" for rank in range(self.world_size)]
        # A blocking Store.get can hold Python's GIL and prevent the progress
        # thread from reporting. Poll availability before reading ready payloads.
        deadline = time.monotonic() + self.timeout.total_seconds()
        while not self.store.check(keys):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Preflight timed out waiting for workers during {phase}")
            time.sleep(.1)
        results = [pickle.loads(self.store.get(key)) for key in keys]
        # The final reader releases the shared payload after every rank has its
        # own records. Keep large trajectories out of the rendezvous store during training.
        if self.store.add(f"{phase}/readers", 1) == self.world_size:
            for key in keys:
                self.store.delete_key(key)
        return results

    def progress(self, phase: str, total: int) -> PreparationProgress:
        return PreparationProgress(self, phase, total)

    def log(self, message: str) -> None:
        if self.rank == 0:
            print(f"[{self.label}] {message}", flush=True)


class PreparationProgress(AbstractContextManager[Callable[[], None]]):
    """Publish global counters even while rank zero waits for a slower worker."""

    def __init__(self, workers: PreflightWorkers, phase: str, total: int):
        self.workers, self.phase, self.total = workers, phase, total
        self.completed = 0
        self.started = time.monotonic()
        self.published = self.started
        self.stopped = Event()
        self.thread: Thread | None = None

    def __enter__(self) -> Callable[[], None]:
        self.publish()
        self.workers.log(f"{self.phase}：0/{self.total}；{self.workers.world_size} 个进程分摊。")
        if self.workers.rank == 0:
            self.thread = Thread(target=self.monitor, daemon=True)
            self.thread.start()
        return self.advance

    def advance(self) -> None:
        self.completed += 1
        assigned = len(range(self.workers.rank, self.total, self.workers.world_size))
        if self.completed == assigned or time.monotonic() - self.published >= 1:
            self.publish()

    def publish(self) -> None:
        if self.workers.store is not None:
            self.workers.store.set(f"progress/{self.phase}/{self.workers.rank}", str(self.completed))
        self.published = time.monotonic()

    def log(self) -> None:
        completed = self.completed
        if self.workers.store is not None:
            keys = [f"progress/{self.phase}/{rank}" for rank in range(self.workers.world_size)]
            completed = sum(int(self.workers.store.get(key)) for key in keys if self.workers.store.check([key]))
        elapsed = time.monotonic() - self.started
        percentage = 100 * completed / self.total if self.total else 100
        self.workers.log(f"{self.phase}：{completed}/{self.total} ({percentage:.1f}%)，耗时 {elapsed:.1f} 秒。")

    def monitor(self) -> None:
        while not self.stopped.wait(20):
            self.log()

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 traceback: TracebackType | None) -> None:
        self.publish()
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
            self.log()


class ValidationCache(BaseModel):
    version: Literal[1]
    key: str
    failures: dict[int, Rejection]


def cache_key(config: TrainingConfig, data: DatasetBundle, sources: dict[str, str]) -> str:
    # Training geometry is deliberately excluded: it cannot change template encoding.
    payload = {"model": config.model.model_dump(mode="json"),
               "template": config.template.model_dump(mode="json"),
               "data": config.data.model_dump(mode="json"),
               "sources": sources, "code": code_identity(), "versions": backend_versions(),
               "processor_versions": {name: importlib.metadata.version(name)
                                      for name in ("pillow", "tokenizers", "numpy", "torchvision")},
               "targets": {split: [example.id for example in data.examples(split)]
                           for split in ("train", "validation", "test")}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_encoding(config: TrainingConfig, data: DatasetBundle,
                      template: Callable[[], BaseTemplate]) -> None:
    workers = PreflightWorkers()
    examples = data.train + data.validation + data.test
    images = {path for example in examples for index in example.indices
              for path in example.record.images_by_message[index]}
    processor_files = {path for path in config.model.path.iterdir()
                       if path.is_file() and path.suffix in {".model", ".txt", ".tiktoken"}}
    paths = sorted(images | set(model_sources(config)) | processor_files)
    preflight_log(f"核对 {len(paths)} 个图片及模型源文件；{workers.world_size} 个进程分摊检查。")
    local_sources = {str(path): hash_source(path) for path in paths[workers.rank::workers.world_size]}
    sources = dict(data.source_hashes)
    for shard in workers.gather("sources", local_sources):
        sources.update(shard)
    key = cache_key(config, data, sources)
    path = config.experiment.output_dir / "preflight_cache.json"
    cached = None
    if workers.rank == 0 and path.is_file():
        saved = ValidationCache.model_validate_json(path.read_text())
        if saved.key == key:
            cached = saved.model_dump(mode="json")
    decisions = workers.gather("cache", {"result": cached})
    cached = decisions[0]["result"]
    if cached is not None:
        failures = ValidationCache.model_validate(cached).failures
        preflight_log(f"命中预检查缓存：复用 {len(examples)} 条样本的编码检查结果，"
                      f"保留 {len(examples) - len(failures)}，拒绝 {len(failures)}；不重复编码。")
    else:
        preflight_log(f"无匹配缓存：分摊编码 train/validation/test 共 {len(examples)} 条；"
                      f"进程 0 负责 {len(examples[::workers.world_size])} 条。")
        failures: dict[int, Rejection] = {}
        last_log = time.monotonic()
        count = 0
        shown_errors = 0
        for index in range(workers.rank, len(examples), workers.world_size):
            example = examples[index]
            try:
                template().encode(example)
            except (TemplateError, OSError, ValueError) as exc:
                failures[index] = {"id": example.id, "path": str(example.record.path),
                                   "reason": str(exc), "stage": "encoding",
                                   "too_long": isinstance(exc, SampleTooLong)}
                if not isinstance(exc, SampleTooLong) and shown_errors < 3:
                    preflight_log(f"编码失败示例 {example.id}：{str(exc)[:500]}")
                    shown_errors += 1
            count += 1
            if time.monotonic() - last_log >= 20:
                preflight_log(f"进程 0 编码 {count}/{len(examples[::workers.world_size])}，"
                              f"本地拒绝 {len(failures)}。")
                last_log = time.monotonic()
        # Serialize validation outcomes only; no conversations, pixels or training tensors.
        shards = workers.gather("encoding", {"failures": failures})
        failures = {int(index): rejection for shard in shards for index, rejection in shard["failures"].items()}
        if workers.rank == 0:
            saved = ValidationCache(version=1, key=key, failures=failures)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(saved.model_dump_json(indent=2) + "\n")
            temporary.replace(path)
        preflight_log(f"全局编码完成：检查 {len(examples)} 条，保留 {len(examples) - len(failures)}，"
                      f"拒绝 {len(failures)}。")
    # Reconstruct the exact original split and order on every rank, including rejections.
    offset = 0
    for split in ("train", "validation", "test"):
        selected = data.examples(split)
        accepted = []
        for index, example in enumerate(selected, offset):
            if index in failures:
                data.rejected.append(failures[index])
            else:
                accepted.append(example)
        offset += len(selected)
        setattr(data, split, accepted)
    # Reuse verified hashes in the final manifest, retaining only accepted images.
    selected_images = {path for example in data.train + data.validation + data.test for index in example.indices
                       for path in example.record.images_by_message[index]}
    for selected_path in selected_images | set(model_sources(config)):
        data.source_hashes[str(selected_path)] = sources[str(selected_path)]
