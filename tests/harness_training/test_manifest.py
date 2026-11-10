"""Provenance keeps its established fields and source fingerprint rules."""

import hashlib
import json
from pathlib import Path

from harness_training.utils.manifest import build_manifest, hash_file, hash_source, model_sources
from harness_training.trainers.sft_trainer import SFTTrainer
from harness_training.types import Manifest


def test_manifest_records_inputs_and_preserves_fingerprint(config):
    data = SFTTrainer(config).prepare()
    manifest = build_manifest(config, data)
    assert set(manifest) == set(Manifest.__required_keys__)
    payload = {key: value for key, value in manifest.items() if key != "fingerprint"}
    assert manifest["fingerprint"] == hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    assert manifest["config"] == config.model_dump(mode="json")
    assert manifest["targets"]["train"] == [example.id for example in data.train]
    image = data.train[0].record.images_by_message[1][0]
    assert manifest["sources"][str(image)] == hash_file(image)
    package = Path(__file__).resolve().parents[2] / "harness_training"
    for name in (
        "utils/manifest.py", "utils/serving.py", "data/trajectory.py", "chat_templates/qwen.py",
        "models/qwen/qwen_model.py", "models/qwen/qwen_long_context.py",
        "dataloader/sft_dataloader.py", "trainers/sft_trainer.py", "trainers/long_context_sft_trainer.py",
    ):
        assert manifest["code"][name] == hash_file(package / name)


def test_fsdp_sources_include_snapshot_metadata_and_shards(config, tmp_path):
    source = tmp_path / "prior/checkpoints/checkpoint-1"
    shards = source / "pytorch_model_fsdp_0"
    shards.mkdir(parents=True)
    snapshot = source.parent.parent / "resolved_config.yaml"
    snapshot.write_text("model: {}\n")
    metadata = shards / ".metadata"
    metadata.write_bytes(b"metadata")
    shard = shards / "__0_0.distcp"
    shard.write_bytes(b"tensors")
    config.model.init_from = source
    paths = model_sources(config)
    assert {snapshot, metadata, shard} <= set(paths)
    assert hash_source(metadata) == hash_file(metadata)
    stat = shard.stat()
    assert hash_source(shard) == hashlib.sha256(f"{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()
