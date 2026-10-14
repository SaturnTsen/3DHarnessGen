from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness_training.config import load_config


MODEL_PATH = Path.home() / "models/Qwen3.8-27B"


def write_trajectory(root: Path, scene: str = "Single-view", instance: str = "ObjectFactory",
                     *, missing: bool = False, failed_first: bool = False) -> Path:
    from PIL import Image

    path = root / scene / "gpt-6-astra" / instance / "post_train_trajectory" / "trajectory.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), "green").save(path.parent / "reference.png")
    image = "absent.png" if missing else "reference.png"
    conversations = [{"from": "human", "value": "<image>Write a Blender script."},
                     {"from": "gpt", "value": "import bpy\nprint('first')"},
                     {"from": "human", "value": "<image>Fix this script: import bpy\nprint('first')"},
                     {"from": "gpt", "value": "import bpy\nprint('fixed')"}]
    metadata = {"benchmark": scene, "instance": instance, "images_complete": True,
                "steps": [{"message_start": 1, "message_end": 3, "status": "ERR_EXEC" if failed_first else "OK"},
                          {"message_start": 3, "message_end": 5, "status": "OK"}],
                "sharegpt_conversion": {"conversation_source_mapping": [
                    {"source_message_indices": [i + 1], "tool_call_ids": [], "observation_call_ids": []}
                    for i in range(4)]}}
    row = {"system": "Return a complete Blender script.", "conversations": conversations,
           "images": [image, "reference.png"], "metadata": metadata}
    path.write_text(json.dumps(row) + "\n")
    return path


@pytest.fixture(scope="session")
def processor():
    if not MODEL_PATH.is_dir():
        pytest.skip("Local Qwen processor is unavailable")
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    processor.image_processor.size = {"shortest_edge": 1024, "longest_edge": 1024}
    return processor


@pytest.fixture
def config(tmp_path):
    write_trajectory(tmp_path / "runs")
    config = load_config(Path(__file__).resolve().parents[2] / "configs/sft_single_view.yaml")
    config.experiment.name = "test"
    config.experiment.output_dir = tmp_path / "output"
    config.data.runs_root = tmp_path / "runs"
    config.data.validation_fraction = 0
    config.data.test_fraction = 0
    config.model.dtype = "float32"
    config.method.backend.bf16 = False
    config.method.backend.dataloader_pin_memory = False
    config.template.min_image_pixels = 1024
    config.template.max_image_pixels = 1024
    config.template.max_length = 4096
    return config
