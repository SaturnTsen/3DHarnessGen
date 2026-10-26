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
    (path.parent / "images").mkdir(exist_ok=True)
    Image.new("RGB", (32, 32), "green").save(path.parent / "images/reference.png")
    image = "images/absent.png" if missing else "images/reference.png"
    source = {"path": "source.json", "sha256": "a" * 64, "locator": "/"}
    def user(index, text, image):
        return {"index": index, "role": "user", "content": [
            {"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": image}}]}
    row = {"schema_version": 3, "model": "gpt-6-astra", "tools": [], "messages": [
        {"index": 0, "role": "system", "content": "Return a complete Blender script."},
        user(1, "Write a Blender script.", image),
        {"index": 2, "role": "assistant", "content": "import bpy\nprint('first')"},
        user(3, "Fix this script: import bpy\nprint('first')", "images/reference.png"),
        {"index": 4, "role": "assistant", "content": "import bpy\nprint('fixed')"}],
        "metadata": {"scene": scene, "instance": instance, "session_id": "session", "view": "script",
                     "sources": [source], "steps": [
            {"id": "first", "context_indices": [0, 1], "target_index": 2, "target_kind": "script",
             "status": "ERROR" if failed_first else "OK", "source": source},
            {"id": "fix", "context_indices": [0, 3], "target_index": 4, "target_kind": "script",
             "status": "OK", "source": source}]}}
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


@pytest.fixture(scope="session")
def tiny_qwen_path(processor, tmp_path_factory):
    """A real multimodal Qwen architecture with tiny hidden dimensions."""
    from transformers import AutoConfig, Qwen3_5Config, Qwen3_5ForConditionalGeneration

    cfg = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True).to_dict()
    cfg["dtype"] = "float32"
    cfg["text_config"].update(
        dtype="float32", hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        layer_types=["full_attention"], partial_rotary_factor=1.0,
        rope_parameters={"rope_type": "default", "rope_theta": 10000, "mrope_section": [2, 3, 3]},
    )
    cfg["vision_config"].update(depth=1, hidden_size=32, intermediate_size=64,
                                num_heads=2, out_hidden_size=32, deepstack_visual_indexes=[])
    path = tmp_path_factory.mktemp("tiny_qwen")
    model = Qwen3_5ForConditionalGeneration(Qwen3_5Config.from_dict(cfg))
    model.save_pretrained(path)
    processor.save_pretrained(path)
    return path


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
    config.method.backend.fp16 = False
    config.method.backend.dataloader_pin_memory = False
    config.template.min_image_pixels = 1024
    config.template.max_image_pixels = 1024
    config.template.max_length = 4096
    return config
