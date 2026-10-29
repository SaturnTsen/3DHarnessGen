"""Check sample identities and real optimizer recovery across launch topologies."""

import json
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest

from harness_training.config import load_config, save_config
from harness_training.data.elastic import GlobalBatchSampler, epoch_order
from harness_training.training.elastic import load_progress, validate_manifest
from conftest import write_trajectory


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def elastic_qwen_path(processor, tmp_path_factory):
    """Keep the real Qwen image path but use a small test vocabulary."""
    from copy import deepcopy
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import AutoConfig, PreTrainedTokenizerFast, Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from conftest import MODEL_PATH

    small_processor = deepcopy(processor)
    original = processor.tokenizer
    words = list(dict.fromkeys(["<unk>", *original.all_special_tokens, "print", "first", "fixed", "import", "bpy"]))
    tokenizer = Tokenizer(WordLevel({word: index for index, word in enumerate(words)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    small_processor.tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", pad_token=original.pad_token,
        eos_token=original.eos_token, additional_special_tokens=original.all_special_tokens,
    )
    config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True).to_dict()
    config["text_config"].update(vocab_size=len(small_processor.tokenizer), hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, layer_types=["full_attention"] * 2, partial_rotary_factor=1.0,
        rope_parameters={"rope_type": "default", "rope_theta": 10000, "mrope_section": [2, 3, 3]},
        pad_token_id=small_processor.tokenizer.pad_token_id, eos_token_id=small_processor.tokenizer.eos_token_id,
        bos_token_id=None)
    for name, token in [("image_token_id", "<|image_pad|>"), ("video_token_id", "<|video_pad|>"),
                        ("vision_start_token_id", "<|vision_start|>"), ("vision_end_token_id", "<|vision_end|>")]:
        config[name] = small_processor.tokenizer.convert_tokens_to_ids(token)
    config["vision_config"].update(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                                    out_hidden_size=32, deepstack_visual_indexes=[])
    torch.manual_seed(42)
    model = Qwen3_5ForConditionalGeneration(Qwen3_5Config.from_dict(config))
    path = tmp_path_factory.mktemp("elastic_qwen")
    model.save_pretrained(path)
    small_processor.save_pretrained(path)
    return path


@pytest.mark.parametrize("size", [1, 3, 8, 11, 32])
@pytest.mark.parametrize("epoch", [0, 1, 5])
def test_global_update_samples_do_not_depend_on_topology(size, epoch):
    groups = []
    for world, batch, accumulation in [(1, 1, 8), (2, 1, 4), (4, 1, 2), (2, 2, 2), (4, 2, 1)]:
        ranks = []
        for rank in range(world):
            sampler = GlobalBatchSampler(size, 42, "random", batch, accumulation, world, rank)
            sampler.set_epoch(epoch)
            ranks.append(list(sampler))
        updates = []
        for update in range(len(ranks[0]) // accumulation):
            updates.append([index for micro in range(accumulation) for rank in range(world)
                            for index in ranks[rank][update * accumulation + micro] if index != -1])
        assert [index for group in updates for index in group] == epoch_order(size, 42, epoch, "random")
        groups.append(updates)
    assert all(group == groups[0] for group in groups)


def run_training(config, path, world, overrides=()):
    save_config(config, path)
    command = [sys.executable]
    if world > 1:
        command += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc-per-node={world}"]
    command += ["-m", "scripts.post_train", "train", "--config", str(path), *overrides]
    env = {**os.environ, "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
           "CUBLAS_WORKSPACE_CONFIG": ":16:8", "PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    if world == 1:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=240)
    (path.with_suffix(".log")).write_text(result.stdout + result.stderr)
    assert result.returncode == 0, (result.stdout + result.stderr)[-20000:]
    return result.stdout


def test_dataloader_skip_uses_saved_epoch_order_with_new_topology():
    from accelerate import skip_first_batches
    from harness_training.data.elastic import EpochDataLoader, IndexedTrainingDataset

    expected = epoch_order(19, 42, 2, "random")[8:16]
    actual = []
    ranks = []
    for rank in range(2):
        sampler = GlobalBatchSampler(19, 42, "random", 1, 4, 2, rank)
        loader = EpochDataLoader(IndexedTrainingDataset(19), batch_sampler=sampler)
        loader = skip_first_batches(loader, 4)
        loader.batch_sampler.batch_sampler.set_epoch(2)
        ranks.append(list(loader))
    for micro in range(4):
        for rank in range(2):
            actual.append(int(ranks[rank][micro]["example_index"][0]))
    assert actual == expected


def test_padding_masks_labels_and_preserves_real_image_inputs(config, processor):
    import torch
    from harness_training.data.elastic import ElasticCollator
    from harness_training.training.collator import SFTCollator
    from harness_training.training.trainer import Trainer

    trainer = Trainer(config)
    data = trainer.prepare()
    collator = SFTCollator(trainer.template, processor.tokenizer.pad_token_id, data.train,
                          label_ignore_index=config.template.label_ignore_index,
                          attention_padding_value=config.template.attention_padding_value,
                          token_type_padding_value=config.template.token_type_padding_value)
    batch = ElasticCollator(collator)([{"example_index": 0, "elastic_padding": False},
                                      {"example_index": 0, "elastic_padding": True}])
    assert (batch["labels"][0] != -100).any()
    assert torch.all(batch["labels"][1] == -100)
    assert torch.equal(batch["input_ids"][0], batch["input_ids"][1])
    assert batch["image_grid_thw"].shape[0] == 2
    assert torch.equal(batch["image_grid_thw"][0], batch["image_grid_thw"][1])


@pytest.mark.parametrize("change", ["data", "model", "template", "targets", "sources", "versions", "code"])
def test_resume_rejects_changed_identity(config, change):
    current = {"config": config.model_dump(mode="json"), "targets": {"train": ["sample"]},
               "sources": {"data": "hash"}, "versions": {"torch": "version"}, "code": {"a.py": "hash"}}
    prior = deepcopy(current)
    if change in {"data", "model", "template"}:
        prior["config"][change]["changed"] = True
    else:
        prior[change]["changed"] = "value"
    with pytest.raises(ValueError):
        validate_manifest(prior, current)


def test_only_explicit_legacy_code_versions_are_accepted(config):
    known = json.loads((ROOT / "harness_training/training/legacy_code.json").read_text())
    current = {"config": config.model_dump(mode="json"), "targets": {"train": ["sample"]},
               "sources": {}, "versions": {}, "code": {"new.py": "new-hash"}}
    for code in known.values():
        prior = deepcopy(current)
        prior["code"] = code
        prior["config"]["method"].pop("resume_policy")
        validate_manifest(prior, current)
        prior["code"] = {**code, "data/trajectory.py": "unknown-change"}
        with pytest.raises(ValueError, match="supported version"):
            validate_manifest(prior, current)


def test_legacy_progress_reconstruction_and_global_batch_guard(config, tmp_path):
    prior = {"config": config.model_dump(mode="json"), "targets": {"train": [f"sample{i}" for i in range(19)]}}
    prior["config"]["method"]["gradient_accumulation_steps"] = 8
    prior["config"]["method"].pop("resume_policy")
    checkpoint = tmp_path / "checkpoint-4"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text('{"global_step":4}')
    (checkpoint / "rng_state.pth").write_bytes(b"state")
    config.method.gradient_accumulation_steps = 4
    progress = load_progress(checkpoint, prior, config, 2)
    assert progress.epoch == 1 and progress.next_sample == 8
    assert progress.order == epoch_order(19, 42, 1, "random")
    config.method.gradient_accumulation_steps = 8
    with pytest.raises(ValueError, match="Keep global batch size 8"):
        load_progress(checkpoint, prior, config, 2)


def setup_config(tmp_path, tiny_qwen_path, strategy, quantization):
    config = load_config(ROOT / "configs/sft_single_view.yaml", [
        f"distributed={strategy}", f"quantization={quantization}", "tracking=wandb",
        "tracking.mode=offline", "method.backend.eval_strategy=no", "method.backend.do_eval=false",
    ])
    config.model.path = tiny_qwen_path
    config.data.runs_root = tmp_path / "runs"
    config.data.validation_fraction = config.data.test_fraction = 0
    for index in range(5):
        write_trajectory(config.data.runs_root, instance=f"Object{index}Factory")
    config.experiment.output_dir = tmp_path / "output"
    config.method.backend.lr_scheduler_type = "constant"
    if quantization == "4bit":
        config.method.backend.loss_type = "chunked_nll"
        config.method.backend.prediction_loss_only = True
    config.method.backend.disable_tqdm = True
    config.method.save_steps = 1
    config.method.save_total_limit = 5
    config.method.warmup_ratio = 0
    config.method.learning_rate = .005
    config.method.max_steps = 1
    config.method.gradient_accumulation_steps = 4
    config.method.gradient_checkpointing = True
    config.model.lora.rank = 2
    config.model.lora.alpha = 4
    config.model.lora.dropout = 0
    config.model.lora.target_modules = ["q_proj", "v_proj"]
    config.tracking.settings.x_disable_stats = True
    return config


def test_cpu_elastic_resume_and_masked_tail(config, tiny_qwen_path, tmp_path):
    config.model.path = tiny_qwen_path
    config.method.resume_policy = "elastic"
    config.method.use_cpu = True
    config.method.max_steps = 1
    config.method.save_steps = 1
    config.method.save_total_limit = 5
    config.method.gradient_accumulation_steps = 3
    config.method.backend.lr_scheduler_type = "constant"
    config.method.backend.eval_strategy = "no"
    config.method.backend.do_eval = False
    config.method.backend.disable_tqdm = True
    config.method.warmup_ratio = 0
    run_training(config, tmp_path / "first.yaml", 1)
    checkpoint = config.experiment.output_dir / "checkpoints/checkpoint-1"
    progress = json.loads((checkpoint / "data_progress.json").read_text())
    assert progress["epoch"] == 1 and progress["next_sample"] == 0
    config.method.max_steps = 2
    config.method.resume_from = checkpoint
    run_training(config, tmp_path / "second.yaml", 1)
    import torch

    portable = torch.load(config.experiment.output_dir / "checkpoints/checkpoint-2/portable_state.pt",
                          map_location="cpu", weights_only=True)
    assert all(int(value["step"]) == 2 for value in portable["optimizer"]["state"].values())


@pytest.mark.parametrize("strategy,quantization,long_context", [
    ("ddp", "none", False), ("fsdp", "none", False), ("fsdp", "4bit", False),
    ("fsdp", "4bit", True),
])
def test_legacy_single_to_two_gpu_and_back(elastic_qwen_path, tmp_path, strategy, quantization, long_context):
    import torch

    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA GPUs")
    config = setup_config(tmp_path, elastic_qwen_path, "single", quantization)
    if long_context:
        from harness_training.config import LongContextConfig

        config.method.long_context = LongContextConfig(
            require_optimized_kernels=False, activation_dtype="bfloat16", checkpoint_group_size=2,
            mlp_chunk_size=16, vision_batch_size=1, loss_chunk_size=8,
        )
        config.method.backend.loss_type = "nll"
        config.method.backend.prediction_loss_only = True
    reference = deepcopy(config)
    run_training(config, tmp_path / "single.yaml", 1)
    output = config.experiment.output_dir
    run_id = json.loads((output / "wandb_run.json").read_text())["id"]
    checkpoint = output / "checkpoints/checkpoint-1"
    assert not (checkpoint / "data_progress.json").exists()
    destination = load_config(ROOT / "configs/sft_single_view.yaml", [f"distributed={strategy}"])
    config.distributed = destination.distributed
    config.method.backend.fsdp = destination.method.backend.fsdp
    config.method.backend.fsdp_config = destination.method.backend.fsdp_config
    config.method.resume_policy = "elastic"
    config.method.gradient_accumulation_steps = 2
    config.method.max_steps = 2
    config.method.resume_from = checkpoint
    log = run_training(config, tmp_path / "double.yaml", 2)
    assert "下一样本位置 4" in log
    checkpoint = output / "checkpoints/checkpoint-2"
    progress = json.loads((checkpoint / "data_progress.json").read_text())
    assert progress["next_sample"] == 8 and progress["world_size"] == 2
    state = torch.load(checkpoint / "portable_state.pt", map_location="cpu", weights_only=True)
    assert all(int(value["step"]) == 2 for value in state["optimizer"]["state"].values())
    assert state["scheduler"]["last_epoch"] == 2
    from conftest import read_wandb_records

    summaries = []
    for segment in (output / "wandb").glob("offline-run-*/run-*.wandb"):
        for record in read_wandb_records(segment):
            if record.HasField("summary"):
                summaries.extend((item.key, json.loads(item.value_json)) for item in record.summary.update)
    assert ("resume/source_world_size", 1) in summaries
    assert ("resume/world_size", 2) in summaries
    # Return to a single GPU and finish the short tail without training padding.
    config.distributed.strategy = "single"
    config.method.backend.fsdp = config.method.backend.fsdp_config = None
    config.method.gradient_accumulation_steps = 4
    config.method.max_steps = 4
    config.method.resume_from = checkpoint
    run_training(config, tmp_path / "single-again.yaml", 1)
    progress = json.loads((output / "checkpoints/checkpoint-4/data_progress.json").read_text())
    assert progress["epoch"] == 1 and progress["next_sample"] == 4
    final = torch.load(output / "checkpoints/checkpoint-4/portable_state.pt", map_location="cpu", weights_only=True)
    assert all(int(value["step"]) == 4 for value in final["optimizer"]["state"].values())
    assert any(not torch.equal(final["model"][name], tensor) for name, tensor in state["model"].items())
    assert json.loads((output / "wandb_run.json").read_text())["id"] == run_id
    if strategy == "ddp" and quantization == "none":
        from safetensors.torch import load_file

        reference.experiment.output_dir = tmp_path / "uninterrupted"
        reference.method.max_steps = 2
        run_training(reference, tmp_path / "uninterrupted.yaml", 1)
        expected = load_file(reference.experiment.output_dir / "final/adapter_model.safetensors")
        for name, value in state["model"].items():
            torch.testing.assert_close(value, expected[name.replace(".default.", ".")], rtol=.03, atol=2e-4)
        expected_optimizer = torch.load(reference.experiment.output_dir / "checkpoints/checkpoint-2/optimizer.pt",
                                        weights_only=True, map_location="cpu")
        for group, old in zip(state["optimizer"]["param_groups"], expected_optimizer["param_groups"], strict=True):
            for name, index in zip(group["params"], old["params"], strict=True):
                for key in ("exp_avg", "exp_avg_sq", "step"):
                    torch.testing.assert_close(state["optimizer"]["state"][name][key],
                                               expected_optimizer["state"][index][key], rtol=.04, atol=2e-5)


@pytest.mark.parametrize("destination_world", [1, 4])
def test_legacy_fsdp_to_different_world(elastic_qwen_path, tmp_path, destination_world):
    import torch

    if torch.cuda.device_count() < max(2, destination_world):
        pytest.skip(f"Requires {max(2, destination_world)} CUDA GPUs")
    config = setup_config(tmp_path, elastic_qwen_path, "fsdp", "none")
    config.method.gradient_accumulation_steps = 2
    run_training(config, tmp_path / "old-fsdp.yaml", 2)
    config.method.resume_policy = "elastic"
    config.method.resume_from = config.experiment.output_dir / "checkpoints/checkpoint-1"
    config.method.max_steps = 2
    config.method.gradient_accumulation_steps = 4 // destination_world
    if destination_world == 1:
        config.distributed.strategy = "single"
        config.method.backend.fsdp = config.method.backend.fsdp_config = None
    run_training(config, tmp_path / "new-world.yaml", destination_world)
    checkpoint = config.experiment.output_dir / "checkpoints/checkpoint-2"
    progress = json.loads((checkpoint / "data_progress.json").read_text())
    assert progress["next_sample"] == 8 and progress["world_size"] == destination_world
    state = torch.load(checkpoint / "portable_state.pt", map_location="cpu", weights_only=True)
    assert all(int(value["step"]) == 2 for value in state["optimizer"]["state"].values())
