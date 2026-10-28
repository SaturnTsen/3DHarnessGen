"""Run on the target server; no downloaded model is needed for the core checks."""

from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from harness_training.config import LongContextConfig, TrainingConfig, load_config
from harness_training.models.chunked_loss import chunked_cross_entropy
from harness_training.models.qwen_long_context import LongContextQwenForConditionalGeneration


def options():
    return LongContextConfig(
        require_optimized_kernels=False, activation_dtype="float32", checkpoint_group_size=2,
        mlp_chunk_size=3, vision_batch_size=1, loss_chunk_size=3,
    )


def tiny_config():
    config = Qwen3_5Config(
        text_config=dict(
            vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            layer_types=["full_attention"] * 4, partial_rotary_factor=1.0,
            rope_parameters={"rope_type": "default", "rope_theta": 10000, "mrope_section": [2, 3, 3]},
            pad_token_id=0, use_cache=False,
        ),
        vision_config=dict(
            depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=32,
            num_position_embeddings=16, patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
        ),
        image_token_id=120, video_token_id=121, vision_start_token_id=122, vision_end_token_id=123,
        tie_word_embeddings=False,
    )
    config.harness_long_context = options().model_dump()
    config._attn_implementation = "eager"
    return config


@pytest.mark.parametrize("denominator", [None, 29])
def test_chunked_loss_and_gradients_match_full_cross_entropy(denominator):
    torch.manual_seed(13)
    hidden = torch.randn(2, 11, 8, requires_grad=True)
    weight = torch.randn(19, 8, requires_grad=True)
    labels = torch.randint(0, 19, (2, 11))
    labels[:, :5] = -100
    labels[1, -2:] = -100
    target = labels[:, 1:].reshape(-1)
    reference = F.cross_entropy(F.linear(hidden[:, :-1], weight).float().reshape(-1, 19),
                                target, ignore_index=-100, reduction="sum")
    reference = reference / ((target != -100).sum() if denominator is None else denominator)
    grads = torch.autograd.grad(reference, (hidden, weight))
    loss, _, _, count = chunked_cross_entropy(hidden, weight, labels, 3, denominator)
    actual = torch.autograd.grad(loss, (hidden, weight))
    torch.testing.assert_close(loss, reference)
    assert count == (target != -100).sum()
    for a, b in zip(actual, grads, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(actual[0][:, :4]) == 0


def test_fully_masked_loss_is_finite_differentiable_zero():
    hidden = torch.randn(1, 4, 8, requires_grad=True)
    weight = torch.randn(19, 8, requires_grad=True)
    loss, correct, entropy, count = chunked_cross_entropy(
        hidden, weight, torch.full((1, 4), -100), 2, None,
    )
    loss.backward()
    assert loss == 0 and correct == 0 and entropy == 0 and count == 0
    assert torch.count_nonzero(hidden.grad) == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_nested_checkpoints_preserve_loss_and_lora_gradients(device):
    from peft import LoraConfig, get_peft_model

    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA hybrid-attention comparison requires a GPU")
        from harness_training.models.kernels import validate_long_context_kernels

        validate_long_context_kernels()
    torch.manual_seed(23)
    config = tiny_config()
    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16
        config.text_config.layer_types = ["linear_attention"] * 3 + ["full_attention"]
        config.text_config.linear_num_key_heads = 1
        config.text_config.linear_num_value_heads = 2
        config.text_config.linear_key_head_dim = 16
        config.text_config.linear_value_head_dim = 16
        config.harness_long_context["activation_dtype"] = "bfloat16"
        config._attn_implementation = "flash_attention_2"
    reference = Qwen3_5ForConditionalGeneration(deepcopy(config))
    optimized = LongContextQwenForConditionalGeneration(deepcopy(config))
    assert reference.state_dict().keys() == optimized.state_dict().keys()
    optimized.load_state_dict(reference.state_dict())
    reference.to(device=device, dtype=dtype)
    optimized.to(device=device, dtype=dtype)
    lora = LoraConfig(r=2, lora_alpha=4, lora_dropout=0, target_modules=["q_proj", "v_proj"],
                      task_type="CAUSAL_LM")
    reference = get_peft_model(reference, deepcopy(lora))
    optimized = get_peft_model(optimized, deepcopy(lora))
    # Nonzero B exercises gradient propagation through the adapters into history.
    with torch.no_grad():
        for name, param in reference.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.01)
    optimized.load_state_dict(reference.state_dict())
    optimized.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ids = torch.randint(1, 90, (1, 13), device=device)
    labels = ids.clone()
    labels[:, :7] = -100
    results = []
    for model in (reference, optimized):
        model.train()
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False)
        output.loss.backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
        results.append((output, gradients))
    rtol, atol = (5e-2, 3e-3) if device == "cuda" else (2e-4, 2e-6)
    torch.testing.assert_close(results[0][0].loss, results[1][0].loss, rtol=rtol, atol=atol)
    assert results[1][0].logits is None
    assert results[1][0].num_valid_tokens == 6
    assert results[0][1].keys() == results[1][1].keys()
    for name, gradient in results[0][1].items():
        torch.testing.assert_close(gradient, results[1][1][name], rtol=rtol, atol=atol)
    assert "forward" not in optimized.get_base_model().__dict__


def test_image_microbatches_preserve_variable_grids_order_and_duplicates():
    torch.manual_seed(31)
    config = tiny_config()
    reference = Qwen3_5ForConditionalGeneration(deepcopy(config)).eval()
    optimized = LongContextQwenForConditionalGeneration(deepcopy(config)).eval()
    optimized.load_state_dict(reference.state_dict())
    grids = torch.tensor([[1, 4, 4], [1, 4, 6], [1, 4, 4]])
    first = torch.randn(16, 3 * 2 * 16 * 16)
    pixels = torch.cat([first, torch.randn(24, first.shape[1]), first])
    with torch.no_grad():
        expected = reference.get_image_features(pixels, grids, return_dict=True).pooler_output
        actual = optimized.get_image_features(pixels, grids, return_dict=True).pooler_output
    assert [len(image) for image in actual] == [4, 6, 4]
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual[0], actual[2])


@pytest.mark.parametrize("override", [
    "method.backend.activation_offloading=true",
    "method.backend.fsdp_config.cpu_offload=true",
    "method.backend.loss_type=chunked_nll",
    "method.backend.prediction_loss_only=false",
    "method.backend.fsdp_config.transformer_layer_cls_to_wrap=[Qwen3_5VisionBlock]",
])
def test_48k_rejects_unsafe_or_incompatible_settings(override):
    with pytest.raises(ValueError):
        load_config("configs/sft_interaction_mix_27b_fsdp.yaml", [override])


def test_long_context_options_are_explicit():
    cfg = load_config("configs/sft_interaction_mix_27b_fsdp.yaml")
    assert cfg.template.max_length == 49152
    for field in LongContextConfig.model_fields:
        payload = cfg.model_dump()
        del payload["method"]["long_context"][field]
        with pytest.raises(ValueError):
            TrainingConfig.model_validate(payload)


def test_long_context_sft_save_resume(config, tiny_qwen_path):
    from harness_training.training.trainer import Trainer

    config.model.path = tiny_qwen_path
    config.method.long_context = options()
    config.method.use_cpu = True
    config.method.max_steps = 1
    config.method.gradient_accumulation_steps = 1
    config.method.save_steps = 1
    config.method.backend.eval_strategy = "no"
    config.method.backend.do_eval = False
    config.method.backend.loss_type = "nll"
    config.method.backend.prediction_loss_only = True
    config.method.gradient_checkpointing = True
    trainer = Trainer(config)
    metrics = trainer.run()
    assert metrics["train_loss"] > 0
    assert "forward" not in trainer.model.model.get_base_model().__dict__
    assert (config.experiment.output_dir / "final/adapter_model.safetensors").is_file()
    config.method.max_steps = 2
    config.method.resume_from = config.experiment.output_dir / "checkpoints/checkpoint-1"
    Trainer(config).run()
    assert (config.experiment.output_dir / "checkpoints/checkpoint-2/trainer_state.json").is_file()
