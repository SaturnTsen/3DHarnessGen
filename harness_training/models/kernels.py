"""Require importable native kernels and verify the Qwen dispatch selection."""

from __future__ import annotations

import inspect
from importlib.metadata import version


def fused_sdpa():
    """Keep the quadratic-memory math backend disabled without global changes."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                        SDPBackend.CUDNN_ATTENTION])


def validate_long_context_kernels() -> None:
    if version("transformers") != "5.17.0":
        raise ValueError("The explicit long-context Qwen classes require transformers 5.17.0")
    import torch
    from causal_conv1d import causal_conv1d_fn
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5 import modeling_qwen3_5

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Optimized long-context training requires a BF16-capable CUDA GPU")
    if not torch.backends.cuda.is_flash_attention_available():
        raise ValueError("This PyTorch build does not include CUDA Flash SDPA support")
    # Transformers captures the selected implementation when its module imports.
    # Checking imports alone would miss an earlier, cached fallback selection.
    for wrapper, expected in [
        (modeling_qwen3_5.causal_conv1d_fn, causal_conv1d_fn),
        (modeling_qwen3_5.torch_chunk_gated_delta_rule, chunk_gated_delta_rule),
    ]:
        current = wrapper
        while inspect.isfunction(current):
            selected = inspect.getclosurevars(current).nonlocals.get("implementation")
            if selected is not None:
                if selected is not expected:
                    raise ValueError(f"{wrapper.__name__} selected a fallback instead of its native kernel; "
                                     "restart the process after installing dependencies")
                break
            current = getattr(current, "__wrapped__", None)
        else:
            raise ValueError(f"Cannot verify the optimized dispatch for {wrapper.__name__}")
