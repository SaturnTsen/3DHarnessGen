"""Target-only cross entropy with bounded logits and recomputed chunks."""

from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _loss_chunk(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor):
    logits = F.linear(hidden, weight).float()
    loss = F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")
    with torch.no_grad():
        valid = labels != -100
        correct = ((logits.argmax(-1) == labels) & valid).sum()
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = (-(log_probs.exp() * log_probs).sum(-1) * valid).sum()
    return loss, correct, entropy


def chunked_cross_entropy(hidden_states: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor,
                          chunk_size: int, num_items_in_batch: torch.Tensor | int | None):
    # Shift before selecting valid positions, including the prompt/target boundary.
    hidden = hidden_states[..., :-1, :].reshape(-1, hidden_states.shape[-1])
    shifted = labels[..., 1:].reshape(-1)
    valid = shifted != -100
    count = valid.sum()
    selected = valid.nonzero(as_tuple=True)[0]
    # A masked batch must still have a differentiable zero and reach the backbone.
    if selected.numel() == 0:
        selected = torch.zeros(1, device=hidden.device, dtype=torch.long)
    hidden, shifted = hidden[selected], shifted[selected]
    weight = weight.to(hidden.dtype)
    loss = hidden.new_zeros((), dtype=torch.float32)
    correct = count.new_zeros(())
    entropy = loss.detach().clone()
    for start in range(0, hidden.shape[0], chunk_size):
        h, targets = hidden[start:start + chunk_size], shifted[start:start + chunk_size]
        if torch.is_grad_enabled():
            part, hits, uncertainty = checkpoint(_loss_chunk, h, weight, targets, use_reentrant=False)
        else:
            part, hits, uncertainty = _loss_chunk(h, weight, targets)
        loss = loss + part
        correct = correct + hits
        entropy = entropy + uncertainty
    denominator = count.clamp(min=1) if num_items_in_batch is None else num_items_in_batch
    return loss / torch.as_tensor(denominator, device=loss.device), correct, entropy, count
