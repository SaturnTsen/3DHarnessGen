"""Long-context SFT owns target selection, chunked loss and token metrics."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.distributed.tensor import DTensor

from .sft_trainer import DataLoaderSFTTrainer

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


class LongContextSFTTrainer(DataLoaderSFTTrainer):
    def __init__(self, *, loss_chunk_size: int, **kwargs):
        super().__init__(**kwargs)
        self.loss_chunk_size = loss_chunk_size
        # Trainer uses this flag to count targets across gradient accumulation
        # and leaves normalization to compute_loss(), rather than dividing twice.
        self.model_accepts_loss_kwargs = True
        self._loss_shifts_labels = True

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        outputs = model(**inputs, return_hidden_states_only=True, return_dict=True)
        weight = self.accelerator.unwrap_model(model).get_output_embeddings().weight
        if isinstance(weight, DTensor):
            # Gather once per forward, outside the rank-dependent target chunks.
            weight = weight.full_tensor()
        loss, correct, entropy, count = chunked_cross_entropy(
            outputs.last_hidden_state, weight, labels, self.loss_chunk_size, num_items_in_batch,
        )
        # The denominator counts targets across ranks; DDP/FSDP averages gradients.
        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            loss = loss * self.accelerator.num_processes
        count = cast(Tensor, self.accelerator.gather_for_metrics(count)).sum()
        correct = cast(Tensor, self.accelerator.gather_for_metrics(correct)).sum()
        entropy = cast(Tensor, self.accelerator.gather_for_metrics(entropy)).sum()
        mode = "train" if model.training else "eval"
        self._metrics[mode]["mean_token_accuracy"].append((correct / count.clamp(min=1)).item())
        self._metrics[mode]["entropy"].append((entropy / count.clamp(min=1)).item())
        if mode == "train":
            tokens = cast(Tensor, self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()))
            self._total_train_tokens += int(tokens.sum())
            self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
        # Evaluation only needs the loss; never retain the full hidden-state graph.
        return (loss, {"loss": loss}) if return_outputs else loss
