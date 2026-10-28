"""SFT metrics for models returning chunked-loss counts instead of full logits."""

from __future__ import annotations

from typing import cast

from torch import Tensor
from transformers import Trainer
from trl.trainer.sft_trainer import SFTTrainer

from harness_training.models.qwen_long_context import LongContextCausalLMOutput


class LongContextSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        # Use the public Trainer loss path. SFTTrainer's ordinary NLL metrics
        # require full logits; its chunked_nll initializer patches model.forward.
        loss, outputs = cast(tuple[Tensor, LongContextCausalLMOutput], Trainer.compute_loss(
            self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch,
        ))
        count = cast(Tensor, self.accelerator.gather_for_metrics(cast(Tensor, outputs.num_valid_tokens))).sum()
        correct = cast(Tensor, self.accelerator.gather_for_metrics(cast(Tensor, outputs.num_correct_tokens))).sum()
        entropy = cast(Tensor, self.accelerator.gather_for_metrics(cast(Tensor, outputs.entropy_sum))).sum()
        mode = "train" if model.training else "eval"
        self._metrics[mode]["mean_token_accuracy"].append((correct / count.clamp(min=1)).item())
        self._metrics[mode]["entropy"].append((entropy / count.clamp(min=1)).item())
        if mode == "train":
            tokens = cast(Tensor, self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()))
            self._total_train_tokens += int(tokens.sum())
            self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
        return (loss, outputs) if return_outputs else loss
