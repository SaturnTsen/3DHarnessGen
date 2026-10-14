from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from harness_training.data.base import TrainingExample
from harness_training.types import TensorInputs

if TYPE_CHECKING:
    from torch import Tensor

from harness_training.templates.base import BaseTemplate


class IndexedSample(TypedDict):
    example_index: int


class DirectSample(TypedDict):
    example: TrainingExample


class SFTCollator:
    def __init__(self, template: BaseTemplate, pad_token_id: int, examples: list[TrainingExample], *,
                 label_ignore_index: int, attention_padding_value: int, token_type_padding_value: int):
        self.template, self.pad_token_id = template, pad_token_id
        self.examples = examples
        self.label_ignore_index = label_ignore_index
        self.attention_padding_value = attention_padding_value
        self.token_type_padding_value = token_type_padding_value

    def __call__(self, examples: list[IndexedSample] | list[DirectSample]) -> TensorInputs:
        import torch
        from torch.nn.functional import pad

        encoded = [self.template.encode(item["example"] if "example" in item else
                                        self.examples[item["example_index"]]) for item in examples]
        length = max(item["input_ids"].shape[1] for item in encoded)
        sequence_keys = {"input_ids", "attention_mask", "labels", "token_type_ids", "mm_token_type_ids"}
        result: TensorInputs = {}
        for key in set().union(*(item.keys() for item in encoded)):
            if key in sequence_keys:
                values: list[Tensor] = []
                for item in encoded:
                    tensor = item.get(key)
                    if tensor is None:
                        if key not in {"token_type_ids", "mm_token_type_ids"}:
                            raise ValueError(f"Missing sequence input: {key}")
                        tensor = torch.full_like(item["input_ids"], self.token_type_padding_value)
                    value = (self.label_ignore_index if key == "labels" else self.pad_token_id if key == "input_ids"
                             else self.attention_padding_value if key == "attention_mask" else self.token_type_padding_value)
                    values.append(pad(tensor, (0, length - tensor.shape[1]), value=value))
                result[key] = torch.cat(values, dim=0)
            elif key in {"pixel_values", "image_grid_thw"}:
                result[key] = torch.cat([item[key] for item in encoded if key in item], dim=0)
            else:
                raise ValueError(f"Unsupported Qwen processor output: {key}")
        return result
