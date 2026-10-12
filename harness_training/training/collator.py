from __future__ import annotations

from typing import Any

from harness_training.templates.base import BaseTemplate


class SFTCollator:
    def __init__(self, template: BaseTemplate, pad_token_id: int, examples=None):
        self.template, self.pad_token_id = template, pad_token_id
        self.examples = examples

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        from torch.nn.functional import pad

        encoded = [self.template.encode(item["example"] if "example" in item else
                                        self.examples[item["example_index"]]) for item in examples]
        length = max(item["input_ids"].shape[1] for item in encoded)
        sequence_keys = {"input_ids", "attention_mask", "labels", "token_type_ids", "mm_token_type_ids"}
        result = {}
        for key in set().union(*(item.keys() for item in encoded)):
            if key in sequence_keys:
                values = []
                for item in encoded:
                    tensor = item.get(key)
                    if tensor is None:
                        if key not in {"token_type_ids", "mm_token_type_ids"}:
                            raise ValueError(f"Missing sequence input: {key}")
                        tensor = torch.zeros_like(item["input_ids"])
                    value = -100 if key == "labels" else self.pad_token_id if key == "input_ids" else 0
                    values.append(pad(tensor, (0, length - tensor.shape[1]), value=value))
                result[key] = torch.cat(values, dim=0)
            elif key in {"pixel_values", "image_grid_thw"}:
                result[key] = torch.cat([item[key] for item in encoded if key in item], dim=0)
            else:
                raise ValueError(f"Unsupported Qwen processor output: {key}")
        return result
