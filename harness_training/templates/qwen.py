from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from harness_training.config import TemplateConfig
from harness_training.data.base import TrainingExample
from .base import BaseTemplate, RenderedExample, SampleTooLong, TemplateError


class QwenTemplate(BaseTemplate):
    def __init__(self, config: TemplateConfig, processor: Any, *, strip_plugin_recommendations: bool = True):
        self.config, self.processor = config, processor
        self.strip_plugin_recommendations = strip_plugin_recommendations
        self.tools = None
        if config.tools_path:
            self.tools = json.loads(config.tools_path.read_text(encoding="utf-8"))
            if not isinstance(self.tools, list) or not all(isinstance(t, dict) and isinstance(t.get("function"), dict)
                                                        and t["function"].get("name") for t in self.tools):
                raise TemplateError("tools_path must contain a list of function tool schemas")

    @staticmethod
    def _tool_list(value: Any) -> list | None:
        if isinstance(value, str):
            value = json.loads(value)
        if value is not None and not isinstance(value, list):
            raise TemplateError("tools must be a list or a JSON-encoded list")
        return value

    def _messages(self, example: TrainingExample) -> tuple[list[dict], list[Path], list | None]:
        row = example.record.row
        tools = self._tool_list(row.get("tools"))
        if tools is not None and self.tools is not None and tools != self.tools:
            raise TemplateError("Recorded tools disagree with configured execution environment")
        tools = tools if tools is not None else self.tools
        allowed_tools = {t.get("function", t).get("name") for t in tools or []}
        mapping = row["metadata"]["sharegpt_conversion"]["conversation_source_mapping"]
        messages = [{"role": "system", "content": row.get("system", "")}]
        images = []
        pending: set[str] = set()
        returned: set[str] = set()
        for index in example.indices:
            conversation = row["conversations"][index]
            role, value = conversation["from"], conversation["value"]
            meta = mapping[index]
            call_ids = meta.get("tool_call_ids", [])
            if role in {"gpt", "function_call"}:
                if pending - returned:
                    raise TemplateError(f"Tool results missing for calls: {sorted(pending - returned)}")
                pending, returned = set(call_ids), set()
            if role == "human" and self.strip_plugin_recommendations:
                value = re.sub(r"<recommended_plugins>.*?</recommended_plugins>", "", value, flags=re.DOTALL)
            if role == "function_call":
                try:
                    calls = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise TemplateError("function_call is not valid JSON") from exc
                calls = calls if isinstance(calls, list) else [calls]
                if len(calls) != len(call_ids):
                    raise TemplateError("Tool call IDs do not match recorded calls")
                parsed = []
                for call, call_id in zip(calls, call_ids, strict=True):
                    if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), dict):
                        raise TemplateError("Tool calls require a name and an arguments object")
                    if call["name"] not in allowed_tools:
                        raise TemplateError(f"No tool schema for {call['name']}")
                    parsed.append({"id": call_id, "type": "function", "function": call})
                messages.append({"role": "assistant", "content": "", "tool_calls": parsed})
                continue
            if role == "gpt" and call_ids:
                # The exporter explicitly marks these already-rendered Qwen calls.
                # Keep their bytes rather than attempting to parse arbitrary code as XML.
                if row["metadata"]["sharegpt_conversion"].get("mixed_assistant_tool_format") != "qwen3_8":
                    raise TemplateError("Unsupported mixed assistant/tool format")
                names = re.findall(r"<function=([^>]+)>", value)
                if len(names) != len(call_ids) or any(name not in allowed_tools for name in names):
                    raise TemplateError("Mixed assistant reply has missing or unknown tool calls")
            if role == "observation":
                ids = meta.get("observation_call_ids", [])
                if not ids or not set(ids) <= pending:
                    raise TemplateError("Tool observation cannot be matched to preceding calls")
                returned.update(ids)
            blocks = []
            bound = iter(example.record.images_by_conversation[index])
            pieces = value.split("<image>")
            for position, text in enumerate(pieces):
                if text:
                    blocks.append({"type": "text", "text": text})
                if position < len(pieces) - 1:
                    image = next(bound)
                    images.append(image)
                    blocks.append({"type": "image"})
            message = {"role": {"human": "user", "gpt": "assistant", "observation": "tool"}[role], "content": blocks}
            if role == "observation":
                message["tool_call_id"] = meta["observation_call_ids"][0]
            messages.append(message)
        if messages[-1]["role"] != "assistant" or example.indices[-1] != example.target_index:
            raise TemplateError("The selected target must be the final assistant message")
        return messages, images, tools

    def render(self, example: TrainingExample) -> RenderedExample:
        messages, images, tools = self._messages(example)
        kwargs = {"tokenize": False, "enable_thinking": self.config.enable_thinking}
        if tools:
            kwargs["tools"] = tools
        prompt = self.processor.apply_chat_template(messages[:-1], add_generation_prompt=True, **kwargs)
        text = self.processor.apply_chat_template(messages, add_generation_prompt=False, **kwargs)
        if not text.startswith(prompt):
            raise TemplateError("Chat template changes the prompt when the target is appended")
        return RenderedExample(prompt, text, tuple(images))

    def encode(self, example: TrainingExample) -> dict[str, Any]:
        import torch
        from PIL import Image

        rendered = self.render(example)
        decoded = []
        for path in rendered.images:
            with Image.open(path) as image:
                decoded.append(image.convert("RGB"))
        kwargs = {"return_tensors": "pt", "add_special_tokens": False, "padding": False,
                  "truncation": False}
        if decoded:
            kwargs["images"] = decoded
        # Include tool screenshots explicitly instead of relying on a role-filtering helper.
        full = dict(self.processor(text=[rendered.text], **kwargs))
        prompt = self.processor(text=[rendered.prompt], **kwargs)
        ids, prompt_ids = full["input_ids"], prompt["input_ids"]
        if ids.shape[1] > self.config.max_length:
            raise SampleTooLong(f"{example.id}: {ids.shape[1]} tokens exceeds {self.config.max_length}")
        boundary = prompt_ids.shape[1]
        if boundary >= ids.shape[1] or not torch.equal(ids[:, :boundary], prompt_ids):
            raise TemplateError("Prompt token IDs are not an exact prefix; cannot safely assign training labels")
        labels = ids.clone()
        labels[:, :boundary] = -100
        labels[full["attention_mask"] == 0] = -100
        full["labels"] = labels
        return full
