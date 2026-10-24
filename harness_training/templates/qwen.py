from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from harness_training.types import ChatMessage, Processor, TensorInputs, ToolSchema
from harness_training.config import TemplateConfig
from harness_training.data.base import TrainingExample
from harness_training.data.contract import AdditionalTools, CustomToolCall, ImageBlock, TextFormat, GrammarFormat
from .base import BaseTemplate, RenderedExample, SampleTooLong, TemplateError


class QwenTemplate(BaseTemplate):
    def __init__(self, config: TemplateConfig, processor: Processor):
        self.config, self.processor = config, processor

    def _messages(self, example: TrainingExample) -> tuple[list[ChatMessage], list[Path], list[ToolSchema] | None]:
        messages: list[ChatMessage] = []
        images: list[Path] = []
        tools: list[ToolSchema] = []
        instructions: list[str] = []
        for index in example.indices:
            item = example.record.row.messages[index]
            if isinstance(item, AdditionalTools):
                for tool in item.tool_context():
                    definition = tool.definition.model_dump(exclude_unset=True, exclude={"type", "format"})
                    if tool.type == "custom":
                        parameter = self.config.custom_tools.get(tool.definition.name)
                        if not parameter:
                            raise TemplateError(f"No custom tool mapping for {tool.definition.name}")
                        definition["parameters"] = {"type": "object", "properties": {
                            parameter: {"type": "string"}}, "required": [parameter], "additionalProperties": False}
                        definition["description"] += "\nOriginal input format: " + json.dumps(
                            cast(TextFormat | GrammarFormat, tool.definition.format).model_dump(), ensure_ascii=False)
                    if tool.namespace:
                        definition["description"] += f"\nNamespace: {tool.namespace}\n{tool.namespace_description or ''}"
                    tools.append(cast(ToolSchema, {"type": "function", "function": definition}))
                continue
            content = item.content
            if isinstance(content, list):
                content = [({"type": "image"} if isinstance(block, ImageBlock)
                            else block.model_dump()) for block in content]
            images.extend(example.record.images_by_message[index])
            if item.role in {"system", "developer"}:
                instructions.append(content if isinstance(content, str) else
                                    "".join(block["text"] for block in content or []))
                continue
            message = cast(ChatMessage, {"role": item.role, "content": content or ""})
            if item.tool_call_id:
                message["tool_call_id"] = item.tool_call_id
            if item.tool_calls:
                calls = []
                for call in item.tool_calls:
                    if isinstance(call, CustomToolCall):
                        parameter = self.config.custom_tools.get(call.custom.name)
                        if not parameter:
                            raise TemplateError(f"No custom tool mapping for {call.custom.name}")
                        name, arguments = call.custom.name, {parameter: call.custom.input}
                    else:
                        name, arguments = call.function.name, json.loads(call.function.arguments)
                    calls.append({"id": call.id, "type": "function", "function": {
                        "name": name, "arguments": arguments}})
                message["tool_calls"] = calls
            messages.append(message)
        if instructions:
            messages.insert(0, {"role": "system", "content": "\n\n".join(instructions)})
        if not messages or messages[-1]["role"] != "assistant" or example.indices[-1] != example.target_index:
            raise TemplateError("The selected target must be the final assistant message")
        return messages, images, tools or None

    def render(self, example: TrainingExample) -> RenderedExample:
        messages, images, tools = self._messages(example)
        prompt = self.processor.apply_chat_template(
            messages[:-1], tokenize=self.config.tokenize, enable_thinking=self.config.enable_thinking,
            add_generation_prompt=self.config.prompt_add_generation_prompt, tools=tools)
        text = self.processor.apply_chat_template(
            messages, tokenize=self.config.tokenize, enable_thinking=self.config.enable_thinking,
            add_generation_prompt=self.config.target_add_generation_prompt, tools=tools)
        if not text.startswith(prompt):
            raise TemplateError("Chat template changes the prompt when the target is appended")
        return RenderedExample(prompt, text, tuple(images))

    def encode(self, example: TrainingExample) -> TensorInputs:
        import torch
        from PIL import Image

        rendered = self.render(example)
        decoded = []
        for path in rendered.images:
            with Image.open(path) as image:
                decoded.append(image.convert(self.config.image_mode))
        # Pass every encoding setting explicitly, including tool screenshots.
        full = dict(self.processor(
            text=[rendered.text], images=decoded or None, return_tensors=self.config.return_tensors,
            add_special_tokens=self.config.add_special_tokens, padding=self.config.padding,
            truncation=self.config.truncation))
        prompt = self.processor(
            text=[rendered.prompt], images=decoded or None, return_tensors=self.config.return_tensors,
            add_special_tokens=self.config.add_special_tokens, padding=self.config.padding,
            truncation=self.config.truncation)
        ids, prompt_ids = full["input_ids"], prompt["input_ids"]
        if ids.shape[1] > self.config.max_length:
            raise SampleTooLong(f"{example.id}: {ids.shape[1]} tokens exceeds {self.config.max_length}")
        boundary = prompt_ids.shape[1]
        if boundary >= ids.shape[1] or not torch.equal(ids[:, :boundary], prompt_ids):
            raise TemplateError("Prompt token IDs are not an exact prefix; cannot safely assign training labels")
        labels = ids.clone()
        labels[:, :boundary] = self.config.label_ignore_index
        labels[full["attention_mask"] == self.config.attention_padding_value] = self.config.label_ignore_index
        full["labels"] = labels
        return full
