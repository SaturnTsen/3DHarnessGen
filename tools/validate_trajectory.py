"""Validate the messages-based trajectory contract without loading models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator
from pydantic_yaml import parse_yaml_raw_as


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


Nonempty = Annotated[str, Field(min_length=1)]
Index = Annotated[int, Field(ge=0)]


class TextBlock(ContractModel):
    type: Literal["text"]
    text: str


class ImageURL(ContractModel):
    url: Nonempty
    detail: Literal["auto", "low", "high"] | None = None

    @field_validator("url")
    @classmethod
    def relative_image(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or len(path.parts) < 2 or path.parts[0] != "images" or "\\" in value:
            raise ValueError("image URL must be a relative images/... path without '..'")
        return value


class ImageBlock(ContractModel):
    type: Literal["image_url"]
    image_url: ImageURL


ContentBlock = Annotated[TextBlock | ImageBlock, Field(discriminator="type")]


class FunctionCall(ContractModel):
    name: Nonempty
    arguments: str

    @field_validator("arguments")
    @classmethod
    def object_arguments(cls, value: str) -> str:
        if not isinstance(load_json(value), dict):
            raise ValueError("function arguments must encode a JSON object")
        return value


class CustomCall(ContractModel):
    name: Nonempty
    input: str


class FunctionToolCall(ContractModel):
    id: Nonempty
    type: Literal["function"]
    function: FunctionCall


class CustomToolCall(ContractModel):
    id: Nonempty
    type: Literal["custom"]
    custom: CustomCall


ToolCall = Annotated[FunctionToolCall | CustomToolCall, Field(discriminator="type")]


class Message(ContractModel):
    index: Index
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[ContentBlock] | None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: Nonempty | None = None

    @model_validator(mode="after")
    def role_fields(self) -> Self:
        if self.tool_calls is not None and (self.role != "assistant" or not self.tool_calls):
            raise ValueError("tool_calls must be a nonempty list on an assistant message")
        if (self.role == "tool") != (self.tool_call_id is not None):
            raise ValueError("tool_call_id is required only on tool messages")
        if self.content is None and not self.tool_calls:
            raise ValueError("null content is allowed only for assistant tool calls")
        if isinstance(self.content, list) and self.role in {"system", "developer", "assistant"}:
            if any(isinstance(block, ImageBlock) for block in self.content):
                raise ValueError("images are allowed only in user and tool messages")
        return self


class FunctionDefinition(ContractModel):
    name: Nonempty
    description: str
    parameters: dict[str, JsonValue]
    strict: bool | None = None

    @field_validator("parameters")
    @classmethod
    def object_schema(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if value.get("type") != "object":
            raise ValueError("function parameters must declare type: object")
        return value


class TextFormat(ContractModel):
    type: Literal["text"]


class GrammarFormat(ContractModel):
    type: Literal["grammar"]
    syntax: Literal["lark", "regex"]
    definition: Nonempty


class CustomDefinition(ContractModel):
    name: Nonempty
    description: str
    format: Annotated[TextFormat | GrammarFormat, Field(discriminator="type")]


class FunctionTool(ContractModel):
    index: Index | None = None
    namespace: Nonempty | None = None
    namespace_description: str | None = None
    type: Literal["function"]
    function: FunctionDefinition


class CustomTool(ContractModel):
    index: Index | None = None
    namespace: Nonempty | None = None
    namespace_description: str | None = None
    type: Literal["custom"]
    custom: CustomDefinition


Tool = Annotated[FunctionTool | CustomTool, Field(discriminator="type")]
ToolContext = list[Tool]


class Source(ContractModel):
    path: Nonempty
    sha256: Annotated[str, Field(pattern=r"^[0-9a-fA-F]{64}$")]
    locator: Nonempty

    @field_validator("path")
    @classmethod
    def relative_source(cls, value: str) -> str:
        if PurePosixPath(value).is_absolute() or ":" in value or "\\" in value:
            raise ValueError("source path must be relative to the trajectory file")
        return value


class ToolsProvenance(ContractModel):
    status: Literal["recorded", "recovered"]
    sources: Annotated[list[Source], Field(min_length=1)]


class Step(ContractModel):
    id: Nonempty
    context_indices: list[Index]
    tool_context_index: Index
    target_index: Index
    target_kind: Literal["script", "tool_call", "final_reply", "status"]
    status: Literal["OK", "ERROR", "UNKNOWN"]
    input_provenance: Literal["recorded", "reconstructed", "unknown"]
    source: Source

    @model_validator(mode="after")
    def boundaries(self) -> Self:
        if self.context_indices != sorted(set(self.context_indices)):
            raise ValueError("context_indices must be strictly increasing")
        if any(index >= self.target_index for index in self.context_indices):
            raise ValueError("context cannot include the target or future messages")
        return self


class ExcludedEvent(ContractModel):
    source: Source
    reason: Nonempty


class Metadata(ContractModel):
    scene: Literal["Single-view", "Multi-view", "ActiveVisual", "Full3DInteraction"]
    instance: Nonempty
    session_id: Nonempty
    view: Literal["script", "agent"]
    sources: list[Source]
    tools_provenance: Annotated[list[ToolsProvenance], Field(min_length=1)]
    steps: list[Step]
    excluded_events: list[ExcludedEvent]


def tool_names(tools: list[Tool]) -> dict[str, str]:
    result = {}
    for tool in tools:
        name = tool.function.name if isinstance(tool, FunctionTool) else tool.custom.name
        if name in result:
            raise ValueError(f"duplicate tool name: {name}")
        result[name] = tool.type
    return result


class Trajectory(ContractModel):
    schema_version: Literal[2]
    model: Nonempty
    messages: Annotated[list[Message], Field(min_length=1)]
    tools: Annotated[list[ToolContext], Field(min_length=1)]
    metadata: Metadata

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @model_validator(mode="after")
    def relationships(self) -> Self:
        if len(self.metadata.tools_provenance) != len(self.tools):
            raise ValueError("tools_provenance must have one entry per tool context")
        for context_index, context in enumerate(self.tools):
            if not context:
                raise ValueError(f"tools[{context_index}]: tool context must not be empty")
            tool_names(context)
            if any(tool.index is not None for tool in context):
                if any(tool.index != index for index, tool in enumerate(context)):
                    raise ValueError(f"tools[{context_index}]: tool index must equal its context array position")
        script = self.metadata.scene in {"Single-view", "Multi-view"}
        if self.metadata.view != ("script" if script else "agent"):
            raise ValueError("metadata.view does not match scene")
        calls: set[str] = set()
        results: set[str] = set()
        for index, message in enumerate(self.messages):
            if message.index != index:
                raise ValueError(f"messages[{index}].index must equal {index}")
            for call in message.tool_calls or []:
                if call.id in calls:
                    raise ValueError(f"messages[{index}]: duplicate call ID {call.id}")
                calls.add(call.id)
            if message.tool_call_id is not None:
                if message.tool_call_id not in calls or message.tool_call_id in results:
                    raise ValueError(f"messages[{index}]: missing earlier call or duplicate result")
                results.add(message.tool_call_id)
        step_ids: set[str] = set()
        targets: set[int] = set()
        for step in self.metadata.steps:
            if step.id in step_ids or step.target_index in targets:
                raise ValueError(f"step {step.id}: duplicate step ID or target")
            step_ids.add(step.id)
            targets.add(step.target_index)
            if step.target_index >= len(self.messages):
                raise ValueError(f"step {step.id}: target_index is outside messages")
            target = self.messages[step.target_index]
            if target.role != "assistant":
                raise ValueError(f"step {step.id}: target must be assistant")
            if (step.target_kind == "tool_call") != bool(target.tool_calls):
                raise ValueError(f"step {step.id}: target_kind does not match tool_calls")
            if script and step.target_kind != "script":
                raise ValueError(f"step {step.id}: script view requires script targets")
            if step.tool_context_index >= len(self.tools):
                raise ValueError(f"step {step.id}: tool_context_index is outside tools")
            names = tool_names(self.tools[step.tool_context_index])
            for call in target.tool_calls or []:
                name = call.function.name if isinstance(call, FunctionToolCall) else call.custom.name
                if names.get(name) != call.type:
                    raise ValueError(f"step {step.id}: tool {name} is unavailable or has the wrong type")
            visible_calls: set[str] = set()
            for index in step.context_indices:
                message = self.messages[index]
                visible_calls.update(call.id for call in message.tool_calls or [])
                if message.tool_call_id is not None and message.tool_call_id not in visible_calls:
                    raise ValueError(f"step {step.id}: tool result has no call in visible context")
        return self


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def load_json(raw: str) -> object:
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)


def validate_record(raw: str) -> Trajectory:
    # Require JSON syntax before using the project's YAML/Pydantic reader.
    value = load_json(raw)
    return parse_yaml_raw_as(Trajectory, json.dumps(value, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="检查轨迹 JSON / JSONL 的字段及消息关联，不加载模型。")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.path.suffix.lower() not in {".json", ".jsonl"}:
        parser.error("文件扩展名必须为 .json 或 .jsonl")
    valid = 0
    failed = 0
    sessions: set[tuple[str, str, str]] = set()
    try:
        raw_file = args.path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        parser.exit(1, f"读取失败：{exc}\n")
    records = enumerate(raw_file.splitlines(), 1) if args.path.suffix.lower() == ".jsonl" else [(1, raw_file)]
    for line, raw in records:
        try:
            record = validate_record(raw)
            key = (record.metadata.scene, record.metadata.instance, record.metadata.session_id)
            if key in sessions:
                raise ValueError("重复的 session_id")
            sessions.add(key)
        except ValueError as exc:
            failed += 1
            print(f"{args.path}:记录/行 {line}: {exc}")
        else:
            valid += 1
    if valid + failed == 0:
        parser.exit(1, "校验失败：空文件\n")
    print(f"格式检查：{valid} 条通过，{failed} 条失败。未检查图片文件、来源文件真实性或训练可用性。")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
