"""Validate the project's v3 post_train_trajectory JSON/JSONL contract.

Messages use role/content, assistant tool_calls and tool tool_call_id.
Additional tool definitions are checked against each step's tool context.
Tool discovery descriptions remain in tool results, with image order preserved.
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationInfo, field_validator, model_validator
from pydantic_yaml import parse_yaml_raw_as

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


class BaseModelStrict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


Nonempty = Annotated[str, Field(min_length=1)]
Index = Annotated[int, Field(ge=0)]


# ----------------------- messages：消息与内容 -----------------------


class TextBlock(BaseModelStrict):
    type: Literal["text"]
    text: str


class ImageURL(BaseModelStrict):
    url: Nonempty
    detail: Literal["auto", "low", "high"] | None = None

    @field_validator("url")
    @classmethod
    def relative_image(cls, value: str, info: ValidationInfo) -> str:
        path = PurePosixPath(value)
        explicit_root = bool(info.context and info.context.get("explicit_image_root"))
        if path.is_absolute() or ".." in path.parts or len(path.parts) < 2 or "\\" in value:
            raise ValueError("image URL must be a relative path without '..'")
        if not explicit_root and path.parts[0] != "images":
            raise ValueError("image URL must be a relative images/... path without '..'")
        return value


class ImageBlock(BaseModelStrict):
    type: Literal["image_url"]
    image_url: ImageURL


ContentBlock = Annotated[TextBlock | ImageBlock, Field(discriminator="type")]


class FunctionToolCall(BaseModelStrict):
    class Function(BaseModelStrict):
        name: Nonempty
        arguments: str

        @field_validator("arguments")
        @classmethod
        def object_arguments(cls, value: str) -> str:
            if not isinstance(load_json(value), dict):
                raise ValueError("function arguments must encode a JSON object")
            return value

    id: Nonempty
    type: Literal["function"]
    function: Function


class CustomToolCall(BaseModelStrict):
    class Custom(BaseModelStrict):
        name: Nonempty
        input: str

    id: Nonempty
    type: Literal["custom"]
    custom: Custom


ToolCall = Annotated[FunctionToolCall | CustomToolCall, Field(discriminator="type")]


class Message(BaseModelStrict):
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


# ----------------------- additional tools -----------------------

class TextFormat(BaseModelStrict):
    type: Literal["text"]


class GrammarFormat(BaseModelStrict):
    type: Literal["grammar"]
    syntax: Literal["lark", "regex"]
    definition: Nonempty


class ToolDefinition(BaseModelStrict):
    name: Nonempty
    description: str
    type: Literal["function", "custom", "namespace"] | None = None
    parameters: dict[str, JsonValue] | None = None
    strict: bool | None = None
    format: TextFormat | GrammarFormat | None = Field(default=None, discriminator="type")
    tools: list[ToolDefinition] | None = None

    @field_validator("parameters")
    @classmethod
    def object_schema(cls, value: dict[str, JsonValue] | None) -> dict[str, JsonValue] | None:
        if value is not None and value.get("type") != "object":
            raise ValueError("function parameters must declare type: object")
        return value

    def check_kind(self, kind: str) -> None:
        common = {"type", "name", "description"}
        if kind == "function":
            allowed = common | {"parameters", "strict"}
            if self.parameters is None:
                raise ValueError("function definition requires parameters")
        elif kind == "custom":
            allowed = common | {"format"}
            if self.format is None:
                raise ValueError("custom definition requires format")
        elif kind == "namespace":
            allowed = common | {"tools"}
            if not self.tools:
                raise ValueError("namespace definition requires nonempty tools")
            for tool in self.tools:
                if tool.type not in {"function", "custom"}:
                    raise ValueError("namespace members must declare function or custom type")
                tool.check_kind(tool.type)
        else:
            raise ValueError(f"unsupported tool definition type: {kind}")
        if self.model_fields_set - allowed:
            raise ValueError(f"unexpected fields for {kind} definition")
        if "type" in self.model_fields_set and self.type != kind:
            raise ValueError("definition type does not match its tool")


class ToolNameSpace(BaseModelStrict):
    type: Literal["namespace"]
    name: Nonempty
    description: str
    tools: list[ToolDefinition] = Field(min_length=1)

    def check_kind(self, kind: str) -> None:
        if kind != self.type:
            raise ValueError("definition type does not match its namespace")
        for tool in self.tools:
            if tool.type not in {"function", "custom"}:
                raise ValueError("namespace members must declare function or custom type")
            tool.check_kind(tool.type)

    @model_validator(mode="after")
    def definition_fields(self) -> Self:
        self.check_kind(self.type)
        return self

    
class AdditionalTools(BaseModelStrict):
    index: Index
    type: Literal["additional_tools"]
    role: Literal["developer"]
    tools: list[ToolNameSpace] = Field(min_length=1)

    @field_validator("tools")
    @classmethod
    def definitions(cls, value: list[ToolDefinition]) -> list[ToolDefinition]:
        for definition in value:
            if definition.type is None:
                raise ValueError("additional tool definition requires type")
            definition.check_kind(definition.type)
        return value

    def tool_context(self) -> list[Tool]:
        result: list[Tool] = []
        for item in self.tools:
            namespace = (
                {"namespace": item.name, "namespace_description": item.description}
                if item.type == "namespace" else {}
            )
            definitions = item.tools if item.type == "namespace" else [item]
            for definition in cast(list[ToolDefinition], definitions):
                payload = definition.model_dump(exclude={"type"}, exclude_unset=True)
                result.append(Tool.model_validate({
                    "type": definition.type, definition.type: payload, **namespace,
                }))
        tool_names(result)
        return result


TrajectoryItem = AdditionalTools | Message


# ----------------------- tools：工具定义与上下文 -----------------------


class Tool(BaseModelStrict):
    index: Index | None = None
    namespace: Nonempty | None = None
    namespace_description: str | None = None
    type: Literal["function", "custom"]
    function: ToolDefinition | None = None
    custom: ToolDefinition | None = None

    @model_validator(mode="after")
    def definition_fields(self) -> Self:
        definition = self.function if self.type == "function" else self.custom
        other = "custom" if self.type == "function" else "function"
        if definition is None or other in self.model_fields_set:
            raise ValueError("tool must contain only its matching definition")
        definition.check_kind(self.type)
        return self

    @property
    def definition(self) -> ToolDefinition:
        return cast(ToolDefinition, self.function if self.type == "function" else self.custom)


ToolContext = list[Tool]


def tool_names(tools: list[Tool]) -> dict[str, str]:
    result = {}
    for tool in tools:
        name = tool.definition.name
        if name in result:
            raise ValueError(f"duplicate tool name: {name}")
        result[name] = tool.type
    return result


def context_payload(context: list[Tool]) -> list[dict]:
    return [tool.model_dump(exclude={"index"}) for tool in context]


# ----------------------- metadata：来源与训练步骤 -----------------------


class Source(BaseModelStrict):
    path: Nonempty
    sha256: Annotated[str, Field(pattern=r"^[0-9a-fA-F]{64}$")]
    locator: Nonempty

    @field_validator("path")
    @classmethod
    def relative_source(cls, value: str) -> str:
        if PurePosixPath(value).is_absolute() or ":" in value or "\\" in value:
            raise ValueError("source path must be relative to the trajectory file")
        return value


class Step(BaseModelStrict):
    id: Nonempty
    context_indices: list[Index]
    target_index: Index
    target_kind: Literal["script", "tool_call", "final_reply", "status"]
    status: Literal["OK", "ERROR", "UNKNOWN"]
    source: Source

    @model_validator(mode="after")
    def boundaries(self) -> Self:
        if self.context_indices != sorted(set(self.context_indices)):
            raise ValueError("context_indices must be strictly increasing")
        if any(index >= self.target_index for index in self.context_indices):
            raise ValueError("context cannot include the target or future messages")
        return self


class Metadata(BaseModelStrict):
    scene: Literal["Single-view", "Multi-view", "ActiveVisual", "Full3DInteraction"]
    instance: Nonempty
    session_id: Nonempty
    view: Literal["script", "agent"]
    sources: list[Source]
    steps: list[Step]


# ----------------------- Trajectory：完整轨迹与校验入口 -----------------------


class Trajectory(BaseModelStrict):
    schema_version: Literal[3]
    model: Nonempty
    messages: Annotated[list[TrajectoryItem], Field(min_length=1)]
    tools: list[ToolContext]
    metadata: Metadata

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @model_validator(mode="after")
    def relationships(self) -> Self:
        script = self.metadata.scene in {"Single-view", "Multi-view"}
        if self.metadata.view != ("script" if script else "agent"):
            raise ValueError("metadata.view does not match scene")
        if script:
            if self.tools or any(
                isinstance(item, AdditionalTools)
                or isinstance(item, Message) and (item.tool_calls or item.role == "tool")
                for item in self.messages
            ):
                raise ValueError("script view must omit Codex tools and tool events")
        elif not self.tools:
            raise ValueError("agent view requires nonempty tool contexts")
        for context_index, context in enumerate(self.tools):
            if not context:
                raise ValueError(f"tools[{context_index}]: tool context must not be empty")
            tool_names(context)
            if any(tool.index is not None for tool in context):
                if any(tool.index != index for index, tool in enumerate(context)):
                    raise ValueError(f"tools[{context_index}]: tool index must equal its context array position")
        calls: dict[str, str] = {}
        results: set[str] = set()
        additional_contexts: dict[int, list[Tool]] = {}
        for index, message in enumerate(self.messages):
            if message.index != index:
                raise ValueError(f"messages[{index}].index must equal {index}")
            if not script and isinstance(message, Message) and message.role == "system":
                raise ValueError("agent instructions must retain the recorded developer role")
            if isinstance(message, AdditionalTools):
                context = message.tool_context()
                payload = context_payload(context)
                if payload not in [context_payload(context) for context in self.tools]:
                    raise ValueError(f"messages[{index}]: additional_tools does not match any tool context")
                additional_contexts[index] = context
            if isinstance(message, Message):
                for call in message.tool_calls or []:
                    if call.id in calls:
                        raise ValueError(f"messages[{index}]: duplicate call ID {call.id}")
                    calls[call.id] = call.type
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
            if not isinstance(target, Message) or target.role != "assistant":
                raise ValueError(f"step {step.id}: target must be assistant")
            if (step.target_kind == "tool_call") != bool(target.tool_calls):
                raise ValueError(f"step {step.id}: target_kind does not match tool_calls")
            if script and step.target_kind != "script":
                raise ValueError(f"step {step.id}: script view requires script targets")
            if script:
                continue
            visible_tools = [additional_contexts[index] for index in step.context_indices if index in additional_contexts]
            if len(visible_tools) != 1:
                raise ValueError(f"step {step.id}: context requires exactly one additional_tools item")
            names = tool_names(visible_tools[0])
            for call in target.tool_calls or []:
                name = call.function.name if isinstance(call, FunctionToolCall) else call.custom.name
                if names.get(name) != call.type:
                    raise ValueError(f"step {step.id}: tool {name} is unavailable or has the wrong type")
            visible_calls: set[str] = set()
            for index in step.context_indices:
                message = self.messages[index]
                if isinstance(message, Message):
                    visible_calls.update(call.id for call in message.tool_calls or [])
                    if message.tool_call_id is not None and message.tool_call_id not in visible_calls:
                        raise ValueError(f"step {step.id}: tool result has no call in visible context")
        return self



def validate_record(raw: str, *, explicit_image_root: bool = False) -> Trajectory:
    # Require JSON syntax before using the project's YAML/Pydantic reader.
    value = load_json(raw)
    if explicit_image_root:
        return Trajectory.model_validate(value, context={"explicit_image_root": True})
    return parse_yaml_raw_as(Trajectory, json.dumps(value, ensure_ascii=False))
