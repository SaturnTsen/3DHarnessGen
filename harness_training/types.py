from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, NotRequired, Protocol, TypedDict

if TYPE_CHECKING:
    from PIL.Image import Image
    from torch import Tensor
    from transformers import PreTrainedTokenizerBase

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type TensorInputs = dict[str, Tensor]
type Metrics = dict[str, float | int]


class ToolFunction(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: NotRequired[JsonObject]


class ToolSchema(TypedDict):
    type: str
    function: ToolFunction


class FunctionCall(TypedDict):
    name: str
    arguments: JsonObject


class ToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: FunctionCall


class TextBlock(TypedDict):
    type: Literal["text"]
    text: str


class ImageBlock(TypedDict):
    type: Literal["image"]


class ChatMessage(TypedDict):
    role: str
    content: str | list[TextBlock | ImageBlock]
    tool_calls: NotRequired[list[ToolCall]]
    tool_call_id: NotRequired[str]


class ImageProcessor(Protocol):
    size: dict[str, int]


class Processor(Protocol):
    tokenizer: PreTrainedTokenizerBase
    image_processor: ImageProcessor
    image_token: str

    def replace_image_token(self, image_inputs: TensorInputs, image_idx: int) -> str: ...

    def apply_chat_template(self, conversation: list[ChatMessage], *, tokenize: Literal[False],
                            enable_thinking: bool, add_generation_prompt: bool,
                            tools: list[ToolSchema] | None) -> str: ...

    def __call__(self, *, text: list[str], images: list[Image] | None,
                 return_tensors: Literal["pt"], add_special_tokens: bool,
                 padding: Literal[False], truncation: Literal[False]) -> TensorInputs: ...

    def save_pretrained(self, save_directory: str | Path) -> list[str]: ...


class Rejection(TypedDict):
    path: str
    reason: str
    id: NotRequired[str]
    line: NotRequired[int]
    stage: NotRequired[str]
    too_long: NotRequired[bool]


class DataReport(TypedDict):
    objects: int
    samples: dict[str, int]
    rejected: int
    assignments: dict[str, str]


class AuditReport(DataReport):
    rejections: list[Rejection]


class PreviewItem(TypedDict):
    id: str
    source: NotRequired[str]
    line: NotRequired[int]
    prompt: NotRequired[str]
    target: NotRequired[str]
    images: NotRequired[list[str]]
    tokens: NotRequired[int]
    target_tokens: NotRequired[int]
    error: NotRequired[str]


class ManifestPayload(TypedDict):
    config: JsonObject
    sources: dict[str, str]
    versions: dict[str, str]
    code: dict[str, str]
    targets: dict[str, list[str]]


class Manifest(ManifestPayload):
    fingerprint: str
