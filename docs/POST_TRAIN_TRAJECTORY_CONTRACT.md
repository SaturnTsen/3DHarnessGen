# post_train_trajectory 交付规范 v3

本文以当前 [`tools/validate_trajectory.py`](../tools/validate_trajectory.py) 的字段定义和关联校验为准。归档保存消息、工具定义和训练目标边界，不是可直接发送给模型服务的请求体。格式校验通过不代表训练读取器、模板或模型已经支持该记录。

## 1. 文件布局与场景

```text
3DHarnessBench/runs/<setting>/<teacher-model>/<instance>/post_train_trajectory/
  trajectory.jsonl
  images/<文件名>
```

UTF-8 JSONL 每行保存一个完整会话对象；`.json` 可保存单个展开对象。独立重试或重新启动的会话分别记录，同一对象可以有多个会话。不另交付拆好的训练样本、token IDs 或模型专用格式。

| `metadata.scene` | `metadata.view` | 工具约束 |
| --- | --- | --- |
| `Single-view`、`Multi-view` | `script` | 顶层 `tools` 必须为 `[]`，消息不得包含工具声明、工具调用或工具结果，步骤目标必须为 `script` |
| `ActiveVisual`、`Full3DInteraction` | `agent` | 顶层 `tools` 必须非空，每个工具上下文也必须非空；步骤历史必须包含恰好一条工具声明 |

交互场景保留日志中的 `developer` 指令角色，不接受 `system` 消息。脚本场景可以使用 `system`。

训练侧不执行日志中的代码，不修改原始轨迹、图片或模型。图片以输入 JSON/JSONL 所在目录解析，保持出现顺序和重复位置；缺失图片交由数据修复流程处理。

## 2. 顶层字段与通用类型

顶层只接受以下五个字段，均必填：

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `schema_version` | integer | 固定为 `3`，不接受布尔值、字符串或其他版本 |
| `model` | 非空 string | 教师模型标识 |
| `messages` | 非空 array | 普通消息或 `AdditionalTools` 工具声明，按归档顺序保存 |
| `tools` | `list[ToolContext]` | 每个 `ToolContext` 是 `list[Tool]`；是否允许外层为空由场景决定 |
| `metadata` | object | 场景、来源和步骤，见第 5 节 |

校验器使用严格类型并禁止模型未定义的字段。`Index` 是非负整数，不接受布尔值、负数或浮点数；`Nonempty` 是长度至少为 1 的字符串。工具参数中的 JSON Schema 是 JSON 对象，不受模型字段白名单限制。

`ToolContext` 只是工具列表的类型别名，没有额外包装对象。`tools: [[]]`、`tools: null`、`tools: [null]` 和旧版平铺工具列表均不符合当前结构。

## 3. 消息、调用与图片

### 3.1 普通消息 `Message`

| 字段 | 类型 | 必填与约束 |
| --- | --- | --- |
| `index` | Index | 必填，必须等于在 `messages` 中的位置，从 0 开始 |
| `role` | string | 必填，取 `system`、`developer`、`user`、`assistant`、`tool` |
| `content` | string、内容块数组或 null | 必填；只有带工具调用的 assistant 可以为 null |
| `tool_calls` | 调用数组或 null | 可省略；非 null 时仅限 assistant，且数组非空 |
| `tool_call_id` | 非空 string 或 null | tool 消息必须为非空字符串；其他角色只能省略或为 null |

内容块只接受以下两种结构：

```json
{"type":"text","text":"按参考图重建"}
```

```json
{"type":"image_url","image_url":{"url":"images/reference.png","detail":"auto"}}
```

`image_url.url` 必填，必须为相对的 `images/...` 路径，或清洗数据使用的 `imgs/...` 路径；不允许绝对路径、`..` 路径段、反斜杠、远程 URL 或 base64。`detail` 可省略或为 null；非 null 时取 `auto`、`low`、`high`。图片只允许出现在 user 和 tool 消息，其他角色的内容块只能是文字。

原始归档保留图片重复位置和错误输出，工具结果不改成 user，不补造成功回复。用户授权的派生清洗数据可以按明确规则移除失败调用及其返回，但须保留来源和索引映射；剩余图片出现顺序与重复位置不变。推理事件不混入普通 assistant 答案，原始证据保留在来源日志中；当前元数据没有额外的排除事件字段。

### 3.2 工具调用与结果

Function 调用的 `arguments` 必须是能解析成 JSON 对象的字符串；custom 调用的 `input` 保留自由文本原文：

```json
{"id":"call_1","type":"function","function":{"name":"wait","arguments":"{\"cell_id\":\"demo\"}"}}
```

```json
{"id":"call_2","type":"custom","custom":{"name":"exec","input":"text('example');"}}
```

调用对象的 `id`、`type` 和对应的 `function` 或 `custom` 都必填，内部名称必须非空。不得互换调用类型，尤其不能把 `exec` 中的代码改成直接调用内部工具。

同一会话的调用 ID 唯一。每条 tool 消息的 `tool_call_id` 必须指向之前已有的调用，同一调用最多有一条结果消息；当前校验器不要求每次调用最终都有结果。多个返回内容块按原顺序保存在同一结果消息中，异步 exec 和后续 wait 的返回分别属于各自调用。

Astra 数据按原始 session log 将 assistant 文字和随后的工具调用分别保存为两条 assistant 消息；调用消息使用 `content: null` 和现有 `tool_calls` 格式，工具返回保持原有 tool 消息结构。

### 3.3 工具声明 `AdditionalTools`

工具声明与普通消息共用 `messages` 数组及连续的 `index`，只有以下四个必填字段，没有 `content`：

| 字段 | 值或类型 |
| --- | --- |
| `index` | Index |
| `type` | 固定 `additional_tools` |
| `role` | 固定 `developer` |
| `tools` | 非空的 `ToolNameSpace` 数组 |

每个 `ToolNameSpace` 必须包含 `type: "namespace"`、非空的 `name`、字符串 `description` 和非空的 `tools` 数组。说明可以为空字符串。内部工具必须显式声明 `type: "function"` 或 `type: "custom"`，不允许嵌套命名空间，也不允许把工具直接放在 `AdditionalTools.tools` 外层。

```json
{
  "index": 0,
  "type": "additional_tools",
  "role": "developer",
  "tools": [{
    "type": "namespace",
    "name": "functions",
    "description": "示例工具",
    "tools": [{
      "type": "custom",
      "name": "exec",
      "description": "示例执行入口",
      "format": {"type": "text"}
    }]
  }]
}
```

## 4. 工具定义与顶层工具上下文

工具定义的 `name` 和 `description` 必填，名称非空，说明允许为空。各类型的其余字段如下：

| 类型 | 必填内容 | 允许的其他字段 |
| --- | --- | --- |
| function | `parameters` 为 JSON 对象，且其中 `type` 为 `object` | `strict` 可省略、为布尔值或 null |
| custom | `format` 为格式对象 | 不接受 `parameters`、`strict` 或 `tools` |

`format` 接受 `{"type":"text"}`，或 `{"type":"grammar","syntax":"lark","definition":"原始语法"}`。grammar 的 `syntax` 只能是 `lark` 或 `regex`，`definition` 必须为非空字符串。原始参数、说明、strict 和 grammar 按日志保留，不为适配而改写。

顶层 `tools` 中的每个上下文，是将一条 `additional_tools` 按命名空间顺序、内部工具顺序展开后的列表。每个 `Tool` 包含：

| 字段 | 必填与约束 |
| --- | --- |
| `type` | 必填，`function` 或 `custom` |
| `function` / `custom` | 必须且只能出现与 type 对应的定义；另一字段即使写 null 也不接受 |
| `namespace` | 可省略或为 null；非 null 时为非空字符串 |
| `namespace_description` | 可省略、为字符串或 null |
| `index` | 可省略或为 null；非 null 时为 Index |

外层 `type` 已说明工具类型，因此内部定义通常省略 `type`；如果显式填写，必须与外层一致。展开工具声明时，命名空间名称和说明分别写入 `namespace`、`namespace_description`。

同一上下文只要有一个工具的 `index` 非 null，所有工具就必须提供从 0 开始且等于局部位置的 index。不同上下文可以重复使用工具名称，但同一上下文内的工具名称不能重复，即使属于不同命名空间也不行。调用按工具的裸名称匹配，例如 `exec`。

每条工具声明展开后，必须与顶层某个工具上下文一致。比较忽略工具的局部 `index`，其余模型字段及工具顺序参与比较。工具集合相同但说明、参数、grammar 或命名空间不同，仍是不同上下文。

顶层 `tools` 保存可核对的完整工具上下文；每个步骤实际使用哪一组工具，由该步骤历史中唯一的 `additional_tools` 决定，不需要步骤上的工具索引字段。历史发现结果中的接口说明留在原消息位置，不提前合并进旧工具定义。

## 5. 元数据与步骤

`metadata` 只接受以下六个必填字段：

| 字段 | 类型与含义 |
| --- | --- |
| `scene` | 第 1 节列出的四种场景之一 |
| `instance` | 非空字符串，对象名称 |
| `session_id` | 非空字符串，会话标识 |
| `view` | `script` 或 `agent`，必须与 scene 对应 |
| `sources` | 来源对象数组，允许为空 |
| `steps` | 步骤数组，允许为空 |

来源对象 `Source` 的三个字段均必填：

| 字段 | 约束 |
| --- | --- |
| `path` | 非空相对路径，允许 `../`，禁止绝对路径、冒号和反斜杠 |
| `sha256` | 恰好 64 位十六进制字符串，大小写均可 |
| `locator` | 非空字符串，例如行号、JSON 指针或事件 ID |

来源路径以当前 JSON/JSONL 所在目录为起点，用于审计，不是图片路径。校验器只检查这些字段的格式，不读取来源文件或核对摘要。

每个 `Step` 只接受以下六个必填字段：

| 字段 | 类型与约束 |
| --- | --- |
| `id` | 非空字符串，会话内唯一 |
| `context_indices` | Index 数组，严格递增且均小于 target_index；允许不连续 |
| `target_index` | Index，必须指向现有的普通 assistant 消息；同一目标只能出现一次 |
| `target_kind` | `script`、`tool_call`、`final_reply` 或 `status` |
| `status` | `OK`、`ERROR` 或 `UNKNOWN` |
| `source` | 单个来源对象 |

带工具调用的目标必须标记为 `tool_call`，`tool_call` 目标也必须有调用。脚本场景只接受 `script` 目标。`status` 表示记录的执行状态，校验器只检查枚举值，不据此确认执行成功或训练可用。

交互场景还要求：

- `context_indices` 中恰好包含一条 `additional_tools`；没有声明或同时包含多条都报错。
- 当前目标调用的名称和类型必须属于这条声明展开后的工具列表。
- 历史工具结果必须有同样出现在可见历史中的先前调用。历史调用不要求属于当前工具集合。

步骤历史不得包含目标本身或之后的消息、工具结果、截图和最终产物。输入不一定是全部消息前缀，恢复或压缩后的历史选择需要原始证据支持。

当前 `Metadata` 不接受 `tools_provenance`、`excluded_events`，当前 `Step` 不接受 `tool_context_index`、`input_provenance`、`available_tools`、`available_tool_indices` 或步骤级 `tools_provenance`。即使旧记录的 `schema_version` 已为 3，只要包含这些字段，仍会被拒绝。

## 6. 输入选择与示例

以下代码演示对已经校验过的交互记录选择输入，`additional_tools` 本身也是归档中的历史条目：

```python
from tools.validate_trajectory import AdditionalTools

step = trajectory.metadata.steps[0]
visible_items = [trajectory.messages[i] for i in step.context_indices]
declaration = next(item for item in visible_items if isinstance(item, AdditionalTools))
visible_tools = declaration.tool_context()
target = trajectory.messages[step.target_index]
```

脚本场景没有工具声明，输入直接按 `context_indices` 选择，工具列表为空。模板负责把归档对象转换为模型输入，去除归档 index，并区分可见历史与当前监督目标。

完整且可校验的交互、脚本结构示例见[示例说明](TRAJECTORY_CONTRACT_EXAMPLE.md)。迁移旧数据时，应依据原始请求恢复命名空间、工具定义和可见声明，再调整字段；本规范不授权直接修改已有原始轨迹。

## 7. 校验与验收边界

在项目根目录运行：

```bash
pixi run validate-trajectory path/to/trajectory.json
pixi run validate-trajectory path/to/trajectory.jsonl
```

校验器先严格解析 JSON，拒绝重复键和 NaN、Infinity 等非标准常量，再通过 `pydantic_yaml` 创建 Pydantic 类型化对象。JSONL 空行会作为无效记录报错，空文件也不通过。同一文件中 `(scene, instance, session_id)` 重复的有效记录被拒绝。

通过时退出码为 0；记录校验失败时退出码为 1，并报告记录/行号及原因。格式检查覆盖字段类型、场景与视图、消息和工具 index、命名空间与上下文一致性、步骤时序、当前目标工具可用性、调用 ID、结果关联、图片路径和来源摘要格式。

它不检查图片存在或解码、不读取来源文件、不核对摘要真实性、不核对目录身份、不确认历史是否真实、不验证完整 JSON Schema 参数约束，也不加载 Processor、模型权重或执行工具。`sources`、`steps` 为空仍可能通过格式检查；格式通过不能作为数据完整或可训练的证明。

训练验收仍需先按对象划分，再选择目标；仅监督当前回复，历史、用户输入、工具结果、图片和补齐位置不计算误差。缺失图片和过长完整脚本须报告并按配置跳过或报错，不能静默删图或截断。
