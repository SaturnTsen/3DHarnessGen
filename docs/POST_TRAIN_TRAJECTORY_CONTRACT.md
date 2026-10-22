# post_train_trajectory 交付规范 v2

本文是数据合作者与训练代码之间的交付接口。采用 messages 归档格式，保留原始工具调用语义；不是可直接发送的 Chat Completions API 请求体。v2 将完整工具上下文集中存入 `tools`，每个 Step 用 `tool_context_index` 显式引用。旧版平铺工具池、步骤内覆盖和工具子集索引均不再接受。

本次实现的是独立格式校验器。训练读取器、Qwen 模板和服务请求构建尚未接入 v2；不得把格式校验通过等同于可训练。

## 1. 分工与文件布局

合作者负责从原始日志和对应运行环境恢复工具定义、消息、图片、请求边界及来源。训练侧负责校验、对象划分、目标选择和模型适配，不猜测工具定义、不执行日志代码、不修改原始轨迹或图片。

```text
3DHarnessBench/runs/<setting>/<teacher-model>/<instance>/post_train_trajectory/
  trajectory.jsonl
  images/<文件名>
```

- 场景为 Single-view、Multi-view、ActiveVisual、Full3DInteraction。
- UTF-8 JSONL，一行一个完整会话对象。独立重试、重新启动的会话分别交付；同一对象可以有多个会话。
- `.json` 可保存一个展开的对象用于查看和校验；正式交付使用 `.jsonl`。
- 不另交付 token IDs、Qwen XML 或拆好的训练样本，训练侧在内存中选择输入和目标。
- 图片相对当前 JSON/JSONL 文件目录解析，使用 `images/...`，禁止绝对路径、`..`、远程 URL 和 base64。图片须随数据交付。
- 四种场景均遵守非空工具上下文约束。单图、多图可选择完整脚本为目标，但仍须保存该请求实际可用的工具。确实没有工具或工具定义未知的请求不符合 v2，不能用空列表、null、占位工具或其他请求的定义补齐。

## 2. 顶层字段与类型

所有表中字段必须显式存在。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `schema_version` | integer | 固定 `2`，不接受布尔值或旧版本 |
| `model` | string | 教师模型标识 |
| `messages` | 非空消息数组 | 按归档顺序保存 |
| `tools` | `list[ToolContext]` | 非空的完整工具上下文列表，每个上下文也必须非空 |
| `metadata` | object | 来源、场景、步骤和排除记录 |

Python 中显式定义类型别名：

```python
Tool = Annotated[FunctionTool | CustomTool, Field(discriminator="type")]
ToolContext = list[Tool]
```

`ToolContext` 是工具列表的别名，不是新增的包装对象。非空要求由校验器执行。`tools: []`、`tools: [[]]`、`tools: null`、`tools: [null]` 和旧版平铺 `list[Tool]` 均拒绝。

空列表禁令针对 `tools` 及其每个 ToolContext；没有排除事件时 `excluded_events: []` 仍合法，工具自身 JSON Schema 中的空 `required` 等也不受影响。

## 3. 消息、工具与图片

### 3.1 消息

消息必有 `index`、`role`、`content`。`index` 从 0 连续递增，必须等于消息在 `messages` 中的位置；它用于检查，不用于重新排序，构建模型输入时去掉。

角色允许 system、developer、user、assistant、tool。`content` 为字符串或有序内容块数组；仅带工具调用的 assistant 可为 null。内容块仅支持 text 与 image_url：

```json
{"index":1,"role":"user","content":[{"type":"text","text":"按参考图重建"},{"type":"image_url","image_url":{"url":"images/reference.png","detail":"auto"}}]}
```

`detail` 可省略，原请求存在则保留。system、developer、assistant 限文字，图片仅在 user 和 tool 中出现。不支持的音频、视频或文件内容须报告，不能静默删除。

Function 调用保留参数的 JSON 对象字符串；custom 调用保留自由文本原文：

```json
{"index":2,"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"custom","custom":{"name":"exec","input":"text('example');"}}]}
```

```json
{"index":3,"role":"tool","tool_call_id":"call_1","content":[{"type":"text","text":"渲染完成"},{"type":"image_url","image_url":{"url":"images/render.png"}}]}
```

- `tool_calls` 只允许出现在 assistant 中，且必须非空。调用 ID 在会话内唯一，调用顺序保留。
- 一个调用最多对应一条逻辑结果消息，`tool_call_id` 指向之前已有的调用。同次返回的多个块按原顺序合并；异步 exec 与后续 wait 的返回分别属于各自调用。
- 工具结果不能改成 user；旧数据中 `name: tool_output` 的内容须根据原始证据归属到正确调用，不能靠邻接关系猜测。
- 图片保留所有出现位置和重复引用，不去重。完整原文保留，不删错误、不补成功回复。
- reasoning 不混入普通答案；已归档的推理事件保留在原日志，并在 `excluded_events` 记录排除原因。

### 3.2 工具定义与完整上下文

Function 定义：

```json
{"type":"function","function":{"name":"函数名","description":"原工具说明","parameters":{"type":"object","properties":{},"required":[]}}}
```

Custom 定义：

```json
{"type":"custom","custom":{"name":"exec","description":"原执行环境完整说明","format":{"type":"text"}}}
```

原工具带 grammar 时必须保留 `{"type":"grammar","syntax":"lark或regex","definition":"原始语法"}`。训练侧不支持时明确拒绝，不改成 text。function 的 `strict` 属性若有来源也须保留。

每个 `tools[i]` 包含该次请求可见的全部工具及完整定义，保留原顺序；不是增量、工具子集索引或单个 exec 的引用。集合不变但说明、参数、grammar 或命名空间变化，也构成不同上下文。同一上下文内工具名不能重复；不同上下文允许存在同名工具的不同版本。多个 Step 可以复用同一上下文，前提是完整定义和可见性一致。

原生命名空间及其说明分别保存在可选的 `namespace`、`namespace_description`。工具可带冗余 `index`；同一上下文一旦使用该字段，所有工具必须提供它，并从 0 开始等于在该上下文中的位置。此字段不是 Step 的引用目标。构建服务请求时去掉归档 index 并恢复命名空间分组。

`exec` 的动态拼接说明如果属于原工具定义，保留在对应的 `custom.description`；内部工具说明原本出现在 system/developer 或发现结果中，就保留在对应消息位置。不能遗漏教师依赖的接口说明，也不能把后续发现的接口提前写入旧工具上下文。

## 4. 元数据与步骤

| metadata 字段 | 类型 | 含义 |
| --- | --- | --- |
| `scene` / `instance` | string | 正式交付须与目录一致 |
| `session_id` | string | 对象内唯一的稳定会话标识，不用行号替代 |
| `view` | string | 单图、多图为 script；交互场景为 agent |
| `sources` | array | 会话的来源列表 |
| `tools_provenance` | 非空 array | 与 `tools` 等长，位置 i 记录 `tools[i]` 的来源 |
| `steps` | array | 可选择的 assistant 目标及可见上下文 |
| `excluded_events` | array | `{"source":来源对象,"reason":"原因"}`；无排除为 `[]` |

来源对象固定为 `{"path":"相对当前JSON/JSONL目录的来源路径","sha256":"64位十六进制摘要","locator":"行号、JSON指针或事件ID"}`。来源路径允许 `../`，用于审计，不是图片路径，不能执行来源内容。

每个 `tools_provenance[i]` 是 `{"status":"recorded或recovered","sources":[来源对象]}`，sources 必须非空。recorded 表示直接记录，recovered 表示有对应版本运行环境或原始证据支持的恢复。v2 不接受工具来源 unknown、not_applicable 或 null；未知记录留在原始数据和检查报告中，不伪装为符合契约的记录。

每个 Step 对应一个 assistant 目标，字段均显式填写：

```json
{
  "id": "step_1",
  "context_indices": [0, 1],
  "tool_context_index": 0,
  "target_index": 2,
  "target_kind": "tool_call",
  "status": "UNKNOWN",
  "input_provenance": "recorded",
  "source": {"path":"../raw/session.jsonl","sha256":"实际64位摘要","locator":"line:42"}
}
```

- `context_indices` 指向 messages，严格递增且均小于 `target_index`；target 必须是 assistant。不能假定输入总是全部消息前缀，压缩、重试、恢复均须有证据。
- `tool_context_index: Index` 是必填非负整数，范围为 `0 <= index < len(tools)`；不能为 null、布尔值、负数或浮点数。没有继承或默认工具集合。
- 当前请求的工具唯一取自 `trajectory.tools[step.tool_context_index]`，来源取自同位置的 `metadata.tools_provenance`。不合并其他上下文，也不从调用样例补定义。
- `available_tools`、`available_tool_indices`、步骤级 `tools_provenance` 已删除，出现即报错。
- `target_kind` 为 script、tool_call、final_reply、status。单图、多图只交付 script 目标。带调用的回复必须为 tool_call，说明与调用保持在同一目标中。
- 只校验当前目标的调用是否属于所选上下文且类型匹配。历史调用可能来自先前工具上下文，不能要求它仍在当前工具列表中。上下文中的工具结果仍须有可见的先前调用。
- `input_provenance` 为 recorded、reconstructed、unknown。重建输入须有证据；unknown 不进入训练。
- `status` 为 OK、ERROR、UNKNOWN，只描述可核验执行结果，不表示重建质量。不能把未知解释为成功或失败。
- 目标之后的消息、工具结果、截图和产物不能进入该目标输入；最终产物可以用于筛选。

构建一步输入的关系为：

```python
messages = [trajectory.messages[i] for i in step.context_indices]
tools = trajectory.tools[step.tool_context_index]
target = trajectory.messages[step.target_index]
```

这里得到的是归档对象；后续模板适配负责去掉冗余 index、规范化调用并生成模型输入。

## 5. 示例与迁移

完整 JSON 和来源边界见[示例说明](TRAJECTORY_CONTRACT_EXAMPLE.md)。示例演示两个非空 ToolContext、多个 Step 共享上下文，以及后续 Step 切换上下文。

v2 是破坏性变更，`schema_version` 从 1 升为 2。迁移必须依据各次请求：确定完整工具集合 → 保存为 ToolContext → 按相同位置保存来源 → 为每个 Step 填写索引 → 删除旧选择字段。旧工具池的全局 index 不能直接当作工具上下文索引；如保留工具自身 index，须按各上下文内位置重新编号。

旧版无工具脚本视图、unknown 工具记录不能直接迁移。可以在证据完整时恢复真实非空上下文；无法恢复就保留原始记录并报告，不制造占位工具。

## 6. 校验与验收边界

```bash
pixi run validate-trajectory docs/examples/bathroom_sink/trajectory.json
pixi run validate-trajectory path/to/trajectory.jsonl
```

校验器 `tools/validate_trajectory.py` 使用 Pydantic 定义类型与关系，严格解析 JSON，再用 pydantic_yaml 创建类型化对象。`.json` 接受单个对象，`.jsonl` 一行一个对象。成功退出码 0，错误退出码 1，报告记录/行号和错误位置。

格式检查覆盖字段、非空工具上下文、来源数量与摘要格式、消息及工具局部 index、步骤引用范围、当前目标工具可用性、调用 ID、结果关联、上下文时序、图片路径格式和同文件重复会话标识。

它不读取图片或来源文件、不验证摘要真实性、不确认请求快照与实际历史一致、不验证完整 JSON Schema 参数约束，也不加载 Processor、权重或执行工具。图片存在性、解码、目录身份、来源真实性和训练可用性仍需独立检查。

正式验收先按对象划分，再选择目标；仅监督当前回复，历史、工具结果、图片和补齐位置不计算误差。缺失图片和过长完整脚本须报告并按配置跳过或报错，不能静默删图或截断。各场景先验收代表对象，再批量补齐。
