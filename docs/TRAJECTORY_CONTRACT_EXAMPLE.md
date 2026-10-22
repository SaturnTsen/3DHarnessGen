# 工具上下文索引示例

对应[交付规范 v2](POST_TRAIN_TRAJECTORY_CONTRACT.md)。完整可校验 JSON 位于 [examples/bathroom_sink/trajectory.json](examples/bathroom_sink/trajectory.json)。

**这是“真实消息片段 + 合成工具上下文”的结构演示，不是恢复完成的真实训练轨迹。** 不得用它证明教师当时看到了这些工具，也不能用于训练。

## 来源与改造范围

消息来自项目内：

```text
3DHarnessBench/runs/Full3DInteraction/gpt-5-6-sol/BathroomSinkFactory/post_train_trajectory/trajectory.jsonl
```

源记录共有 150 条消息、18 个图片位置。示例保留前 26 条消息（0–25），包含 12 个 assistant Step 和 4 个工具返回图片位置。原始文件保持不变；不删除片段内的消息，不改写其中代码或工具返回文本，不执行代码。

原导出没有完整 tools 字段，原始 rollout 也没有逐次请求的工具快照，因此不能声称已恢复真实 ToolContext。为演示索引变化，另行编写了 [tool_context_demo.json](examples/bathroom_sink/tool_context_demo.json)：

| 上下文索引 | 完整工具列表 | 示例中引用它的目标消息 |
| --- | --- | --- |
| 0 | exec、wait | 4、5、7、8、10、12、14、16、18 |
| 1 | exec、wait、demo_inspect | 20、22、24 |

其中所有工具定义及切换时点都是合成设计；`demo_inspect` 明确是虚构工具。`tools_provenance` 的 recorded 仅指这些定义直接记录在示例来源文件中，不代表原教师请求有这些定义。所有 Step 的 `input_provenance` 都是 unknown；`excluded_events` 也记录这一限制和后续消息未纳入示例的原因。

真实交付只能用原请求或可核验恢复得到的定义，不能照搬本例的工具说明和参数。

## 消息保真

- 补入从 0 开始的消息 index，不改原有顺序和文字。
- 按 call ID 对照原始 rollout，恢复 exec 的 custom 调用和自由文本输入；原导出将它包装成 function，此处依据原日志恢复，不从名称猜测。
- wait 保留原始 function 调用及 JSON 参数字符串。
- 无正文的工具调用消息使用 `content: null`。
- 按原导出的图片位置关联图片，并逐一比对原始工具输出中的图片字节；复制到示例目录的 images 下，原图片不变。
- 来源路径按示例目录重新计算，SHA-256 为实际来源文件摘要。正文和调用中原有路径属于历史文本，不执行。

## 如何构造请求

下面仅展示选择关系，不是完整 JSON：

```python
step = trajectory.metadata.steps[0]
visible_messages = [trajectory.messages[i] for i in step.context_indices]
visible_tools = trajectory.tools[step.tool_context_index]
target = trajectory.messages[step.target_index]
```

例如目标 18 使用 `tools[0]`；目标 20 使用 `tools[1]`。两者都是完整工具列表，不需要覆盖、合并或遍历全局工具池。多个 Step 可以引用同一个上下文，工具上下文数量不要求等于 Step 数量。

目标 22 调用 wait 时，只看到 0–21 的消息，不能看到它自己的结果图（消息 23）；目标 24 可以看到消息 23 的图片。历史调用按原文保留，不要求它仍属于当前请求的工具集合。

`metadata.tools_provenance[0]` 与 `tools[0]` 对应，`metadata.tools_provenance[1]` 与 `tools[1]` 对应。Step 不再携带 available_tools、available_tool_indices 或 tools_provenance。

## 校验

在项目根目录运行：

```bash
pixi run validate-trajectory docs/examples/bathroom_sink/trajectory.json
```

格式校验结果应为 `1 条通过，0 条失败`。校验覆盖结构和引用关系，不会确认合成工具属于原始教师请求。示例构造时另行核对了四张图片与原日志字节、来源摘要及原轨迹未变；这不等同于真实请求恢复、训练或完整 benchmark 验证。
