# 轨迹 v3 格式示例

对应[交付规范 v3](POST_TRAIN_TRAJECTORY_CONTRACT.md)，字段与数据模块的 [`contract.py`](../harness_training/data/contract.py) 一致；原独立校验脚本按用户要求冻结，仍只支持旧格式。下面两个完整 JSON 对象分别演示交互场景和脚本场景，可以各自保存为 `.json` 后运行格式校验。

**示例全部为合成数据，仅用于说明结构，不能用于训练。** 工具说明、消息、来源路径和摘要都是演示值；格式通过不证明这些来源或调用真实存在。代码文本只作为数据保存，不执行。

## 1. 交互场景：从可见工具声明确定工具

此例有 7 个消息条目和 3 个目标。消息 0 是工具声明，包含一个命名空间；顶层 `tools[0]` 是它展开后的完整工具列表。

```json
{
  "schema_version": 3,
  "model": "demo-teacher",
  "messages": [
    {
      "index": 0,
      "type": "additional_tools",
      "role": "developer",
      "tools": [
        {
          "type": "namespace",
          "name": "functions",
          "description": "合成示例工具",
          "tools": [
            {
              "type": "custom",
              "name": "exec",
              "description": "合成的自由文本执行入口",
              "format": {
                "type": "text"
              }
            },
            {
              "type": "function",
              "name": "wait",
              "description": "合成的等待入口",
              "strict": false,
              "parameters": {
                "type": "object",
                "properties": {
                  "cell_id": {
                    "type": "string"
                  }
                },
                "required": [
                  "cell_id"
                ],
                "additionalProperties": false
              }
            }
          ]
        }
      ]
    },
    {
      "index": 1,
      "role": "user",
      "content": "演示一次工具调用和等待。"
    },
    {
      "index": 2,
      "type": "custom_tool_call",
      "call_id": "call_exec",
      "name": "exec",
      "input": "text('demo');"
    },
    {
      "index": 3,
      "type": "custom_tool_call_output",
      "call_id": "call_exec",
      "output": "演示任务仍在运行，cell_id=demo。"
    },
    {
      "index": 4,
      "type": "function_call",
      "call_id": "call_wait",
      "name": "wait",
      "arguments": "{\"cell_id\":\"demo\"}"
    },
    {
      "index": 5,
      "type": "function_call_output",
      "call_id": "call_wait",
      "output": [
        {
          "type": "input_text",
          "text": "演示结果。"
        },
        {
          "type": "input_image",
          "image_url": "images/demo.png",
          "detail": "auto"
        }
      ]
    },
    {
      "index": 6,
      "role": "assistant",
      "content": "示例调用结束。"
    }
  ],
  "tools": [
    [
      {
        "index": 0,
        "namespace": "functions",
        "namespace_description": "合成示例工具",
        "type": "custom",
        "custom": {
          "name": "exec",
          "description": "合成的自由文本执行入口",
          "format": {
            "type": "text"
          }
        }
      },
      {
        "index": 1,
        "namespace": "functions",
        "namespace_description": "合成示例工具",
        "type": "function",
        "function": {
          "name": "wait",
          "description": "合成的等待入口",
          "strict": false,
          "parameters": {
            "type": "object",
            "properties": {
              "cell_id": {
                "type": "string"
              }
            },
            "required": [
              "cell_id"
            ],
            "additionalProperties": false
          }
        }
      }
    ]
  ],
  "metadata": {
    "scene": "Full3DInteraction",
    "instance": "DemoObject",
    "session_id": "demo-agent",
    "view": "agent",
    "sources": [
      {
        "path": "../raw/demo.jsonl",
        "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "locator": "demo:session"
      }
    ],
    "steps": [
      {
        "id": "step_exec",
        "context_indices": [
          0,
          1
        ],
        "target_index": 2,
        "target_kind": "tool_call",
        "status": "UNKNOWN",
        "source": {
          "path": "../raw/demo.jsonl",
          "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
          "locator": "demo:exec"
        }
      },
      {
        "id": "step_wait",
        "context_indices": [
          0,
          1,
          2,
          3
        ],
        "target_index": 4,
        "target_kind": "tool_call",
        "status": "UNKNOWN",
        "source": {
          "path": "../raw/demo.jsonl",
          "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
          "locator": "demo:wait"
        }
      },
      {
        "id": "step_reply",
        "context_indices": [
          0,
          1,
          2,
          3,
          4,
          5
        ],
        "target_index": 6,
        "target_kind": "final_reply",
        "status": "UNKNOWN",
        "source": {
          "path": "../raw/demo.jsonl",
          "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
          "locator": "demo:reply"
        }
      }
    ]
  }
}
```

各目标可见的边界如下：

| 目标消息 | 可见历史 | 可见工具声明 | 是否可见消息 5 的图片 |
| --- | --- | --- | --- |
| 2：exec 调用 | 0、1 | 消息 0 | 否 |
| 4：wait 调用 | 0–3 | 消息 0 | 否 |
| 6：最终回复 | 0–5 | 消息 0 | 是 |

工具结果分别关联 `call_exec` 和 `call_wait`，不共用调用 ID。图片仅是路径示例；格式校验不会读取 `images/demo.png`，也不会核对全零摘要。

工具声明里的工具显式包含 `type`，顶层工具上下文则用外层 `type` 与对应的 `custom` / `function` 包装定义。展开声明后，顺序、命名空间、说明、参数和格式必须与顶层上下文一致；比较时忽略工具局部 index。

## 2. 多个上下文与历史选择

对已经校验过的交互记录，可以这样选择一步的输入：

```python
from harness_training.data.contract import AdditionalTools

step = trajectory.metadata.steps[0]
visible_items = [trajectory.messages[i] for i in step.context_indices]
declaration = next(item for item in visible_items if isinstance(item, AdditionalTools))
visible_tools = declaration.tool_context()
target = trajectory.messages[step.target_index]
```

示例的三个步骤复用消息 0 的声明。若后续请求的工具定义改变，应在消息中保存新的 `additional_tools`，并在顶层 `tools` 中保存与它一致的完整上下文。新步骤的 `context_indices` 选择新的声明，排除旧声明，使可见历史仍恰好包含一条工具声明。

历史中的普通消息、调用与结果可以按实际输入保留；历史结果仍须有可见的先前调用。工具校验只要求当前目标的调用属于当前声明，不要求历史调用属于新工具集合。没有声明或同时选择两条声明都会报错，不会自动选取最近一条。

当前步骤没有 `tool_context_index` 或 `input_provenance`；当前元数据没有 `tools_provenance` 或 `excluded_events`。这些旧字段不能出现在上述 JSON 中。

## 3. 脚本场景：空工具列表

此例演示 `Single-view` 的完整脚本目标。`Multi-view` 使用同样的工具约束；实际任务可在 user 的有序内容块中保留多张图片。

```json
{
  "schema_version": 3,
  "model": "demo-teacher",
  "messages": [
    {"index": 0, "role": "system", "content": "返回完整脚本。"},
    {"index": 1, "role": "user", "content": "提供一个合成脚本示例。"},
    {"index": 2, "role": "assistant", "content": "print('demo')\n"}
  ],
  "tools": [],
  "metadata": {
    "scene": "Single-view",
    "instance": "DemoObject",
    "session_id": "demo-script",
    "view": "script",
    "sources": [{
      "path": "../raw/demo-script.jsonl",
      "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "locator": "demo:session"
    }],
    "steps": [{
      "id": "step_script",
      "context_indices": [0, 1],
      "target_index": 2,
      "target_kind": "script",
      "status": "UNKNOWN",
      "source": {
        "path": "../raw/demo-script.jsonl",
        "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "locator": "demo:script"
      }
    }]
  }
}
```

脚本场景要求 `tools: []`，没有 `additional_tools`、原生工具调用或返回事件。完整脚本是 assistant 的正文，不包装成工具调用，也不执行示例脚本。

## 4. 校验方式与已有示例文件

将第 1 节和第 3 节的完整 JSON 分别保存到工作区 `temp/` 下，例如 `trajectory_agent_example.json` 和 `trajectory_script_example.json`，在项目根目录运行：

```bash
pixi run python -c 'from pathlib import Path; from harness_training.data.contract import validate_record; validate_record(Path("temp/trajectory_agent_example.json").read_text()); print("格式检查通过")'
pixi run python -c 'from pathlib import Path; from harness_training.data.contract import validate_record; validate_record(Path("temp/trajectory_script_example.json").read_text()); print("格式检查通过")' 
```

每份示例的预期输出均为 `格式检查通过`。检查仅覆盖结构与关联关系，不读取图片或来源文件，也不加载模型。

已有 [bathroom_sink/trajectory.json](examples/bathroom_sink/trajectory.json) 和 [tool_context_demo.json](examples/bathroom_sink/tool_context_demo.json) 是先前的结构演示材料，本次只更新 Markdown，未同步修改这些 JSON。不能把本页示例的通过结果视为已有文件已符合当前验证器；已有文件需要单独校验。

真实轨迹的工具、输入边界、来源与图片必须来自可核验记录。这里的合成内容和摘要仅说明字段格式，不应被复制为真实训练数据。
