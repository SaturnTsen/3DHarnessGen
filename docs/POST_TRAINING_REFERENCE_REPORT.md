# Agent 后训练技术结论

本文记录本项目采用的技术方向和参考模块，不记录调研过程。数据交付以 [轨迹交付规范](POST_TRAIN_TRAJECTORY_CONTRACT.md) 为准；后续开发以 [最小改动计划](POST_TRAINING_IMPLEMENTATION_PLAN.md) 为准。独立 v2 格式校验器已实现，训练与服务新增接口尚未实现。

## 1. 技术选择

- 使用已有教师轨迹做 SFT（监督微调）：根据当时可见的历史，学习当前回复或工具调用。
- 使用 Qwen 自带的 Processor、tokenizer 和 chat template 处理图片、文字及特殊标记。模型架构以本地 `config.json` 为准，不以目录名称推断。
- 使用 PEFT 管理 LoRA，使用 TRL `SFTTrainer` 执行训练。自己的代码控制轨迹选择、工具适配、图片关联和标签。
- Single-view、Multi-view 继续学习完整脚本，不恢复教师内部工具执行过程；ActiveVisual、Full3DInteraction 学习原执行方式下的工具调用与最终回复。四种场景分别训练。
- 保留 `messages` 为主体的归档格式，增加非空 `tools: list[ToolContext]`（`ToolContext = list[Tool]`，每组非空）和必要元数据，允许 tool 消息包含图片。它是项目归档格式，不是保证可原样提交的 Chat Completions 请求。
- 不增加 Responses 中转层，也不另存重复的训练对话。数据模块在内存中选择上下文与目标，模板模块直接适配 Qwen。
- 推理复用现有 vLLM 服务和 Qwen 工具解析器；训练和服务请求共用工具构建与协议适配。服务返回调用，执行环境执行调用，二者分工明确。

## 2. 可借鉴模块及输入输出

下表的输入输出描述参考代码的接口，不是要求项目照搬其数据格式。

| 来源及模块 | 输入 | 输出 | 采用方式 |
| --- | --- | --- | --- |
| SWE-smith `get_messages` | SWE-agent 轨迹字典，含 `trajectory[].query/messages` | `list[dict]` 对话消息 | 借鉴日志与训练对话分离；不采用从最后请求近似恢复全部历史的假设 |
| SWE-smith `process_single_trajectory` | 原始轨迹、评估报告、转换函数 | 消息及任务 ID、轨迹 ID、模型和执行结果 | 借鉴来源、状态与对话分离；不搬软件工程环境和重新导出层 |
| ms-swift agent template 工具构建 | 工具 schema、system 文本 | 模型格式的工具说明文本 | 参考职责划分；实际序列化优先复用 Qwen 自带模板，避免双重注入 tools |
| ms-swift 工具调用格式化与解析 | 调用结构／模型输出文本 | 模型调用文本／结构化函数调用 | 参考双向接口；本项目输出解析优先复用已安装的 vLLM Qwen 解析器 |
| Qwen-3D 答案标签构造代码 | token ID、答案边界 | 目标区间保留 token ID，其余为 `-100` 的 labels | 借鉴显式目标监督；不复制固定偏移定位方式 |
| Qwen-3D LoRA 与冻结代码 | 模型、LoRA 配置、冻结配置 | PEFT 模型与可训练参数集合 | 继续复用 PEFT，不复制其修改过的 Qwen 架构和空间任务模块 |
| TRL `SFTTrainer` | 模型、配置、Dataset、批次整理函数 | 训练指标、参数更新、checkpoint | 保留现有后端，数据与标签由项目提供 |
| vLLM `parse_chat_messages`、`safe_apply_chat_template` | 规范化后的消息、工具、模型配置 | 模板消息、多模态数据、渲染文本或 token | 用于与服务端一致的预处理与模板渲染 |
| vLLM `Qwen3EngineToolParser` | 输出文本/token 增量、请求 tools | `content`、结构化 `tool_calls` 或流式增量 | 直接复用，不自写 XML 正则解析器 |

不引入 Molmo 预处理器：图片编码必须与 Qwen 匹配。Qwen-3D 使用 Detectron2 训练循环服务于分割、定位和生成等联合任务；本项目不增加这些任务，不引入 Detectron2。SWE-Gym 的 Modal/TorchTune 启动封装不适合现有 Pixi/TRL 流程，仅作为实验参考。verl、rLLM 留待在线强化学习阶段评估。

## 3. 数据与训练不变量

- 先按对象划分，再选训练目标；保持原始会话边界，不混合独立执行过程。
- 仅监督当前目标。system、developer、用户输入、历史回复、工具结果、图片及补齐位置不计算误差。
- Step 以必填 `tool_context_index` 引用完整工具上下文，来源由 `metadata.tools_provenance` 同位置提供；先选出当前目标之前的消息及该组工具，再进行角色规范化；绝不提前注入未来指令、截图、工具定义或结果。
- 图片跟随所属消息，以 JSONL 目录解析路径，保留顺序与重复位置。缺失、损坏和过长样本必须报告，不伪造、不删图继续、不截断。
- `exec` 内的代码保持原文。custom 调用可包装成目标模型支持的 function，但不能改写为直接 Blender 调用。
- 原始工具定义由数据合作者提供；训练侧不从调用参数反推定义，不将当前配置文件冒充历史事实。
- 工具集合及其来源必须完整且非空；未知或无工具的记录不能交付为 v2，包括旧版无工具脚本视图。消息输入边界仍可标 unknown，但不进入训练。执行结果未知单独记 UNKNOWN，不能解释为成功。

## 4. 本地适配依据与验证边界

2026-10-21 核对环境：vLLM 0.30.0、Transformers 5.17.0、TRL 1.13.0、PEFT 0.21.2；本地模型配置为 `qwen3_5` / `Qwen3_5ForConditionalGeneration`。

已通过无模型权重的合成样例检查：

- 规范化后，Chat Completions 与 Responses 的同义函数调用输入可以得到相同模板文本和文字 token；因此不需要为训练增加 Responses 中转。
- 本地 Qwen 模板不直接接受 developer；vLLM 渲染层会转换为 system 并合并。此操作只能作用于当前样本已经选定的可见上下文。
- 本地 vLLM 不直接接受原始 custom 调用，需要项目适配成 function。
- 本地 vLLM 消息解析接受 tool 消息中的图片，保持文字、图片和重复出现的顺序；Qwen Processor 能编码该输入。
- 工具定义必须显式传给模板的 `tools` 参数，不能只保存进报告。
- Qwen 工具解析器能将合成的完整 exec 调用解析为 function 参数，并保留其中代码的换行和原文。

以上是解析与编码检查，不代表完整 vLLM 引擎推理、训练后任务质量或 27B 训练已经验证。输出解析的完整测试集、流式解析、训练/服务图像预算一致性属于实现计划中的独立验收项。

## 5. 参考源码

- [SWE-smith 轨迹转换](https://github.com/SWE-bench/SWE-smith/blob/main/swesmith/train/traj_mgr/utils.py)、[轨迹收集](https://github.com/SWE-bench/SWE-smith/blob/main/swesmith/train/traj_mgr/collect_trajs.py)
- [SWE-Gym 训练配置](https://github.com/SWE-Gym/SWE-Gym/tree/main/scripts/training/openhands/configs/policy)
- [ms-swift Qwen 工具模板](https://github.com/modelscope/ms-swift/blob/main/swift/agent_template/qwen3_coder.py)
- [Qwen-3D 模型与标签](https://github.com/ll220/qwen3d/blob/main/qwen3d/model.py)、[Detectron2 训练接入](https://github.com/ll220/qwen3d/blob/main/train.py)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [vLLM 消息处理](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/entrypoints/chat_utils.py)、[HF 模板渲染](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/renderers/hf.py)、[Qwen 解析器](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/parser/qwen3.py)

上游主分支可能变化；实现时以 Pixi 锁定版本与契约测试为准，不直接复制上游主分支代码。
