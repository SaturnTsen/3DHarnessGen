# Qwen Agent 后训练最小改动计划

进度更新：已提供契约 v3 的独立格式校验器 `tools/validate_trajectory.py`，入口为 `pixi run validate-trajectory 文件路径`。下面的训练读取器、工具适配和服务改造仍未实施。

状态：开发计划，尚未实施。结论见 [参考报告](POST_TRAINING_REFERENCE_REPORT.md)，数据接口见 [交付规范](POST_TRAIN_TRAJECTORY_CONTRACT.md)。目标是接通现有 messages 轨迹、TRL 训练与 vLLM 服务协议，保留当前分层和训练入口。

## 1. 改动范围

保留 `Trainer`、`QwenModel`、`QwenTemplate`、`SFTMethod`、`SFTCollator` 的职责；保留四场景独立配置、对象划分、长度检查、目标标签、保存恢复。只替换旧 ShareGPT 依赖并补充工具与服务协议。

最多新增两个核心实现文件：`harness_training/tools.py` 和 `harness_training/serving.py`。其余在现有模块内修改。新增验证用例放现有测试目录；不另建数据导出层、模型工厂、注册框架、HTTP 网关或 Blender 执行器。

| 模块 | 具体改动 | 保留部分 |
| --- | --- | --- |
| `types.py`、`data/base.py` | 用交付规范的消息、工具、来源及 step 类型替代 ShareGPT 类型；TrainingExample 保留记录引用、上下文索引与目标索引，补 step ID 和必填 tool_context_index，ToolContext 显式为 list[Tool] 别名 | 内存样本视图，不复制对话文件 |
| `data/trajectory.py` | 读取严格 JSONL；按 schema_version 校验，检查每条消息的 index 与数组位置一致；直接绑定 content 图片；依据 steps 选择消息、目标及 tools[tool_context_index]；检查非空上下文、同位置来源、调用和缺失定义 | 路径查找、对象划分、图片检查、拒绝报告 |
| `tools.py`（新增） | 构建 Qwen/vLLM 函数工具定义，适配 custom 调用，反向还原 runtime 调用 | 不执行工具，不构造完整训练样本 |
| `serving.py`（新增） | 封装 vLLM 消息预处理、模板渲染、输出调用解析；提供规范化服务请求构建与独立无权重验证入口 | HTTP、模型调度、流式传输全部使用原 vLLM |
| `templates/qwen.py` | 调用上述共享协议，使用目标步骤可见 tools；生成文字、图片和当前目标 labels | Qwen Processor 编码、精确前缀检查、不截断 |
| `models/qwen.py` | 按需要向协议层提供 Processor/tokenizer/模型配置；记录模板标识 | 权重加载、PEFT、冻结与保存 |
| `training`、`cli.py` | collator 仅在字段不兼容时调整；manifest 纳入新工具/适配配置与来源；preview 展示规范化调用和目标区间 | TRL 训练循环和四个现有命令 |
| 配置 | 明确 schema_version、工具包装和服务模板参数；更新四份 YAML 及完整性校验 | Pixi 依赖版本、全部后端显式参数 |

v2 不兼容 v1 平铺工具池、available_tools 覆盖或 available_tool_indices 子集选择，也不自动兼容旧 ShareGPT 或缺字段的 messages；audit 给出交付缺项，不在读取时猜测迁移。旧数据与图片保持不变，补齐由合作者负责。

现有 `scripts/rebuild_astra_contract.py` 和 `scripts/recover_interactive_contract.py` 仍按旧契约构造记录，尚未迁移；其输出不能直接交给 v2 校验器。本次不执行这些脚本或重写数据。后续迁移须同时更新其契约相关测试，不能用旧格式测试代表 v2 验收。

## 2. 工具 build 与双向适配

`ToolAdapter` 直接构造，不使用注册器。定义三个小接口，结果类型放现有 types 模块：

| 接口 | 输入 | 输出 |
| --- | --- | --- |
| `build(tools, custom_mappings)` | trajectory.tools[step.tool_context_index]；显式 custom 包装配置 | `BuiltTools`：function schema 列表、原始/目标名称与参数映射、指纹 |
| `normalize_calls(calls, built_tools)` | 原始 function/custom 调用、BuiltTools | vLLM 可接受的 function 调用列表，保留 call ID |
| `restore_calls(calls, built_tools)` | vLLM 解析出的 function 调用、BuiltTools | runtime 调用列表，custom 恢复为原始 name/input，function 保持 name/arguments |

Function 参数在归档中为 JSON 字符串；适配后由 vLLM 的消息预处理解析为 dict，供 Qwen 模板遍历。只接受对象参数；不能把坏 JSON、数组或未知工具静默变成空参数。

当前唯一预置包装规则由 YAML 显式填写：custom `exec` → function `exec`，参数名 `input`，值为原始代码字符串；生成 schema 必填字符串 input、禁止额外字段，说明来自原 custom 定义。包装是学生接口，不冒充教师历史定义。`wait`、`sleep` 若源为 function，则保留实际 schema，不自行补齐可选参数。未知 custom 工具、grammar custom 工具、名称冲突先报错，待新增明确适配规则后再支持。

函数构建不输出手写 `<tools>` 或 `<tool_call>` 字符串；Qwen 模板统一负责格式化。规范化和还原要求代码、引号、换行以及参数类型不发生语义变化。既有 `configs/tools/agent_runtime.json` 不作为历史定义 fallback；其必要执行说明只有经来源确认后才能进入交付数据或部署配置。

## 3. 服务端预处理、token 与输出解析

### 3.1 共用协议层

`QwenProtocol` 接收模型配置、Processor、BuiltTools 和显式模板参数；训练与请求构建复用同一规范化规则。

| 接口 | 输入 | 输出 |
| --- | --- | --- |
| `prepare_messages(messages, built_tools, image_root)` | 已选定的可见消息、工具、图片根目录 | 规范化 function 调用、图像引用和顺序表；去掉归档用 index，不改原记录 |
| `render(messages, built_tools, add_generation_prompt)` | 规范化消息和显式模板开关 | Qwen 模板文本及有序图片引用，供现有 Processor 编码 |
| `build_request(...)` | 当前可见消息、模型名、tools、完整服务采样配置 | vLLM ChatCompletionRequest 可验证的字典；custom 已转 function，图片已转换为可传输内容 |
| `parse_output(text, request)` | 模型完整输出文本、含 tools 的请求 | vLLM 的 content/tool_calls，供 ToolAdapter 还原；不执行 |

直接复用已锁定 vLLM：

- `vllm.entrypoints.chat_utils.parse_chat_messages`：处理内容块、工具参数和多模态位置；内部调用 `_postprocess_messages`，项目不再直接散落调用私有函数。
- `vllm.renderers.hf.safe_apply_chat_template`：统一模板选择、developer 兼容转换和 tools 注入。vLLM 接口依赖只集中在 serving 模块，避免扩散。
- `vllm.tool_parsers.qwen3_engine_tool_parser.Qwen3EngineToolParser`：直接构造并调用 `extract_tool_calls`；完整和流式解析均用同一上游引擎，不复制 ms-swift 的正则解析。

必须先截取当前样本可见上下文，再调用 developer→system 合并。否则未来 developer 指令可能被合并进早期训练目标。多段当前可见指令按 vLLM 原顺序合并；归档角色与顺序保持原状。

模板文本交给 Qwen Processor 产生 token 和视觉特征。不要在 tokenize 后再用字符串替换修补 `<tool_call>`、`<function=...>`、`<parameter=...>`、`<tool_response>` 或 `<think>`。只监督当前目标及其结束标记；历史与工具输出保持 mask。

### 3.2 复用原 vLLM 服务

服务入口使用 `vllm serve`，不实现另一个 Web 服务。新增显式服务配置并由配置构建/验证命令打印启动参数，不在 audit/preview 时启动引擎：

- `enable_auto_tool_choice: true`、`tool_call_parser: qwen3_xml`、`reasoning_parser: qwen3`，模板参数 `enable_thinking: false`。本地 0.30.0 中 qwen3_xml 与 qwen3_coder 同指向 Qwen3EngineToolParser；固定一个名称即可。
- 模型、dtype、上下文长度、并行参数、图片像素预算、模板路径/哈希、LoRA 服务方式全部显式填写。保存结果中的 Processor/template 与服务使用的模板一致，图片预算通过服务支持的 mm_processor_kwargs 设置并核对。
- 由 `build_request` 显式发送 `tools`、`tool_choice: auto` 和一致的 chat_template_kwargs。本版使用普通自动工具调用，不默认开启 strict/强制工具生成；这些会影响请求与结束标记，另行验证后接入。
- 本地图片在发送前转换为 data URL，维持顺序与重复次数。服务端用原 vLLM 预处理加载；离线训练仍读本地图片，不重新保存对话。
- HTTP 响应已经包含结构化 tool_calls 时，不再次 XML 解析；仅校验并还原 custom 包装。`parse_output` 用于原始生成文本、离线测试和无 HTTP 推理路径。
- 流式输出沿用 vLLM parser 与响应协议；调用参数完整且可验证之前不执行工具。每次生成使用独立解析状态，不能在请求之间共享。
- LoRA 推理通过 vLLM 原生支持的适配器加载，明确基础模型与 adapter；不为了部署自动合并或修改模型文件。

本阶段只形成请求/解析边界与服务配置，不实现 exec 执行器，不改写内部 Blender 操作。完成实际 rollout 前，必须由已有执行环境接收还原后的调用并按 call ID 返回结果。

## 4. 配置、恢复与验证

保持 Pydantic 和 pydantic_yaml。新配置字段要求显式填写：交付版本、custom 映射（无映射为 []）、协议/解析器名称、模板开关和服务参数；不利用后端默认值补缺。训练与服务共享模型模板和图像预算，独立服务参数放单独 YAML，避免给现有训练配置增加无关字段。

manifest 增加交付版本、工具上下文及其对应来源摘要、每步上下文索引、包装规则、模板、vLLM 版本及解析配置的指纹。旧 ShareGPT checkpoint 与新轨迹契约不视为可恢复的同一次训练；指纹变化须开启新输出目录。

验收分为以下层级，分别报告：

1. **数据检查，无 Processor/权重**：四场景样例；严格 JSONL；空 tools、空 ToolContext、null、来源数量错配、旧选择字段、缺 tools、未知工具来源、非法索引、重复/悬空 call ID、缺图、损坏图、独立会话、动态工具集合；单图/多图只选完整脚本且仍要求真实非空工具上下文；旧版无工具脚本记录不满足 v2。未知输入拒绝，重建输入保留标记并统计。
2. **工具 build 与解析，无权重**：function 保真；custom exec 包装/还原；多行代码、引号、数字/布尔/嵌套 JSON；未知工具、grammar、名称冲突拒绝。完整输出、多个调用、普通文本、截断调用、流式标签/参数分片；解析失败不产生可执行调用。不得自动修复缺失闭合标记后执行。
3. **模板/Processor 对齐**：同一规范化请求走训练与 vLLM 预处理得到相同渲染文本、token、图片顺序、网格和处理后的像素张量；校验 developer 指令时点、工具截图和重复图；完整目标与 prompt 前缀一致，labels 只覆盖目标。
4. **小 Qwen 训练**：沿用现有临时小模型测试，验证 LoRA 更新、有限 loss、保存恢复与空验证集提前报错；CPU/float32 显式关闭混合精度。不运行 27B 训练。
5. **服务链路**：无权重协议测试与实际引擎集成分开。合适的小型兼容模型可用时，使用 vLLM 服务发非流式/流式请求，核对 tools、reasoning/content 分离和调用结果回填；模型自由生成不保证命中特定工具，解析正确性由固定输出测试独立保证。真实 27B 推理和 Blender 完整任务效果需要另行运行，不能以离线测试代替。

先用合成样例开发新接口，再验收合作者每场景一个对象，最后批量 audit/preview。保持一次只改一个链路：交付读取 → 工具 build → 共享协议与模板 → TRL 回归 → 服务对齐。依赖使用现有 Pixi 环境，不升级已锁定 NumPy/vLLM；确需新增依赖才经 Pixi 同步维护 manifest 与 lock。

## 5. 文档实施边界

已交付 v2 规范、独立格式校验器、针对性测试和真实片段配合合成工具上下文的 JSON 示例。该示例不用于训练。数据合作者恢复历史 tools、训练代码改造、服务配置及端到端训练均是后续实施工作。当前 README 和旧设计文档中 ShareGPT 入口说明描述现有读取器，不代表新交付契约已接通；实施时一并更新对应说明和 AGENTS.md 中已过时的数据格式描述，不更改其他用户约定。
