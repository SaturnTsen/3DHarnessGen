# v3 轨迹后训练设计

数据字段和关联检查以 [v3 交付规范](POST_TRAIN_TRAJECTORY_CONTRACT.md) 及 `harness_training/data/contract.py` 为准。`tools/validate_trajectory.py` 保留原校验命令和公共导入入口，与训练侧共用同一份实现。

## 处理流程

`Trainer` 直接组合 `TrajectoryDataset`、`QwenTemplate`、`QwenModel` 和 `SFTMethod`。四种场景各自读取、划分和保存，默认从同一基础模型开始。

1. 根据场景和实例配置查找原始 JSONL，先按对象划分训练、验证、测试。同一对象的多个会话与步骤不跨集合。
2. 使用 `validate_record` 严格检查每条记录；旧 ShareGPT 格式、重复会话、错误角色、调用关系或索引均记录为拒绝原因。
3. 用每个 step 的 `context_indices` 加 `target_index` 创建内存视图，按显式状态配置筛选。最终回复由 `include_final_reply` 控制，不推导旧的 chain 或原始消息映射。
4. 图片从消息内容块绑定，路径相对于原轨迹所在目录。顺序和重复位置不变，仅检查当前选中样本涉及的图片；缺失或不可读时按配置跳过样本或报错。
5. Qwen 模板仅接收已经选定的历史和目标。调用 Processor 渲染并编码文本和图片，确认 prompt 的 token 是完整输入的精确前缀。
6. 只有当前目标回复及结束标记参与损失计算；历史、工具结果、图片和补齐位置的 labels 为 -100。超长样本拒绝或记录，不截断。
7. SFT 使用内存样本索引及自定义批次补齐器训练。启用验证却无可用验证样本时，在加载权重前报错。

## Qwen 适配

本地 Qwen 模板不直接接收 developer 和 custom 调用。对当前样本已经可见的 system/developer 指令按原顺序合并为首条 system 消息；不读取目标之后的指令。

工具来自该步骤唯一可见的 `additional_tools` 声明，并由契约验证与顶层工具上下文一致。函数 schema 保留其声明；function 参数严格解析为对象。custom 包装由 YAML 的 `template.custom_tools` 明确指定名称和字符串参数名，当前为 `exec: input`。包装保留原输入，不把 exec 中的代码改写为直接 Blender 调用。命名空间与输入格式保留在学生接口的工具说明中，不修改归档。

工具结果仍是 tool 消息，调用 ID 和内容块顺序保留。工具发现返回的说明留在结果中，不提升为工具声明。未知 custom 映射报错，不猜测转换。静态 runtime 工具文件与旧 ShareGPT 映射不再参与处理。

## 检查、预览与训练

- `audit`：检查数据和图片，不加载 Processor 或权重。
- `prepare`：保存检查报告和对象划分，不导出对话。
- `preview`：加载 Processor，展示选定历史、目标、图片与 token 数；不加载权重。
- `train`：先预编码检查样本，再加载权重执行 SFT。

配置使用 Pydantic 和 pydantic_yaml，字段必须显式填写。`data.schema_version` 固定为 3；四份配置均包含 custom 映射。TRL、LoRA 和 adapter 参数完整性检查保持原有行为。所有依赖继续由 Pixi 管理。

保存内容包括配置快照、源文件及图片指纹、包代码指纹、对象划分、样本 ID、拒绝原因、checkpoint 和最终模型。LoRA 保存适配器与 Processor。恢复时拒绝数据、模板、模型或关键训练设置变化。

## 验证边界

测试覆盖严格契约、对象与会话划分、显式历史、未来信息隔离、工具与重复图片、目标标签、长度限制，以及小 Qwen 的 LoRA 更新、保存和恢复。小模型验证不代表完整 27B 训练已验证。

本次不实现 vLLM 服务协议、模型输出工具解析、Blender 执行器或重建质量评估。数据修复脚本独立于训练入口；修复证据保存在其 mapping 中，不作为 v3 交付字段。
