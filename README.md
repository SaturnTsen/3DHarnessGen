# 3DBootstrapLM 后训练

四种场景各自训练，直接读取已有 v3 轨迹：

```text
3DHarnessBench/runs/<setting>/gpt-6-astra/<instance>/post_train_trajectory/trajectory.jsonl
```

代码位于项目根目录的 `harness_training/`。第一版实现 SFT。`Trainer` 是统一入口；data、templates、models 和 training methods 通过各自的 base 定义接口，使用具体类直接初始化。模型类持有模型和 Processor，template 将已选定的 v3 消息转为 Qwen 输入。数据不重新导出。

设计说明见 [docs/POST_TRAINING_DESIGN.md](docs/POST_TRAINING_DESIGN.md)。

## 环境

所有依赖由 `pixi.toml` 和 `pixi.lock` 管理，继续使用 `.pixi/envs/default`，不创建额外环境。

```bash
pixi install --locked
```

NumPy 和 vLLM 固定为 `2.5.3`、`0.30.0`。NumPy 已有的兼容例外通过 Pixi 的 dependency override 明确保留。后训练依赖、本项目的 editable 安装和开发测试依赖都纳入 Pixi 管理。

## 检查与预览

```bash
pixi run post-train audit --config configs/sft_single_view.yaml
pixi run post-train prepare --config configs/sft_single_view.yaml
pixi run post-train preview --config configs/sft_single_view.yaml --limit 1
```

`audit` 检查数据，输出可用数量和拒绝原因，不加载模型。`prepare` 只保存检查报告和对象划分，不创建新的对话数据。`preview` 加载 Processor，展示模型输入、当前训练目标、图片和 token 数量；`--render-only` 可以跳过图片与 token 编码。

四个配置分别为：

| 场景 | 配置 |
| --- | --- |
| Single-view | `configs/sft_single_view.yaml` |
| Multi-view | `configs/sft_multi_view.yaml` |
| ActiveVisual | `configs/sft_active_visual.yaml` |
| Full3DInteraction | `configs/sft_full_3d_interaction.yaml` |

相对路径以 YAML 所在目录为起点，`~` 自动展开。`prepare()` 先读取当前场景的轨迹路径，再按 `data.instances` 筛选，最后将选中的路径传给 `_split_objects(paths)`。该方法用配置种子打乱路径副本，切分后生成训练、验证和测试分配；测试和验证数量向下取整，余下归入训练集。未选中的对象和其他场景不参与划分。同一组选中对象使用相同种子和比例时可复现；选中对象集合变化时可能重新分组。

轨迹格式以 [v3 交付规范](docs/POST_TRAIN_TRAJECTORY_CONTRACT.md) 为准。`tools/validate_trajectory.py` 和训练读取器共用 `harness_training/data/contract.py` 的严格校验，不兼容旧的 `conversations` / `sharegpt_conversion` 数据。

按 `metadata.steps` 的 `context_indices` 和 `target_index` 选择训练内容，按 `data.accepted_statuses` 筛选状态，`include_final_reply` 控制是否学习最终回复。脚本、工具调用与状态回复均按步骤声明选择。对象的所有会话保持在同一个集合，样本 ID 包含会话和步骤标识。

图片直接读取消息内容块中的 `images/...` 路径，以输入 JSONL 所在目录为起点解析，保留出现顺序与重复位置。缺失图片或无法解码的样本跳过并记录；`data.missing_images: error` 可以改为立即报错。

## 训练

配置中的参数是起始值，不是对 27B 模型显存占用的保证。默认 BF16 LoRA 的基础模型仍以 BF16 加载，需要足够显存；LoRA 本身不压缩基础模型。先调整模型与运行参数，再执行训练。

```bash
pixi run post-train train --config configs/sft_single_view.yaml
```

训练会先用 Processor 检查所有选中的样本。过长或无法可靠转换的样本记录后跳过，不自动截断。检查通过后才加载模型权重。每条样本只学习当前选定的 assistant 回复，system、用户请求、历史回复、工具返回、图片与补齐位置均不计算训练误差。

每个配置有独立输出目录。已有训练结果时，代码拒绝直接覆盖；可以修改 `experiment.output_dir` 开始另一轮，或在 YAML 的 `method.resume_from` 中指定恢复 checkpoint（新训练显式填写 `null`）：

```yaml
method:
  # 其余 method 参数仍须保留；路径相对于 YAML 所在目录。
  resume_from: ../outputs/post_training/Single-view/sft_v1/checkpoints/checkpoint-100
```

恢复时检查数据、图片、模板、模型配置及关键训练参数是否变化。允许调整训练总步数和日志／保存频率。

输出包含配置快照、数据报告、对象划分、拒绝原因、checkpoint 和 `final`。LoRA 保存 adapter 与 Processor；全参数训练保存完整模型。验证集 loss 由 SFT 后端计算。实际 Blender 重建质量和交互任务评估尚未接入。

交互场景仅使用该步骤历史中选定的 `additional_tools` 声明，不回退到静态工具文件。选定历史内的 system/developer 指令按顺序合并为 Qwen 的 system 输入；归档角色不修改。function 参数从 JSON 字符串解析为对象；custom 工具根据 YAML 的 `template.custom_tools` 显式映射（当前 `exec: input`），包装成函数调用，原始代码保持不变。工具命名空间和 custom 输入格式作为工具说明保留。未知 custom 映射会记录为编码失败。

训练侧不执行工具，也不实现部署时的工具调用解析或执行环境。部署方需将模型输出的 `exec(input=...)` 还原为原 runtime 的 custom `exec` 调用；本次未接入服务推理。

## 验证代码

通过现有 CLI 执行冒烟测试：

```bash
pixi run post-train smoke --include-unknown
pixi run post-train smoke --scene ActiveVisual --include-unknown --limit 2
```

每个场景选择一个真实对象检查数据并用 Processor 编码预览，再通过已有 Pixi 测试任务验证小 Qwen 训练、保存和恢复。报告保存在 `temp/smoke-post-training-*/`；不加载 27B 权重。`--include-unknown` 只调整此次检查的状态筛选，不修改正式配置。失败或测试被跳过时返回非零退出码。

```bash
pixi run test-post-training
```

测试覆盖配置、对象划分、显式步骤选择、缺失图片、工具截图、工具调用关系和训练标签。另用临时创建的小 Qwen 模型验证 LoRA 更新、SFT 保存与断点恢复，不训练下载的 27B 模型。使用真实 Processor 的测试在本地模型文件不存在时会跳过。
