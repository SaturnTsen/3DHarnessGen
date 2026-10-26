# 3DHarnessGen 后训练

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

## Hydra 配置组合

四个场景配置现在通过 Hydra 组合公共参数，训练前仍由 Pydantic 严格校验。
`configs/data`、`template`、`model`、`method` 保存完整公共参数；`scene` 保存场景差异，
`tuning`、`quantization`、`distributed` 分别选择参数更新、量化和并行方式。
组合后的字段必须齐全，不使用代码默认值补参数。原有命令继续可用，并支持 Hydra 覆盖：

```bash
pixi run post-train audit --config configs/sft_single_view.yaml scene=active_visual
pixi run post-train preview --config configs/sft_single_view.yaml --limit 1 quantization=4bit
```

原生 Hydra 入口支持检查配置和批量实验；默认动作是 `audit`，不会启动训练：

```bash
pixi run post-train-hydra --cfg job --resolve
pixi run post-train-hydra command.action=audit scene=multi_view
CUDA_VISIBLE_DEVICES=0 pixi run post-train-hydra --multirun command.action=train method.learning_rate=0.00001,0.00002
```

批量实验按顺序运行，每个任务的训练输出在配置的输出目录下追加独立的批次及任务目录。
Hydra 自身日志与配置保存在当前项目的 `outputs/hydra` 或 `outputs/hydra_sweeps`。
不改变工作目录；数据、模型、输出、恢复路径仍相对于入口 YAML 所在目录解析，
不相对于某个配置组目录或启动目录解析。`~` 仍指向当前用户的主目录。
训练目录中的 `resolved_config.yaml` 是完整展开且已解析路径的配置，可独立读取。
参数值为字符串 `no` 时必须保留引号，避免 Hydra 的 YAML 1.1 将其当作布尔值。

项目迁移后，当前代码、配置和 Pixi editable 安装使用当前项目目录。
旧训练输出中的配置快照、启动记录及指纹属于历史记录，不自动改写；
不要直接把其中的旧绝对路径用于新训练或恢复。已有 checkpoint 的配置和代码指纹校验仍然生效。

### W&B 实验记录

`tracking` 配置组控制 W&B（Weights & Biases）的指标记录。四个场景默认使用 `tracking=wandb`、
`tracking.mode=online`，正式训练自动向 W&B 的 `3DHarnessGen` 项目记录配置和指标。
项目名、账号／团队、实验名称、分组、标签和 SDK 设置都在 `configs/tracking/wandb.yaml` 中显式配置。
`tracking.entity: null` 表示在线模式使用登录账号的默认空间；Git 提交署名不作为 W&B 账号。

```bash
# 先展开并检查配置，不加载模型。
pixi run post-train-hydra --cfg job --resolve

# 单卡 QLoRA，默认在线记录。
CUDA_VISIBLE_DEVICES=0 pixi run post-train-hydra command.action=train \
  quantization=4bit \
  experiment.output_dir=../outputs/post_training/Single-view/qlora_wandb_v1

# 双卡 DDP，同一实验仅由全局主进程记录。
pixi run post-train-distributed train --config configs/sft_single_view.yaml \
  distributed=ddp quantization=4bit \
  experiment.output_dir=../outputs/post_training/Single-view/ddp_qlora_wandb_v1
```

训练使用已有 W&B 登录凭据或 `WANDB_API_KEY` 环境变量；首次登录可直接执行 `pixi run wandb login --verify`。
不需要在训练命令上额外启用在线模式。不提供单独的登录任务。
如需离线记录，显式填写 `tracking.mode=offline`；完全关闭记录使用 `tracking=disabled`。
需要指定团队时填写 `tracking.entity=团队名`；密钥通过登录或 `WANDB_API_KEY` 环境变量提供，不写入 YAML。
W&B 按场景分组，自动接收训练后端的 loss、学习率、梯度范数、验证指标和训练汇总。
只有 `train` 创建 W&B 实验，`audit`、`prepare` 和 `preview` 不创建实验。

记录位于本次训练目录的 `wandb/`，`wandb_run.json` 保存实验 ID 及归属信息。
在线新训练使用 `resume=never`，恢复 checkpoint 时读取原 ID 并使用 `resume=must`，
不会把恢复操作静默变成一个新实验。离线恢复保留同一 ID，SDK 会生成独立的离线记录段；
它不具备在线模式的服务端续写语义。需要上传时，先同步首段，再用 `pixi run wandb sync --append 路径`
按时间顺序追加后续记录段。恢复时保留 `wandb_run.json` 和原有 tracking 配置。

当前只记录配置和指标，不上传模型、checkpoint 或代码，不采集终端输出，
配置中的 `hub_token` 也不写入项目提供给 W&B 的完整配置。
原生 Transformers 回调需要的 `WANDB_LOG_MODEL=false`、`WANDB_WATCH=false` 在训练期间由 YAML 明确控制，
结束后恢复原环境。训练成功或异常退出时都会结束该 W&B 实验，避免 Hydra 下一次任务复用上一次实验。

### 训练方式

| 选择 | Hydra 参数 | 含义 |
| --- | --- | --- |
| BF16 LoRA | `tuning=lora quantization=none` | BF16 基础权重，加可训练适配器 |
| QLoRA | `tuning=lora quantization=4bit` | NF4 基础权重、双重量化、BF16 计算，加可训练适配器 |
| 全参数 | `tuning=full quantization=none` | 更新未被冻结的基础参数；视觉冻结另由 `model.freeze_vision` 控制 |
| 单进程 | `distributed=single` | 单卡；显式设置 `device_map` 时可按层放置 |
| DDP | `distributed=ddp` | 每个进程保存完整模型，处理不同数据并同步梯度 |
| FSDP2 | `distributed=fsdp` | 分片模型状态，训练 checkpoint 使用分片格式，最终导出收集完整状态 |

精度、量化与并行配置可以组合；不兼容配置在加载权重前报错。
单卡训练须通过 `CUDA_VISIBLE_DEVICES` 只暴露一张卡，避免后端自动启用 DataParallel：

```bash
CUDA_VISIBLE_DEVICES=0 pixi run post-train train --config configs/sft_single_view.yaml \
  quantization=4bit experiment.output_dir=../outputs/post_training/Single-view/qlora_v1
```

DDP/FSDP 必须通过多进程启动器启动。以下 Pixi 任务在本机启动两个进程：

```bash
pixi run post-train-distributed train --config configs/sft_single_view.yaml \
  distributed=ddp quantization=4bit experiment.output_dir=../outputs/post_training/Single-view/ddp_qlora_v1
pixi run post-train-distributed train --config configs/sft_single_view.yaml \
  distributed=fsdp experiment.output_dir=../outputs/post_training/Single-view/fsdp_lora_v1
```

需要其他进程数时，使用 `pixi run python -m torch.distributed.run --standalone --nnodes=1
--nproc-per-node=N -m harness_training.cli train --config ...`。
DDP/FSDP 要求 `model.device_map: null`，每个进程使用各自的 GPU。
FSDP2 使用模型声明的层类型进行分片，使用普通梯度检查点节省中间结果内存。
当前版本组合下，分片后端的 `activation_checkpointing` 会导致 LoRA 最终导出为空，
因此配置明确禁止该选项；普通梯度检查点会增加部分参数通信。
QLoRA + FSDP 要求量化存储类型与 `model.dtype` 一致。
不要在各个 torchrun 进程内运行 Hydra `--multirun`。

相对路径以 YAML 所在目录为起点，`~` 自动展开。`prepare()` 先读取当前场景的轨迹路径，再按 `data.instances` 筛选，最后将选中的路径传给 `_split_objects(paths)`。该方法用配置种子打乱路径副本，切分后生成训练、验证和测试分配；测试和验证数量向下取整，余下归入训练集。未选中的对象和其他场景不参与划分。同一组选中对象使用相同种子和比例时可复现；选中对象集合变化时可能重新分组。

轨迹格式以 [v3 交付规范](docs/POST_TRAIN_TRAJECTORY_CONTRACT.md) 为准。`tools/validate_trajectory.py` 和训练读取器共用 `harness_training/data/contract.py` 的严格校验，不兼容旧的 `conversations` / `sharegpt_conversion` 数据。

按 `metadata.steps` 的 `context_indices` 和 `target_index` 选择训练内容，按 `data.accepted_statuses` 筛选状态，`include_final_reply` 控制是否学习最终回复。脚本、工具调用与状态回复均按步骤声明选择。对象的所有会话保持在同一个集合，样本 ID 包含会话和步骤标识。

CLI 的 `audit`、`prepare`、`preview` 和 `train` 默认额外接受 `UNKNOWN` 步骤，并在标准错误输出中发出警告；原始状态保持不变，不改写 YAML。直接调用 `Trainer` 时仍按传入配置筛选。

图片直接读取消息内容块中的 `images/...` 路径，以输入 JSONL 所在目录为起点解析，保留出现顺序与重复位置。缺失图片或无法解码的样本跳过并记录；`data.missing_images: error` 可以改为立即报错。

## 训练

配置中的参数是起始值，不是对 27B 模型显存占用的保证。默认 BF16 LoRA 的基础模型仍以 BF16 加载，需要足够显存；LoRA 本身不压缩基础模型。先调整模型与运行参数，再执行训练。

`model.device_map` 必须显式填写，当前配置组填写 `null`；单进程跨多张 GPU 放置模型时可填写 `balanced`。这是按层分配权重，不能据此推断多进程训练吞吐或显存余量。

```bash
CUDA_VISIBLE_DEVICES=0 pixi run post-train train --config configs/sft_single_view.yaml
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
pixi run post-train smoke
pixi run post-train smoke --scene ActiveVisual --limit 2
```

每个场景选择一个真实对象检查数据并用 Processor 编码预览，再通过已有 Pixi 测试任务验证小 Qwen 训练、保存和恢复。报告保存在 `temp/smoke-post-training-*/`；不加载 27B 权重。默认接受 `UNKNOWN` 步骤并输出警告，不修改正式配置；保留 `--include-unknown` 以兼容旧命令。失败或测试被跳过时返回非零退出码。

终端实时显示阶段、耗时、对象和样本数量、预览图片及 token 数，以及逐项测试和训练输出。阶段日志保存在 `smoke.log`，完整测试日志保存在 `tests.log`，并列出最慢的 10 项测试。

```bash
pixi run test-post-training
```

测试覆盖配置、对象划分、显式步骤选择、缺失图片、工具截图、工具调用关系和训练标签。另用临时创建的小 Qwen 模型验证 LoRA 更新、SFT 保存与断点恢复，不训练下载的 27B 模型。使用真实 Processor 的测试在本地模型文件不存在时会跳过。
