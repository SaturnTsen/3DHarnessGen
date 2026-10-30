# 3DHarnessGen 后训练

四种场景各自训练，直接读取已有 v3 轨迹：

```text
3DHarnessBench/runs/<setting>/gpt-6-astra/<instance>/post_train_trajectory/trajectory.jsonl
```

代码位于项目根目录的 `harness_training/`。第一版实现 SFT。`Trainer` 是统一入口；data、templates、models 和 training methods 通过各自的 base 定义接口，使用具体类直接初始化。模型类持有模型和 Processor，template 将已选定的 v3 消息转为 Qwen 输入。默认直接读取原始轨迹，下文提供独立的可选清洗入口。

设计说明见 [docs/POST_TRAINING_DESIGN.md](docs/POST_TRAINING_DESIGN.md)。

## GPT-6-Astra 轨迹清洗

```bash
pixi run python -m scripts.clean_astra_trajectories
```

只读取四种场景的 `gpt-6-astra/*/post_train_trajectory/trajectory.jsonl`，输出到
`data/trajectory_cleaned/<setting>/<object>/trajectory.jsonl`，引用的原图复制到对象目录的
`imgs/`，不修改输入。每行保留一个原有训练目标：开头的用户任务及 developer 指令、
精简后的工具声明、此前最近五轮未失败的工具调用及返回和关联的 agent 文字，当前目标另外保留。
窗口外图片不进入该样本；同一图片可被多个样本共享，出现顺序和重复引用均保留。

脚本检查全量调用后，仅删除候选列表中实际未使用的声明及 exec 内嵌说明，包括 web、
图片生成、目标管理、协作 agent、询问用户及 MCP 资源发现/读取。失败按步骤 `ERROR`、
`Script failed`、实际结果中的 `isError: true`、错误状态或非零退出码识别，删除失败调用和返回，
也不将失败调用作为训练目标；`UNKNOWN` 保留。一次带工具调用的 assistant 消息计为一轮，
`wait` 同样计一轮，不改写调用代码或跨窗口补造变量状态。

输出仍为 v3 格式，重新编号消息、历史和目标索引，来源及旧索引映射写入
`cleaning_manifest.jsonl`，汇总写入 `cleaning_report.json`。缺失或损坏的保留图片会使任务报错；
全部记录校验和图片复制通过后才发布输出目录。已有输出不会覆盖，可用 `--output-root` 指定新目录，
`--settings` 选择场景，`--keep-tool-rounds` 调整历史轮数。训练读取清洗结果时，通过 `data.files`
显式列出对象 JSONL，`image_root` 保持 `null`，图片相对各 JSONL 解析。清洗不保证样本满足 token
预算或跨窗口代码依赖完整；训练前仍需按配置执行编码检查。

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

另外提供两份 9B + NF4 QLoRA + 双卡 FSDP2 混合训练配置：

| 混合场景 | 配置 |
| --- | --- |
| Single-view + Multi-view | `configs/sft_view_mix_9b_fsdp.yaml` |
| ActiveVisual + Full3DInteraction | `configs/sft_interaction_mix_9b_fsdp.yaml` |

混合配置通过 `experiment.scene` 列表选择场景，直接读取各场景原有轨迹，按可用样本数量合并，不重采样。
先将所有场景的对象合并去重，再按对象统一划分训练、验证、测试集（80% / 10% / 10%，数量向下取整），同一对象不会跨集合。
两组各自从基础模型开始训练一轮，分别保存结果；接受 `OK` 和 `UNKNOWN` 步骤，缺失图片的样本会记录并跳过。

```bash
pixi run post-train audit --config configs/sft_view_mix_9b_fsdp.yaml
pixi run post-train audit --config configs/sft_interaction_mix_9b_fsdp.yaml
pixi run post-train-distributed train --config configs/sft_view_mix_9b_fsdp.yaml
pixi run post-train-distributed train --config configs/sft_interaction_mix_9b_fsdp.yaml
```

对应的 27B 混合配置为 `configs/sft_view_mix_27b_fsdp.yaml` 和
`configs/sft_interaction_mix_27b_fsdp.yaml`，数据、模板和对象划分分别与上述两份 9B 配置一致。
两份 YAML 通过 `defaults` 选择公共模型配置 `qwen38_27b`，基础模型路径为 `~/models/Qwen3.8-27B`，不继承任何 9B 任务入口；
实验名与输出目录均使用 `qwen38_27b`。该目录配置中的架构字段为 `qwen3_5`，基础架构为 `Qwen3_5ForConditionalGeneration`，训练使用其显式子类 `LongContextQwenForConditionalGeneration`。
目标硬件为两张每卡至少 80 GB 的 H100，采用 NF4 四位量化、BF16 计算和量化存储、
FSDP2 分片、两级梯度检查点与模型内分块损失（后端填写 `loss_type: nll`），每卡批次 1、梯度累积 8，
有效批次为 16，训练一轮。27B 长度预算默认为 48K（49,152 token），不截断；超过预算会记录并跳过。
当前仅验证配置组合和数据设置，尚未在 H100 上验证训练及峰值显存。
迁移机器时需按实际位置覆盖 `model.path` 和 `data.runs_root`。

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 \
  pixi run post-train-distributed train --config configs/sft_view_mix_27b_fsdp.yaml
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 \
  pixi run post-train-distributed train --config configs/sft_interaction_mix_27b_fsdp.yaml
```

## 3DCodeVerse 分轮混合训练

三轮配置使用 Qwen3.5-9B、NF4 QLoRA 和 FSDP2，每轮训练 2 个 epoch。
Replay 表示全量保留此前数据，不额外重采样；Blender 使用 Bioinspired3D 和
3DCodeBench 的精选 `instances_geo` / `instances_tex`，不使用 `*_full`。

| 配置 | Blender geo / tex | DeepCAD | Articraft CadQuery | URDF | 原始记录总数 |
| --- | --- | --- | --- | --- | --- |
| `configs/3dcodeverse_r2_sft_9b_fsdp.yaml` | 6,510 / 3,737 | v1：26,566 | 全部 6 个子集：6,878 | 无 | 43,691 |
| `configs/3dcodeverse_r3_sft_9b_fsdp.yaml` | 6,510 / 3,737 | v1、v1_1、v1_2：79,888 | 6,878 | 无 | 97,013 |
| `configs/3dcodeverse_r4_sft_9b_fsdp.yaml` | 6,510 / 3,737 | 全部 5 个子集：128,198 | 6,878 | geo / tex：6,145 | 151,468 |

每轮通过 `model.init_from` 加载上一轮的适配器权重，`method.resume_from: null`
使优化器、调度器和训练步数重新开始；R3、R4 分别指向 R2、R3 输出目录下的 `final`。
R2 指向用户指定的 R1 `checkpoint-1080`，迁移机器时需检查该绝对路径是否可用。
每轮分别按对象划分验证集，比例沿用 Blender 配置；表中数量尚未扣除验证集、缺失图片或超长样本。

```bash
pixi run post-train audit --config configs/3dcodeverse_r2_sft_9b_fsdp.yaml
pixi run post-train-distributed train --config configs/3dcodeverse_r2_sft_9b_fsdp.yaml
pixi run post-train-distributed train --config configs/3dcodeverse_r3_sft_9b_fsdp.yaml
pixi run post-train-distributed train --config configs/3dcodeverse_r4_sft_9b_fsdp.yaml
```

## 独立配置与命令行覆盖

`configs/*.yaml` 的所有顶层入口只通过 `defaults` 引用公共配置组，任务入口之间不互相继承。
模型、数据、模板、训练后端、并行和 W&B 的公共参数保留在相应子目录中；实验名称、输出目录、
场景选择，以及训练轮数、步数上限、批次、学习率等任务参数直接写在各自入口中。
训练前由 Pydantic 严格校验完整组合结果，不使用代码默认值补参数。
修改一个任务入口不会影响其他入口；例如 Blender 的训练轮数直接在
`configs/sft_blender_9b_fsdp.yaml` 的 `method.epochs` 中设置，当前为 `2.0`。

`distributed=single`、`tracking=disabled`、`model=qwen35_9b` 等写法由 Hydra 原生处理，
用于替换 `defaults` 中选中的公共配置。字段覆盖如 `method.epochs=1`、`tracking.entity=...` 同样可用。

训练总序列长度预算由 `defaults` 选中的公共模型配置的 `max_length` 指定，入口可通过
`model.max_length` 显式覆盖。`configs/model/qwen35_9b.yaml` 为 96K（98,304 token），
`configs/model/qwen38_27b.yaml` 为 48K（49,152 token）；短程实验配置以其显式覆盖值为准。
公共模板通过 `${model.max_length}` 引用该值，无需重复修改。
这里的长度包含历史、图片 token 和当前目标回复，不是生成时的 `max_new_tokens`。
命令行可用 `model.max_length=...` 显式调整；这些预算不代表已完成对应长度的显存实测。

```bash
pixi run post-train audit --config configs/sft_single_view.yaml scene=active_visual
pixi run post-train preview --config configs/sft_single_view.yaml --limit 1 quantization=4bit
```

原生 Hydra 入口支持检查配置和批量实验；默认动作是 `audit`，不会启动训练：

```bash
pixi run post-train-hydra --cfg job --resolve
pixi run post-train-hydra command.action=audit scene=multi_view
CUDA_VISIBLE_DEVICES=0 pixi run post-train-hydra --multirun command.action=train distributed=single method.learning_rate=0.00001,0.00002
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

每份入口通过 `defaults` 选择 W&B（Weights & Biases）公共配置，任务差异写在自己的 `tracking` 段。
四个场景默认使用 `tracking=wandb`、`tracking.mode=online`，正式训练自动向 W&B 的 `3DHarnessGen` 项目记录配置和指标。
公共项目名、账号／团队和 SDK 设置保留在 `configs/tracking/wandb.yaml`，也可在任务入口或命令行覆盖。
`tracking.entity: null` 表示在线模式使用登录账号的默认空间；Git 提交署名不作为 W&B 账号。

```bash
# 先展开并检查配置，不加载模型。
pixi run post-train-hydra --cfg job --resolve

# 单卡 QLoRA，默认在线记录。
CUDA_VISIBLE_DEVICES=0 pixi run post-train-hydra command.action=train \
  distributed=single quantization=4bit \
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
| 全参数 | `tuning=full quantization=none model.long_context=null` | 更新未被冻结的基础参数；视觉冻结另由 `model.freeze_vision` 控制 |
| 单进程 | `distributed=single` | 单卡；显式设置 `device_map` 时可按层放置 |
| DDP | `distributed=ddp` | 每个进程保存完整模型，处理不同数据并同步梯度 |
| FSDP2 | `distributed=fsdp` | 分片模型状态，训练 checkpoint 使用分片格式，最终导出收集完整状态 |

精度、量化与并行配置可以组合；不兼容配置在加载权重前报错。
单卡训练须通过 `CUDA_VISIBLE_DEVICES` 只暴露一张卡，避免后端自动启用 DataParallel：

```bash
CUDA_VISIBLE_DEVICES=0 pixi run post-train train --config configs/sft_single_view.yaml \
  distributed=single quantization=4bit experiment.output_dir=../outputs/post_training/Single-view/qlora_v1
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

27B 的 Single-view 分块损失与双卡 FSDP2 短训配置如下，保留完整输入和图片，
当前配置直接选择公共 qwen38 模型、四位 QLoRA 和 FSDP 配置，在自身文件中设置梯度累积为 1，最多更新 3 步，并关闭验证和在线记录：

```bash
pixi run post-train-distributed train --config configs/sft_single_view_fsdp_trial.yaml
```

27B 显式模型类分块计算当前回复的词表分数与交叉熵，后端填写 `loss_type: nll`；长序列的模型内部激活仍占用显存。
该配置用于检查训练能否运行，不能代替完整训练或验证集评估。
重复试跑时需显式指定新的 `experiment.output_dir`。
在双卡 A6000 上，27B BF16 LoRA 的最长 26,749-token 样本仍然 OOM；
详见 [实测记录](docs/reports/CHUNKED_NLL_FSDP_2026-10-27.md)。

### 官方 Qwen3.5-9B

`configs/sft_single_view_9b_fsdp.yaml` 使用 9B + NF4 QLoRA + 双卡 FSDP2，
保留全部长样本训练一轮。最长 26,749-token 样本的两步压力测试已通过，
两卡峰值分配约 25.72 / 23.51 GiB，详见
[显存测试记录](docs/reports/QWEN35_9B_FSDP_2026-10-27.md)。

该配置已完成 Single-view 全量一轮训练（320 条训练、40 条验证、20 次更新），
训练损失 0.346174，验证损失 0.339337；最终 LoRA 已在新进程中精确重载并通过图片输入生成验证。
[完整报告与加载方法](outputs/post_training/Single-view/qwen35_9b_qlora_fsdp_epoch1_v1/RUN_REPORT.md)
包含权重校验、逐样本数据来源、完整日志与 W&B 结果链接。

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 \
  pixi run post-train-distributed train --config configs/sft_single_view_9b_fsdp.yaml
```

9B 入口在各自的 `model.path` 中指定下载到 `~/models/Qwen3.5-9B` 的官方多模态权重。
`configs/sft_single_view_9b.yaml` 为 Single-view 的双卡 DDP + NF4 QLoRA 配置，
基础计算使用 BF16，视觉模块冻结；沿用完整脚本、图片及对象划分规则，默认在线记录 W&B。
使用 TRL 原生 `loss_type: chunked_nll` 分块计算相同的交叉熵，避免完整词表分数矩阵导致显存不足。
模型配置组也可通过 `model=qwen35_9b` 用于其他场景。

先执行短程配置检查真实 9B 模型的训练与保存：

```bash
pixi run post-train-distributed train --config configs/sft_single_view_9b_trial.yaml
```

该配置训练 3 步，每卡每步 1 条样本，梯度累积为 1，每步保存，关闭验证，
并记录显存指标。输入长度预算保持 32768，不截断脚本。
试跑目录是 `outputs/post_training/Single-view/qwen35_9b_qlora_trial_v2`。
已经存在训练结果时，使用新的输出目录或显式恢复；恢复示例：

```bash
pixi run post-train-distributed train --config configs/sft_single_view_9b_trial.yaml \
  method.resume_from=../outputs/post_training/Single-view/qwen35_9b_qlora_trial_v2/checkpoints/checkpoint-2
```

正式配置启用验证，使用独立输出目录和原有梯度累积设置：

```bash
pixi run post-train-distributed train --config configs/sft_single_view_9b.yaml
```

相对路径以 YAML 所在目录为起点，`~` 自动展开。`prepare()` 先读取当前场景的轨迹路径，再按 `data.instances` 筛选，最后将选中的路径传给 `_split_objects(paths)`。该方法用配置种子打乱路径副本，切分后生成训练、验证和测试分配；测试和验证数量向下取整，余下归入训练集。未选中的对象和其他场景不参与划分。同一组选中对象使用相同种子和比例时可复现；选中对象集合变化时可能重新分组。

轨迹格式以 [v3 交付规范](docs/POST_TRAIN_TRAJECTORY_CONTRACT.md) 为准。`tools/validate_trajectory.py` 和训练读取器共用 `harness_training/data/contract.py` 的严格校验，不兼容旧的 `conversations` / `sharegpt_conversion` 数据。

按 `metadata.steps` 的 `context_indices` 和 `target_index` 选择训练内容，按 `data.accepted_statuses` 筛选状态，`include_final_reply` 控制是否学习最终回复。脚本、工具调用与状态回复均按步骤声明选择。对象的所有会话保持在同一个集合，样本 ID 包含会话和步骤标识。

CLI 的 `audit`、`prepare`、`preview` 和 `train` 默认额外接受 `UNKNOWN` 步骤，并在标准错误输出中发出警告；原始状态保持不变，不改写 YAML。直接调用 `Trainer` 时仍按传入配置筛选。

图片直接读取消息内容块中的 `images/...` 路径，以输入 JSONL 所在目录为起点解析，保留出现顺序与重复位置。缺失图片或无法解码的样本跳过并记录；`data.missing_images: error` 可以改为立即报错。

## 训练

配置中的参数是起始值，不是对 27B 模型显存占用的保证。原有 27B 入口默认使用四位 QLoRA 和双卡 FSDP2；BF16 用于计算。训练前仍需在目标硬件验证峰值显存。

`model.device_map` 必须显式填写，当前配置组填写 `null`；单进程跨多张 GPU 放置模型时可填写 `balanced`。这是按层分配权重，不能据此推断多进程训练吞吐或显存余量。

```bash
pixi run post-train-distributed train --config configs/sft_single_view.yaml
```

训练会先用 Processor 检查所有选中的样本。过长或无法可靠转换的样本记录后跳过，不自动截断。检查通过后才加载模型权重。每条样本只学习当前选定的 assistant 回复，system、用户请求、历史回复、工具返回、图片与补齐位置均不计算训练误差。

每个配置有独立输出目录。已有训练结果时，代码拒绝直接覆盖；可以修改 `experiment.output_dir` 开始另一轮，或在 YAML 的 `method.resume_from` 中指定恢复 checkpoint（新训练显式填写 `null`）：

```yaml
method:
  # 其余 method 参数仍须保留；路径相对于 YAML 所在目录。
  resume_from: ../outputs/post_training/Single-view/sft_v1/checkpoints/checkpoint-100
```

通过 Pixi 的 `post-train` / `post-train-distributed` 入口恢复时，也可以在命令行填写
`method.resume_from=auto`，按步数从大到小选择原输出目录中可恢复的 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=3 pixi run post-train train \
  --config configs/3dcodeverse_r2_sft_9b_fsdp.yaml \
  distributed=single \
  tracking.entity=lingada60-institut-polytechnique-de-paris \
  tracking.project=huggingface \
  method.resume_from=auto
```

保留原训练的所有配置覆盖项及输出目录。启动层检查训练状态 JSON、权重、优化器、调度器及
各进程随机状态文件；跳过缺文件、JSON 未写完或进程数不匹配的候选并打印原因，没有候选则报错，
不会从头开始。文件检查不能证明所有张量未损坏，实际加载失败仍会报错。
`auto` 仅是这两个 Pixi 入口的命令行选项，YAML 中仍填写 `null` 或具体路径。
默认 `method.resume_policy: strict`，原始配置、数据、代码指纹检查和 W&B `resume=must`
策略照常执行。保留原目录的 `manifest.json`、`resolved_config.yaml` 和 `wandb_run.json`。
跨卡数恢复须显式选择下述 `elastic` 策略，单独指定 `auto` 不会放宽检查。

### 改变 GPU 数量后恢复

`method.resume_policy=elastic` 支持在 data、model、template 不变时，改变 GPU 数量、
single/DDP/FSDP2 并行方式、每卡 batch 和梯度累积，但必须保持每次参数更新的全局样本数不变：

```text
全局 batch = GPU 数量 × method.batch_size × method.gradient_accumulation_steps
```

例如原单卡 batch=1、累积=8，改用双卡时将累积改成 4。仍使用原输出目录、
数据路径、初始化来源、W&B 项目和团队，续接同一 run，而不是设置 `model.init_from` 开启新阶段：

```bash
CUDA_VISIBLE_DEVICES=2,3 \
PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 \
pixi run post-train-distributed train \
  --config configs/3dcodeverse_r2_sft_9b_fsdp.yaml \
  distributed=fsdp \
  tracking.entity=lingada60-institut-polytechnique-de-paris \
  tracking.project=huggingface \
  method.resume_policy=elastic \
  method.gradient_accumulation_steps=4 \
  method.resume_from=/mnt/disk5/bingchen/3DHarnessGen/outputs/post_training/3DCodeVerse/r2_qwen35_9b_qlora_fsdp_epoch2_v1/checkpoints/checkpoint-2840
```

`method.resume_from=auto` 同样可与 elastic 组合。四进程须改用
`pixi run python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4
-m scripts.post_train train --config ...`，上例的累积相应为 2。默认 Pixi 分布式任务固定启动两进程。

elastic 采用独立随机种子确定每轮全局样本排列，在已完成参数更新的边界保存下一样本位置。
更改卡数仅重新分配同一全局批次，轮次尾部通过不计算误差的占位样本补齐通信，不额外训练真实样本。
新 checkpoint 增加 `data_progress.json`、`resume_manifest.json` 和 `portable_state.pt`，
后者仅收集可训练 LoRA 与按参数名组织的优化器状态，不收集完整冻结基础模型；会增加 checkpoint
大小和保存时间。原始 checkpoint 仍保留。更新状态成功写完后才发布数据位置文件。

旧 checkpoint 从 `trainer_state.json`、原 manifest、采样种子和每进程随机状态重建数据位置；
只接受已列入兼容清单的旧训练代码或当前代码，以及可重建的 random/sequential 采样。
当前旧版本清单覆盖 `b9a0498`、`4c5fe94`、`59f7fb3`、`276195d`、`ead7332`、`3ecf319` 的训练包代码快照；
其间仅变更配置或文档、训练包代码相同的提交同样适用。
不支持 drop_last、非可复现随机采样和其他自定义采样。无法证明数据位置时拒绝启动；不会仅因
配置指纹变化就整体绕过数据、模型、模板、版本和代码检查。旧 checkpoint 本身不被改写。

当前 elastic 限定 LoRA/QLoRA、PyTorch AdamW、FSDP2 分片格式以及固定的全局 batch。
学习率、优化器种类、数据、模板规则、LoRA 和精度设置仍须一致。原优化器、调度器和 global_step
被恢复，W&B 使用原 ID；线上已写到 checkpoint 之后的记录不会自动回滚。
原 run 的 `resume/*` 汇总记录保存恢复时的步数、数据位置和新旧 GPU 数量。
改变 GPU 数量后各卡随机流会重新确定，不能承诺逐位相同的 dropout 或浮点结果。

### 恢复前检查与长度预算

把恢复命令中的 `train` 替换成 `check-resume`，可以在不加载 Processor、不编码图片、
不加载模型张量、不连接 W&B 的情况下，检查 checkpoint 状态文件、原配置、版本、代码、
源文件摘要、全局 batch、数据位置及本地 W&B 身份。例如：

```bash
CUDA_VISIBLE_DEVICES=2,3 pixi run post-train-distributed check-resume \
  --config configs/3dcodeverse_r2_sft_9b_fsdp.yaml \
  tracking.entity=lingada60-institut-polytechnique-de-paris \
  tracking.project=huggingface \
  method.resume_policy=elastic method.batch_size=2 method.gradient_accumulation_steps=2 \
  method.resume_from=/mnt/disk5/bingchen/3DHarnessGen/outputs/post_training/3DCodeVerse/r2_qwen35_9b_qlora_fsdp_epoch2_v1/checkpoints/checkpoint-2840
```

正式 `train` 在任何 preflight 编码前也执行相同检查。差异显示为具体配置字段、包版本、
文件路径及原/新摘要，完整报告写入 `temp/resume_check.json`，不覆盖原训练报告。
检查通过只表示上述静态条件满足；实际样本编码、张量加载、GPU 显存及线上 W&B 连通性仍在训练时检查。

elastic 恢复允许提高 `model.max_length` / `template.max_length`，但固定使用原 checkpoint
的 train/validation/test 样本集合和顺序，原先因超长而排除的样本不会插入本轮训练。
这避免长度预算变化被误判为数据变化，以及插入新样本导致位置错乱。
降低任一长度预算会在编码前拒绝，请恢复原值或更高值；需要重新选择样本时应启动新训练。
原始轨迹、图片或模型元数据实际变更时仍拒绝恢复，并在报告中指明具体文件。

### 训练预检查的并行与缓存

`train` 的 preflight 把轨迹读取及格式校验、去重后的图片完整性检查、样本编码和源文件摘要计算
分摊给 torchrun 的各进程。每个 JSONL 只由一个进程读取校验，每张引用的图片在完整性检查阶段
全局只检查一次；汇总后保留原 train/validation/test 划分、样本顺序和拒绝原因。
轨迹和图片阶段每约 20 秒报告全局完成数量、百分比及耗时，等待较慢的进程时也继续报告。
这些步骤使用 CPU 和磁盘，不加载模型权重；单个大 JSONL 的解析仍由一个进程完成，
磁盘吞吐和进程间数据汇总也可能限制加速比例。

输出目录中的 `preflight_cache.json` 仅保存编码检查结果，不保存新对话、图片张量或模型权重。
每次 `train` 都重新核对源文件内容、样本集合、模型及模板配置、代码和处理器相关依赖版本。
全部匹配时，日志显示“命中预检查缓存”，跳过逐样本编码；GPU 数量、每卡 batch 和
梯度累积步数变化不影响缓存。长度预算、图片、轨迹或编码配置变化会重新检查。
首次运行新代码或恢复时原样本集合不同，仍需完成一次编码并建立缓存。
删除该文件即可强制重新编码检查。正式训练取样时仍正常编码，不把图片张量长期保存在内存或磁盘。
编码缓存命中前仍会完成一次分摊后的轨迹读取和图片完整性检查，不复用旧图片检查状态。

本优化只影响训练的数据预检查；`check-resume` 保持轻量检查流程。

恢复时检查数据、图片、模板、模型配置及关键训练参数是否变化。允许调整训练总步数和日志／保存频率。

要在同一个基础模型上继承上一阶段的 LoRA 权重、改用另一份配置的数据训练，设置
`model.init_from` 为上一阶段的 checkpoint 或 `final` 目录，并保持新阶段的
`method.resume_from=null`。基础模型路径、LoRA 设置必须与来源一致；FSDP 分片还要求
适配器设置一致，并保留来源运行目录中的 `resolved_config.yaml`。新阶段可以修改数据、
学习率和训练时长，使用新的输出目录与 W&B run，优化器、调度器和步数从头开始。
当前这个入口支持 LoRA/QLoRA，直接读取 FSDP2 checkpoint 中的可训练权重，无需合并基础模型。
`model.init_from` 默认在公共模型配置中显式填写为 `null`，路径相对于当前 YAML 所在目录。

例如，View-mix 的 `checkpoint-40` 接着训练 Interaction-mix：

```bash
WANDB_NAME=qwen38-27b-interaction-from-view-v1 \
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
OMP_NUM_THREADS=4 \
pixi run python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 \
  -m harness_training.cli train \
  --config configs/sft_interaction_mix_27b_fsdp.yaml \
  model.init_from=../outputs/post_training/View-mix/qwen38_27b_qlora_fsdp_epoch1_v1/checkpoints/checkpoint-40 \
  method.resume_from=null \
  experiment.output_dir=../outputs/post_training/Interaction-mix/qwen38_27b_from_view_epoch1_v1 \
  tracking.entity=lingada60-institut-polytechnique-de-paris \
  tracking.project=huggingface
```

新阶段中断后仍用新阶段的 config 和相同 `model.init_from`，将 `method.resume_from`
改成新输出目录下的 checkpoint；这时恢复新阶段自己的优化器和步数。

输出包含配置快照、数据报告、对象划分、拒绝原因、checkpoint 和 `final`。LoRA 保存 adapter 与 Processor；全参数训练保存完整模型。验证集 loss 由 SFT 后端计算。实际 Blender 重建质量和交互任务评估尚未接入。

交互场景仅使用该步骤历史中选定的 `additional_tools` 声明，不回退到静态工具文件。选定历史内的 system/developer 指令按顺序合并为 Qwen 的 system 输入；归档角色不修改。function 参数从 JSON 字符串解析为对象；custom 工具根据 YAML 的 `template.custom_tools` 显式映射（当前 `exec: input`），包装成函数调用，原始代码保持不变。工具命名空间和 custom 输入格式作为工具说明保留。未知 custom 映射会记录为编码失败。

训练侧不执行工具，也不实现部署时的工具调用解析或执行环境。部署方需将模型输出的 `exec(input=...)` 还原为原 runtime 的 custom `exec` 调用；本次未接入服务推理。

## 双卡 H100 的 27B 48k 配置

优化仅由 `configs/sft_view_mix_27b_fsdp.yaml` 和
`configs/sft_interaction_mix_27b_fsdp.yaml` 启用，两组任务分别训练、分别保存。
两份入口分别通过 `defaults` 选择公共 Qwen 27B 模型与 FSDP 配置，优化参数直接写在各自入口中；
`method.long_context` 明确配置重算和分块。四个单场景 YAML 同样只引用公共配置组，保留各自原有的训练设置，
不继承其他任务入口，也不另设 `_h100_48k.yaml`。
采用 NF4 QLoRA、FSDP2、PyTorch SDPA 融合注意力、两级 GPU 重算、前馈分块及图片分批编码，不使用 CPU 卸载。
不依赖外部 `flash-attn`；线性注意力仍使用 FLA 和 causal-conv1d，SDPA 禁止回退到 math 后端。
该路径由显式模型子类实现分块损失，不运行时替换 `forward`。
安装、内核检查、测试和试跑命令见 [H100 48k 说明](docs/H100_48K_TRAINING.md)。
本地仅完成静态检查，H100 训练与显存测试待目标服务器执行。

## 使用 vLLM 加载训练 checkpoint

快速切换 checkpoint 的三个通用入口：`scripts/serve_checkpoint.py`、
`scripts/run_3dharness_checkpoint.py`、`scripts/eval_mmsi_checkpoint.py`。
三者根据相同的 `--checkpoint-path` 自动统一服务名，结果按 checkpoint 隔离；
完整命令见 [通用 checkpoint 与 benchmark 启动](docs/CHECKPOINT_BENCHMARKS.md)。

使用 `scripts/serve_checkpoint.py --checkpoint-path ... --base-model-path ...` 启动基础模型与训练后的 LoRA。
其余参数沿用 `vllm serve`；脚本支持标准适配器和 FSDP 分片，只提取 LoRA，不修改原 checkpoint。
客户端须在 `model` 字段填写启动时打印的适配器名称。
完整命令、图片设置、量化说明和并行用法见 [checkpoint 服务说明](docs/SERVE_CHECKPOINT.md)。

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
