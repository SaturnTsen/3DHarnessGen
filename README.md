# 3DHarnessGen

基于已有 v3 轨迹的后训练与 checkpoint 服务。当前实现 SFT；其他训练方法尚未实现。

## 环境

依赖、editable 安装和开发工具统一由 Pixi 管理，使用项目已有的 `.pixi/envs/default`：

```bash
pixi install --locked
pixi lock --check
```

NumPy、vLLM 的固定版本和 NumPy dependency override 保留在 `pixi.toml` / `pixi.lock` 中。

## 入口

Pixi 直接启动 `harness_training/cli/` 下的对应模块：

| 命令 | 脚本 | 用途 |
| --- | --- | --- |
| `pixi run training <动作>` | `harness_training/cli/training.py` | audit、prepare、preview、train、check-runtime |
| `pixi run serve` | `harness_training/cli/serve.py` | 启动 vLLM 服务 |
| `pixi run validate-trajectory` | `harness_training/cli/validate_trajectory.py` | 校验轨迹文件 |

命令入口按业务模块组织：训练流程放在一起，服务和轨迹校验分别独立。Hydra 负责组合配置。

## 配置与数据

实验入口位于 `configs/exp/<数据集>/*.yaml`，通过 `defaults` 选择公共配置组，入口之间不互相继承。
Hydra / OmegaConf 负责组合、覆盖和插值，Pydantic 负责字段及兼容性校验。
项目 `configs/` 内的配置以 `configs/` 为相对路径基准；独立快照以快照所在目录为基准。

数据读取遵循实际组合后的配置：

- `data.files` 非空：读取指定的 v3 JSONL 文件。
- `data.files: null`：从 `data.runs_root/<scene>/<source_model>/*/post_train_trajectory/trajectory.jsonl` 查找。
- 图片默认相对输入 JSONL 解析；显式 `data.image_root` 可以指定图片根目录。
- 数据状态、场景、对象划分、缺图策略与长度预算由配置控制，脚本不追加接受状态。

快照 `resolved_config.yaml` 保存完整配置；不再依赖 `pydantic_yaml`。模型、模板和训练参数必须显式填写。

```bash
pixi run training audit --config configs/exp/3DHarnessBench/sft-view_mix-r1-27b.yaml
pixi run training prepare --config configs/exp/3DHarnessBench/sft-view_mix-r1-27b.yaml
pixi run training preview --config configs/exp/3DHarnessBench/sft-view_mix-r1-27b.yaml --limit 1
pixi run training preview --config configs/exp/3DHarnessBench/sft-view_mix-r1-27b.yaml --limit 1 --render-only
```

`audit` 检查数据，不加载模型或 Processor。`prepare` 保存报告和对象划分。`preview` 加载 Processor；`--render-only` 跳过图片/token 编码，不加载权重。

## 开发机检查

```bash
# audit 会检查配置指定的数据，并且不会加载模型权重。
pixi run training audit --config configs/exp/3DHarnessBench/sft-view_mix-r1-27b.yaml

# preview 会检查指定数量的样本编码，并且不会加载模型权重。
pixi run training preview --config configs/exp/3DHarnessBench/sft-view_mix-r1-27b.yaml --limit 1

# 独立执行 v3 契约校验。
pixi run validate-trajectory path/to/trajectory.jsonl
```

单元测试与实验机集成测试是独立任务。完整测试目录包含 GPU 和多进程训练用例；开发机应按改动范围选择测试。

## 实验机训练

多进程训练使用 PyTorch 自带的 torchrun 启动 `harness_training.cli.training`。进程数由启动参数指定。

```bash
pixi run torchrun --standalone --nnodes=1 --nproc-per-node=2 -m harness_training.cli.training train --config configs/exp/3DHarnessBench/sft-view_mix-r1-27b.yaml
```

当前每次启动新训练，不支持断点恢复。已有训练结果的输出目录会被拒绝，需指定新的 `experiment.output_dir`。`model.init_from` 可以使用旧适配器初始化新训练，优化器和训练步数从头开始。

训练仍保存 checkpoint、最终模型、配置快照、数据报告和来源清单。训练前的编码检查会记录被拒绝的样本，来源清单会记录本次输入的摘要。移除的恢复文件和函数见 [断点恢复清理清单](docs/resume-removal.md)。

## checkpoint 服务

```bash
pixi run serve --checkpoint-path outputs/post_training/<实验>/checkpoints/checkpoint-N --base-model-path ~/models/Qwen3.8-27B --dry-run
pixi run serve --checkpoint-path outputs/post_training/<实验>/checkpoints/checkpoint-N --base-model-path ~/models/Qwen3.8-27B --max-model-len 8192 --port 8000
```

serve 当前支持 LoRA/QLoRA 的标准适配器和 FSDP 适配器导出；使用基础模型权重启动 vLLM。参数中未由项目管理的部分原样传给 vLLM。
服务模块只负责服务执行，训练时的分布式保存由训练后端负责。

本项目只提供模型服务，不生成外部评估配置或启动评估客户端。评估由各 benchmark 工程自行运行，连接服务地址并使用服务输出的模型名称。
详细参数见 [checkpoint 服务](docs/serve-checkpoint.md)。

## 数据运维与源码结构

轨迹清洗仍使用独立的数据运维模块：

```bash
pixi run python -m scripts.build_sft_trajectories.clean_astra_trajectories
```

清洗器保留 v3 契约，输出到 `data/trajectory_cleaned/`，不覆盖已有输出。历史旧格式导出器不是当前训练入口。

```text
harness_training/
  config.py              配置契约、组合及快照读写
  serving.py             适配器服务和 vLLM 启动
  data/                  v3 契约、读取、对象划分和目标选择
  dataloader/            样本索引、模板编码、批次补齐和图片合并
  templates/             模型输入和目标标签
  models/
    base.py              模型接口
    qwen/
      model.py           Qwen 模型、Processor 加载、初始化与环境检查
      long_context.py    自定义长上下文模型结构
  training/              BaseTrainer 公共流程、SFTTrainer 及训练后端扩展
  manifest.py            来源记录、文件指纹与版本
  tracking.py            实验追踪会话
  cli/
    training.py          训练全流程：检查、准备、预览和训练
    serve.py             服务命令参数与进程启动
    validate_trajectory.py 轨迹文件校验命令
scripts/                 数据清洗与历史运维工具
```

`BaseTrainer` 提供数据准备、检查、预览和内部报告保存能力；`SFTTrainer` 直接组织参数校验、后端初始化、训练、验证与最终保存，并持有本次运行的后端。CLI 按 `method.name` 显式构造具体训练器。`MethodConfig` 定义共用设置，`SFTMethodConfig` 定义 SFT 参数及兼容性约束；新增方法时再增加具体训练器、配置类型和构造分支。

长上下文模型结构保留在 `models/qwen/long_context.py`；分块损失与训练指标由 `training/long_context.py` 和 `training/chunked_loss.py` 负责。

`SFTDataLoader` 继承 PyTorch 的 `DataLoader`，并在内部完成多模态批次组装。训练器通过这个加载器读取训练集和验证集，并由 Accelerate 处理分布式分配；项目不再提供独立的 `SFTCollator`。
