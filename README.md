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

所有普通动作从同一个 CLI 启动。以下三种方式调用相同的 `harness_training.cli:main`：

```text
pixi run post-train ...
post-train ...
python -m harness_training.cli ...
```

分布式任务使用 `pixi run post-train-distributed ...`，由 torchrun 启动同一个 CLI。
当前 Pixi 分布式任务固定使用两个进程；其他进程数由实验机上的 torchrun 启动参数指定。

```bash
pixi run post-train --help
pixi run post-train serve --help
pixi run post-train eval --help
```

原生 Hydra 入口 `post-train-hydra` 保留给配置展开和批量运行，只负责参数适配，和普通 CLI 共用动作分发、恢复选择与运行服务。没有独立的训练或恢复实现。

## 配置与数据

实验入口位于 `configs/exp/<数据集>/*.yaml`，通过 `defaults` 选择公共配置组，入口之间不互相继承。
Hydra / OmegaConf 负责组合、覆盖和插值，Pydantic 负责字段及兼容性校验。
项目 `configs/` 内的配置以 `configs/` 为相对路径基准；独立快照以快照所在目录为基准。

数据读取遵循实际组合后的配置：

- `data.files` 非空：读取指定的 v3 JSONL 文件。
- `data.files: null`：从 `data.runs_root/<scene>/<source_model>/*/post_train_trajectory/trajectory.jsonl` 查找。
- 图片默认相对输入 JSONL 解析；显式 `data.image_root` 可以指定图片根目录。
- 数据状态、场景、对象划分、缺图策略与长度预算由配置控制，CLI 不追加接受状态。

快照 `resolved_config.yaml` 保存完整配置；不再依赖 `pydantic_yaml`。模型、模板和训练参数必须显式填写。

```bash
pixi run post-train audit --config configs/exp/3DHarnessBench/sft_view_mix_r1_27b.yaml
pixi run post-train prepare --config configs/exp/3DHarnessBench/sft_view_mix_r1_27b.yaml
pixi run post-train preview --config configs/exp/3DHarnessBench/sft_view_mix_r1_27b.yaml --limit 1
pixi run post-train preview --config configs/exp/3DHarnessBench/sft_view_mix_r1_27b.yaml --limit 1 --render-only
```

`audit` 检查数据，不加载模型或 Processor。`prepare` 保存报告和对象划分。`preview` 加载 Processor；`--render-only` 跳过图片/token 编码，不加载权重。

## 开发机检查

```bash
# 只检查给定配置的少量记录和图片，默认不加载 Processor。
pixi run post-train smoke --config configs/exp/3DHarnessBench/sft_view_mix_r1_27b.yaml --limit 1

# 可选地检查同一批样本的 Processor 编码和目标标签，不执行训练。
pixi run post-train smoke --config configs/exp/3DHarnessBench/sft_view_mix_r1_27b.yaml --limit 1 --encode

# 独立执行 v3 契约校验。
pixi run post-train validate-trajectory path/to/trajectory.jsonl
```

smoke 同时限制读取记录数和检查目标数，不改对象划分、不扩大状态过滤、不运行 pytest、训练或 benchmark。报告写到 `temp/smoke-post-training-*/`。

单元测试与实验机集成测试是独立任务。完整测试目录包含 GPU 和多进程训练用例；开发机不要把运行整个目录当作 smoke。

## 实验机训练与恢复

```bash
pixi run post-train-distributed train --config configs/exp/3DHarnessBench/sft_view_mix_r1_27b.yaml
pixi run post-train-distributed check-resume --config configs/exp/3DHarnessBench/sft_view_mix_r1_27b.yaml method.resume_from=auto
pixi run post-train-distributed train --config configs/exp/3DHarnessBench/sft_view_mix_r1_27b.yaml method.resume_from=auto
```

`auto` 在加载一次配置后解析为最新的候选 checkpoint，然后进入相同的恢复校验流程。
显式路径也可用于 `method.resume_from`。`model.init_from` 表示用旧适配器初始化一个新训练阶段，不恢复旧优化器。

恢复仍检查配置、数据、代码与依赖版本指纹。本次重构没有改写已有 checkpoint 或它们的指纹；旧运行是否可严格恢复，需要按原有兼容性规则检查，不能把重构后的代码默认视为旧代码。

Hydra 示例：

```bash
pixi run post-train-hydra --config-name exp/3DHarnessBench/sft_view_mix_r1_27b --cfg job --resolve
pixi run post-train-hydra --config-name exp/3DHarnessBench/sft_view_mix_r1_27b command.action=check-resume method.resume_from=auto
pixi run post-train-hydra --config-name exp/3DHarnessBench/sft_view_mix_r1_27b --multirun command.action=audit method.learning_rate=0.00001,0.00002
```

Hydra 的 `serve` / `eval` 动作通过 `command.args` 将服务或评估参数交给同一 CLI。直接使用普通 CLI 子命令通常更方便。

## 服务与评估

```bash
pixi run post-train serve --checkpoint-path outputs/post_training/<实验>/checkpoints/checkpoint-N --base-model-path ~/models/Qwen3.8-27B --dry-run
pixi run post-train serve --checkpoint-path outputs/post_training/<实验>/checkpoints/checkpoint-N --base-model-path ~/models/Qwen3.8-27B --max-model-len 8192 --port 8000
pixi run post-train eval harness --checkpoint-path outputs/post_training/<实验>/checkpoints/checkpoint-N --setting single-view --dry-run
pixi run post-train eval mmsi --checkpoint-path outputs/post_training/<实验>/checkpoints/checkpoint-N --dry-run
```

serve 当前支持 LoRA/QLoRA 的标准适配器和 FSDP 适配器导出；使用基础模型权重启动 vLLM。参数中未由项目管理的部分原样传给 vLLM 或 benchmark。
服务模块只负责服务执行，训练时的 collective 保存/恢复仍由训练后端负责。

评估模块默认连接 `http://127.0.0.1:8000/v1`，需要先启动相应服务；`--benchmark-root` 指定外部 benchmark 工程。
`-- --dry-run` 可以将 dry-run 传给支持该参数的 benchmark，自身 `--dry-run` 只生成命令。
详细参数见 [服务与评估](docs/serve-checkpoint.md)。

## 数据运维与源码结构

轨迹清洗仍使用独立的数据运维模块：

```bash
pixi run python -m scripts.clean_astra_trajectories
```

清洗器保留 v3 契约，输出到 `data/trajectory_cleaned/`，不覆盖已有输出。历史旧格式导出器不是当前训练入口。

```text
harness_training/
  cli.py                 普通 CLI 和共用动作分发
  hydra_cli.py           Hydra 参数/批量运行适配
  config.py              配置契约、组合及快照读写
  checkpoints.py         共用 checkpoint 布局和元数据
  serving.py             适配器服务和 vLLM 启动
  inference.py           外部 benchmark 客户端
  data/                  v3 契约、读取、对象划分和目标选择
  templates/             模型输入和目标标签
  models/                模型加载与 LoRA 配置转换
  training/trainer.py    实验组织
  training/methods/      具体训练方法与恢复状态
```

`Trainer` 通过训练方法接口传递本次运行上下文，不直接读写 SFT 的内部恢复字段。`MethodConfig` 只定义共用设置；SFT 参数及兼容性约束在 `SFTMethodConfig` 中。新增方法时显式增加对应配置类型和构造分支，不创建工厂或预留空实现。
