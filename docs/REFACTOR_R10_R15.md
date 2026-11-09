# R10–R15 重构结果

日期：2026-11-08。对应 [源码审查报告](CODE_REVIEW_2026-11-08.md)。本轮允许破坏性修改，旧脚本入口已删除。

| 编号 | 完成的改动 |
| --- | --- |
| R10 | Pixi 直接启动各个 `harness_training/cli/` 模块，删除跨业务的统一入口、console 命令及独立 Hydra 入口。训练相关动作集中于 `harness_training/cli/training.py`，服务和评估分别独立；自动恢复在配置组合后解析，不改写 `sys.argv` 或重新加载配置。多进程直接由 torchrun 启动训练脚本。 |
| R11 | 服务与评估的参数解析和进程启动分别放在 `harness_training/cli/serve.py`、`harness_training/cli/evaluate.py`；适配器处理、checkpoint 解释等核心实现保留在包内。 |
| R12 | `MethodConfig` 与 `SFTMethodConfig` 分离；SFT 自己校验后端约束并持有恢复状态。`Trainer` 通过方法接口传递运行上下文，不注入 SFT 私有字段。没有增加 GRPO 空实现或注册机制。 |
| R13 | Hydra/OmegaConf 组合配置后直接交给 Pydantic 校验；快照统一由 OmegaConf 读写。删除 `pydantic_yaml` 及依赖锁中的对应包；轨迹 JSON 直接校验。 |
| R14 | checkpoint 布局与元数据集中到 `checkpoints.py`，训练和服务共享 LoRA 参数转换。删除未使用的 `BaseModel.save`；实际保存和恢复仍由训练后端执行。 |
| R15 | smoke 只检查配置指定来源中的有限记录、图片及可选 Processor 编码；不执行 pytest、训练或 benchmark，不改状态过滤或对象划分。 |

## 命令迁移

```text
python -m scripts.post_train ...              → pixi run training train ...
python -m harness_training.cli.serve_checkpoint ...        → pixi run serve ...
python -m scripts.run_3dharness_checkpoint ... → pixi run eval harness ...
python -m scripts.eval_mmsi_checkpoint ...     → pixi run eval mmsi ...
```

训练参数仍使用 `--config <实验配置>` 与 Hydra 覆盖项；自动恢复使用 `method.resume_from=auto`。
Pixi 按业务模块启动脚本；训练相关动作由 `harness_training/cli/training.py` 统一处理，服务、评估与轨迹校验各自独立。命令模块随包安装，并包含在包内代码指纹中。服务与评估支持 `--dry-run`。完整示例见 [README](../README.md) 和 [服务说明](serve-checkpoint.md)。

## 验证

- 训练动作归并时检查过参数解析、数据状态过滤、自动恢复与独立服务入口；随后按要求仅将入口迁移到 `harness_training/cli/` 并更新引用，未再运行验证。
- `harness_training/` 和本次实验脚本的 Pyright 类型检查通过：0 错误、0 警告。
- 命令模块随包安装，不设置跨业务的统一入口或 console 命令。
- 12 份现有实验配置均可组合、校验与快照往返；实验配置中仅删除已无用途的 `/command` 默认组引用，实验参数保持不变；同时删除 `configs/command/default.yaml`。
- Pixi 锁文件一致；NumPy 2.5.3、vLLM 0.30.0 及 NumPy dependency override 保留。
- 未加载大模型权重、启动 vLLM 服务或执行完整 benchmark。实际 GPU 训练和跨进程恢复仍需在实验机验证。

## 边界

本轮不重写既有 checkpoint 或放宽恢复指纹检查。代码及依赖变化可能使旧 checkpoint 的严格恢复检查失败，应使用检查报告判断兼容性。

现有 `sft_blender_geo_27b.yaml` 的 `../../../` 路径会将输出目录与数据文件解析到仓库外，对应配置检查曾失败；实验参数和该检查断言均保留，交由配置侧处理。

审查中的评估结果身份冲突、跨轮对象划分漂移、旧数据导出器覆盖风险、checkpoint 分片完整性检查等其他编号的问题不包含在 R10–R15 中，不能据本轮结果视为已解决。当前配置名称沿用合作者整理后的文件，没有恢复旧命名或改写实验参数。

继续修改实验主流程时，按 [实验流程阅读指南](EXPERIMENT_FLOW_GUIDE.md) 从 `harness_training.cli.training.main → Trainer.run → SFTMethod.run` 阅读。
