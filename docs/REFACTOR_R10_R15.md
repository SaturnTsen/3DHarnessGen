# R10–R15 重构结果

日期：2026-11-08。对应 [源码审查报告](CODE_REVIEW_2026-11-08.md)。本轮允许破坏性修改，旧脚本入口已删除。

| 编号 | 完成的改动 |
| --- | --- |
| R10 | `harness_training.cli` 成为公共入口；Pixi、安装命令、torchrun 调用同一 CLI。自动恢复在配置组合后解析，不修改 `sys.argv`、不重新加载配置。Hydra 保留配置查看与 multirun，复用同一动作分发。 |
| R11 | serve 实现迁到 `harness_training.serving`，评估由 `harness_training.inference` 执行，统一为 CLI 子命令。删除四个旧启动脚本。 |
| R12 | `MethodConfig` 与 `SFTMethodConfig` 分离；SFT 自己校验后端约束并持有恢复状态。`Trainer` 通过方法接口传递运行上下文，不注入 SFT 私有字段。没有增加 GRPO 空实现或注册机制。 |
| R13 | Hydra/OmegaConf 组合配置后直接交给 Pydantic 校验；快照统一由 OmegaConf 读写。删除 `pydantic_yaml` 及依赖锁中的对应包；轨迹 JSON 直接校验。 |
| R14 | checkpoint 布局与元数据集中到 `checkpoints.py`，训练和服务共享 LoRA 参数转换。删除未使用的 `BaseModel.save`；实际保存和恢复仍由训练后端执行。 |
| R15 | smoke 只检查配置指定来源中的有限记录、图片及可选 Processor 编码；不执行 pytest、训练或 benchmark，不改状态过滤或对象划分。 |

## 命令迁移

```text
python -m scripts.post_train ...              → pixi run post-train ...
python -m scripts.serve_checkpoint ...        → pixi run post-train serve ...
python -m scripts.run_3dharness_checkpoint ... → pixi run post-train eval harness ...
python -m scripts.eval_mmsi_checkpoint ...     → pixi run post-train eval mmsi ...
```

训练参数仍使用 `--config <实验配置>` 与 Hydra 覆盖项；自动恢复使用 `method.resume_from=auto`。
服务与评估保留原来的具体选项，支持 `--dry-run`。完整示例见 [README](../README.md) 和 [服务说明](serve-checkpoint.md)。

## 验证

- 255 项回归检查通过，9 项真实训练相关检查明确排除。覆盖配置、原生 Hydra、自动恢复、方法接口、轨迹契约、图片与目标标签、服务命令和小型合成 FSDP 适配器导出。
- `harness_training/` 的 Pyright 类型检查通过：0 错误、0 警告。
- 12 份现有实验配置均可组合、校验与快照往返；配置文件内容保持本轮开始时的状态。
- Pixi 锁文件一致；NumPy 2.5.3、vLLM 0.30.0 及 NumPy dependency override 保留。
- 未启动训练循环、大模型权重加载、vLLM 服务或完整 benchmark。实际 GPU 训练和跨进程恢复仍需在实验机验证。

## 边界

本轮不重写既有 checkpoint 或放宽恢复指纹检查。代码及依赖变化可能使旧 checkpoint 的严格恢复检查失败，应使用检查报告判断兼容性。

审查中的评估结果身份冲突、跨轮对象划分漂移、旧数据导出器覆盖风险、checkpoint 分片完整性检查等其他编号的问题不包含在 R10–R15 中，不能据本轮结果视为已解决。当前配置名称沿用合作者整理后的文件，没有恢复旧命名或改写实验参数。

继续修改实验主流程时，按 [实验流程阅读指南](EXPERIMENT_FLOW_GUIDE.md) 从 `cli.execute → Trainer.run → SFTMethod.run` 阅读。
