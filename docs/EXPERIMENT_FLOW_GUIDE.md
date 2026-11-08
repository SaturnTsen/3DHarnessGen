# 实验主流程阅读顺序

配置整理与训练主流程的连接点是完整的 `TrainingConfig`；配置来源和参数的具体值以实际组合结果为准。配置组职责和当前命令见项目 README。

## 入口

`pixi run post-train`、安装后的 `post-train`、`python -m harness_training.cli` 使用同一个 CLI。
`post-train-distributed` 由 torchrun 启动相同模块。原生 Hydra 入口是配置展开/批量运行的适配层，使用相同的动作分发。

```text
cli.main()
  ├─ 解析命令及配置覆盖
  ├─ load_config()：Hydra 组合 → Pydantic 校验 → 路径解析
  └─ execute()
      ├─ train / check-resume：解析 auto 恢复请求
      ├─ audit / prepare / preview：数据检查和预览
      ├─ smoke：少量记录检查，可选 Processor 编码
      └─ train：Trainer.run()
```

serve、eval 和 validate-trajectory 也从 CLI 分发，但不先构造训练 Trainer。

## 训练主流程

```text
Trainer(config)
  ├─ 明确构造当前支持的训练方法、模型和数据读取器
  └─ Trainer.run()
      ├─ 检查启动条件和输出目录
      ├─ method.check_resume()：必要时检查恢复条件
      ├─ prepare(encode=True)：读取、图片与目标编码检查
      ├─ 构造 manifest，method.verify_resume() 检查准备后的输入
      ├─ 保存配置与数据报告，开启实验记录
      └─ method.run(..., context=TrainingRun(...))
          └─ SFTMethod.run()
              ├─ 构造 SFTConfig
              ├─ 加载模型并创建训练后端及 SFTCollator
              ├─ backend.train()
              └─ 验证并通过后端保存模型和训练状态
```

优先阅读：

1. `harness_training/cli.py`：请求怎样变成一次操作。
2. `harness_training/training/trainer.py`：项目怎样组织一次训练。
3. `harness_training/training/methods/base.py`：训练方法接收的运行上下文与恢复契约。
4. `harness_training/training/methods/sft.py`：SFT 怎样配置和调用后端。
5. `harness_training/training/collator.py` 与 `templates/qwen.py`：真实样本怎样成为图片、token 和目标 labels。

`Trainer` 不再转换成 SFTMethod 后注入内部状态。SFT 的数据游标属于 `SFTResumeState`，只由 SFT 解释。真正的前向、反向传播和参数更新仍由训练后端执行。

## 新训练方法

`MethodConfig` 定义共用设置，`SFTMethodConfig` 定义 SFT 后端参数和兼容性校验。当前只实现 SFT。
新增方法时，应明确其配置类型、数据输入和恢复语义，并在配置解析和 Trainer 的构造分支显式接入。无需工厂、注册机制或提前增加空实现。

GRPO 的下一步仍是确定生成单位和奖励：完整脚本还是多轮工具交互，以及怎样评价候选。当前带目标回复的 SFT 编码不能直接作为在线生成提示。
