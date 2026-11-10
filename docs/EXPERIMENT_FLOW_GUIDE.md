# 实验主流程阅读顺序

配置整理与训练主流程的连接点是完整的 `TrainingConfig`；配置来源和参数的具体值以实际组合结果为准。配置组职责和当前命令见项目 README。

## 入口

各动作由 Pixi 直接启动对应脚本，例如 `pixi run training train`、`pixi run training audit` 和 `pixi run serve`。
多进程训练由 torchrun 启动 `harness_training.cli.training`；Hydra 只负责配置组合。

```text
pixi run training train
  └─ harness_training/cli/training.py:main()
      ├─ 解析 --config 和 Hydra 覆盖项
      ├─ load_config()：组合 → 校验 → 路径解析
      └─ SFTTrainer(config).run()
```

数据检查、准备、预览与训练都在 `harness_training/cli/training.py`；服务、轨迹校验分别由各自脚本调用包内实现。评估由外部 benchmark 工程自行运行。

## 训练主流程

```text
SFTTrainer(config)
  ├─ BaseTrainer 构造数据读取器和模型对象，不加载 Processor 或权重
  └─ SFTTrainer.run()
      ├─ _prepare_run()：检查启动条件、输出目录、图片和目标编码
      ├─ manifest.build_manifest()：记录配置、输入来源、版本和代码摘要
      ├─ 保存配置与数据报告
      ├─ 检查验证样本、转换 SFTConfig 参数
      └─ 开启 tracking_session()，初始化并持有 backend
          ├─ 加载模型，使用 SFTDataLoader 接入 TRL
          ├─ backend.train()，按配置验证
          └─ 保存训练状态、最终模型、Processor 和指标
```

优先阅读：

1. `harness_training/cli/training.py`：按 `method.name` 显式选择具体训练器。
2. `harness_training/trainers/base.py`：公共数据流程、训练前检查和内部报告保存。
3. `harness_training/trainers/sft_trainer.py`：SFT 参数转换、训练主流程与加载器接入。
4. `harness_training/trainers/long_context_sft_trainer.py`：长上下文后端的计算接口扩展、分块损失与指标。
5. `harness_training/utils/manifest.py` 与 `harness_training/trainers/tracking.py`：来源记录和实验追踪会话。

`SFTDataLoader` 管理样本索引、模板编码、补齐和图片合并。`trainers/sft_trainer.py` 中的 `DataLoaderSFTTrainer` 只将项目加载器接入 TRL，采样器及多进程加载由后端和 Accelerate 管理。

当前每次启动新训练，不支持断点恢复。真正的前向、反向传播和参数更新由训练后端执行，checkpoint 和最终模型仍正常保存。

## 新训练方法

`MethodConfig` 定义共用设置，`SFTMethodConfig` 定义 SFT 后端参数和兼容性校验。当前只实现 SFT。
新增方法时，应明确其配置类型和数据输入，并在配置解析和 CLI 的构造分支显式接入。无需工厂、注册机制或提前增加空实现。

GRPO 的下一步仍是确定生成单位和奖励：完整脚本还是多轮工具交互，以及怎样评价候选。当前带目标回复的 SFT 编码不能直接作为在线生成提示。

checkpoint 不另设公共类型或模块。模型初始化、服务与来源记录在各自使用处读取标准适配器、`pytorch_model_fsdp_0` 分片和训练快照。FSDP 最终模型保存由所有进程参与，普通报告和指标由主进程写入。来源记录字段及指纹算法保持不变，长上下文模型只提供隐状态和输出层，训练侧负责分块损失与指标。

## 长上下文计算职责

`models/qwen/qwen_long_context.py` 保留自定义模型结构、分组激活重计算和图片微批处理。训练器通过正常前向调用请求最后一层隐状态，通过 `get_output_embeddings()` 读取输出层；默认前向仍沿用 Transformers 的标准输出。

`trainers/long_context_sft_trainer.py` 的 `compute_loss()` 负责调用同一文件中的 `chunked_cross_entropy()`，汇总准确率和熵，并处理梯度累积与分布式目标数归一化。分块损失在标签移位后仅选取有效目标位置，按 `loss_chunk_size` 投影和重计算损失，不生成完整轨迹的词表分数。模型不再包含项目自定义的损失或指标输出类型。
