# 断点恢复清理清单

> 历史记录：文中的旧训练器、方法层与 checkpoint 模块已在后续重组中删除，当前结构见 [实验主流程阅读顺序](EXPERIMENT_FLOW_GUIDE.md)。


日期：2026-11-09。

当前训练入口只启动新训练。已移除断点恢复、自动选择恢复点、进程数变化后的恢复、恢复兼容性检查和 W&B 记录续接。

## 整体删除的实现

以下路径相对于项目根目录；文件已删除，函数和类列出删除前的名称。

| 文件 | 函数和类 | 原职责 |
| --- | --- | --- |
| `harness_training/training/checkpoint.py` | `resolve_resume()`、`checkpoint_problem()` | 自动选择恢复点、检查 checkpoint 是否可恢复 |
| `harness_training/training/resume.py` | `write_resume_report()`、`resume_stage()`、`check_resume()`、`validate_compatibility()`、`verify_encoded_manifest()` | 恢复报告、源文件检查、配置和编码结果兼容性检查 |
| `harness_training/training/elastic.py` | `DataProgress`、`target_hash()`、`normalized_config()`、`validate_manifest()`、`load_progress()` | 记录并恢复数据位置、检查进程数变化后的恢复条件 |
| `harness_training/training/methods/elastic.py` | `ElasticTrainerMixin`、`ElasticSFTTrainer`、`ElasticLongContextSFTTrainer` | 自定义训练状态、优化器及随机数状态的保存和恢复 |
| `harness_training/data/elastic.py` | `epoch_order()`、`GlobalBatchSampler`、`IndexedTrainingDataset`、`ElasticCollator`、`EpochDataLoader` | 根据恢复位置分配样本、处理末尾补齐和采样轮次 |
| `harness_training/training/progress.py` | `ResumeProgressCallback.on_train_begin()` | 从恢复后的步数开始显示进度和估算耗时 |
| `harness_training/training/legacy_code.json` | 无函数 | 允许恢复的旧代码版本摘要白名单 |

`ElasticTrainerMixin` 删除的方法：`__init__()`、`get_train_dataloader()`、`_init_training_state()`、`_state_model()`、`_load_from_checkpoint()`、`_load_optimizer_and_scheduler()`、`_load_rng_state()`、`_save_checkpoint()`。同时删除 `ElasticSFTTrainer.compute_loss()`。

`GlobalBatchSampler` 删除的方法：`__init__()`、`set_epoch()`、`__len__()`、`__iter__()`。
`IndexedTrainingDataset` 删除的方法：`__init__()`、`__len__()`、`__getitem__()`。
`ElasticCollator` 删除的方法：`__init__()`、`__call__()`；`EpochDataLoader` 删除 `set_epoch()`。

## 保留文件中的清理

| 文件 | 函数或类 | 清理内容 |
| --- | --- | --- |
| `harness_training/cli/training.py` | `main()` | 删除 `check-resume` 动作和自动恢复调用 |
| `harness_training/training/trainer.py` | `Trainer.check_resume()` | 删除方法 |
| 同上 | `Trainer.prepare()`、`Trainer.run()`、`Trainer._manifest()` | 删除恢复样本集合筛选、恢复目录检查、状态检查和恢复上下文；来源清单改为记录完整配置 |
| `harness_training/training/methods/base.py` | `ResumeState`、`TrainingRun`、`BaseTrainingMethod.check_resume()`、`BaseTrainingMethod.verify_resume()` | 删除恢复状态、运行上下文容器和恢复接口 |
| 同上 | `BaseTrainingMethod.run()` | 只接收模型、模板、数据和输出目录 |
| `harness_training/training/methods/sft.py` | `SFTResumeState`、`SFTMethod.check_resume()`、`SFTMethod.verify_resume()` | 删除恢复状态和检查方法 |
| 同上 | `SFTMethod.build_arguments()`、`SFTMethod.run()` | 删除恢复路径参数、恢复后端选择和进度回调，直接启动新训练 |
| `harness_training/training/manifest.py` | `config_identity()`、`saved_source_inventory()`、`differences()`、`display()`、`ResumeMismatch` | 删除为恢复而归一化配置、重新核对历史来源和报告差异的逻辑 |
| `harness_training/training/tracking.py` | `record_resume_transition()`、`tracking_session()` | 删除进程数变化记录和旧 W&B 运行续接；保留新运行的记录与结束处理 |
| `harness_training/training/preflight.py` | `PreflightWorkers.__init__()`、`PreflightWorkers.log()` | 删除恢复检查使用的日志分支 |
| `harness_training/checkpoints.py` | `Checkpoint.optimizer_shards`、`Checkpoint.manifest_path` | 删除仅供恢复使用的路径属性 |
| `harness_training/config.py` | `MethodConfig`、`SFTMethodConfig.validate_training()`、`TrackingConfig`、`parse_config()` | 删除恢复参数、相关校验和恢复路径解析 |

## 配置和测试

- 从公共方法配置和 12 份实验配置中删除 `method.resume_from`、`method.resume_policy`。
- 从 W&B 配置中删除 `tracking.new_run_resume`、`tracking.checkpoint_resume`。
- 按项目要求，TRL 后端字段仍完整显式配置；`method.backend.resume_from_checkpoint` 固定为 `null`，拒绝传入恢复路径。W&B 在线运行显式禁止续接。
- 删除旧代码白名单的包资源声明，依赖版本不变。
- 删除 `test_auto_resume.py`、`test_elastic_resume.py`、`test_resume_hashing.py`、`test_resume_precheck.py`、`test_progress.py`。
- 其余训练、长上下文、分布式训练、适配器迁移及 W&B 测试删除恢复阶段，保留新训练和保存检查；增加旧恢复入口与配置被拒绝的检查。
- 修正相关测试中已重命名的实验配置路径，实验配置的训练参数保持不变。

## 保留的功能

训练前的数据、图片、标签和长度检查，单进程及分布式训练，长上下文训练，checkpoint 和最终模型保存，配置快照、数据报告、来源清单及编码预检查缓存均保留。

`manifest.py` 保留 `backend_versions()`、`code_identity()`、`hash_file()`、`model_sources()`、`hash_source()`；这些函数用于训练来源记录和编码缓存，不再检查恢复兼容性。

`model.init_from` 仍可读取旧适配器初始化新训练；优化器和训练步数从头开始。已有训练输出目录会被拒绝，不覆盖已有 checkpoint。checkpoint 服务及旧适配器读取使用的元数据代码保留。

## 验证

- 配置、训练入口、SFT、长上下文、W&B、适配器迁移和 checkpoint 服务测试：135 项通过，1 项失败。通过项包含真实小模型训练、梯度更新、验证、模型保存及以适配器初始化新训练。
- 失败项为 `test_experiment_config_uses_only_common_defaults[exp/3DCodeVerse/sft-blender_geo_only-27b.yaml]`：配置中已有的 `../../../` 路径将输出解析到仓库外。这一问题在历史重构记录中已有说明，本次未修改该路径或放宽断言。
- 后续针对训练入口、旧恢复配置拒绝、分块损失梯度、多进程图片与标签检查、预检查缓存及传输的测试：35 项全部通过。
- `harness_training/` 和 `test_runtime.py` 的 Pyright 类型检查：0 错误。全项目类型检查仍有其他脚本和测试中的错误，未扩大到修复这些文件的类型问题。
- 剩余测试均可正常收集；Python 语法检查、`git diff --check`、`pixi lock --check` 通过。
- 未启动完整 27B 训练或完整 benchmark，未修改已有训练数据、模型或 checkpoint。
