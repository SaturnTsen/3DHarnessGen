# 3DHarnessGen 全量源码审查报告

> 历史记录：文中的旧训练器、方法层与 checkpoint 模块已在后续重组中删除，当前结构见 [实验主流程阅读顺序](EXPERIMENT_FLOW_GUIDE.md)。


后续状态：R10–R15 已完成重构，具体变更、命令迁移及验证边界见 [重构结果](REFACTOR_R10_R15.md)。以下保留审查时的发现。

2026-11-09 补充：项目已移除评估入口和 `inference.py`，不再生成 MMSI 或 3DHarnessBench 的评估配置或启动其客户端。下文 R02 中的评估结果目录构造已不属于当前项目；服务默认名称规则保留，可用 `--model-name` 显式指定。

同日后续：断点恢复实现及旧代码恢复白名单已移除，见 [清理清单](resume-removal.md)。以下恢复相关发现保留为历史记录。

审查日期：2026-11-08。基准：本地工作区，Git HEAD `4fd222c`，包含当时未提交的配置重命名与文档调整。本轮只审查并交付报告，没有修改源码、配置、依赖、数据或模型。

## 结论

项目具备可用的 SFT 实现，但尚未形成一致的实验框架。主要问题是入口分叉、运行时修改配置、多个版本的数据生成协议共存，以及训练方法接口仍依赖 SFT。只移动 YAML 或合并几个脚本，不能解决这些问题。

需要先处理会影响实验正确性的部分：CLI 会扩大接受的数据状态；评估结果身份可能冲突；分轮训练扩大对象集合后，已有对象的训练/验证归属可能变化；旧导出器能够覆盖当前轨迹文件，但输出不符合训练读取器要求。

`serve` 作为一个独立动作是合理的。问题在于它没有进入项目统一 CLI，其适配器转换、校验等核心实现留在 `scripts/`，而训练和评估各有自己的入口与 checkpoint 解释逻辑。

不建议整体推倒重写。图片与目标标签对齐、因果边界、分块损失、部分恢复检查有明确实现和测试，应在整理时保留。

## 阅读范围与验证边界

逐文件通读了以下 **115 个文件，共 13,793 行**，并在阅读结束后比对文件摘要，范围内文件未在审查期间发生变化：

| 范围 | 文件数 | 行数 |
| --- | ---: | ---: |
| `harness_training/` | 36 | 4,058 |
| `scripts/` | 13 | 3,032 |
| `tests/` | 27 | 4,532 |
| `experimental/` | 3 | 85 |
| `configs/` | 32 | 1,455 |
| 根目录两个 Python、两个 TOML 文件 | 4 | 631 |
| 合计 | 115 | 13,793 |

共 80 个 Python、32 个 YAML、2 个 TOML、1 个 Shell 文件。逐文件职责、问题编号和摘要见 [阅读覆盖清单](/home/ubuntu/3DHarnessGen/docs/CODE_REVIEW_FILE_COVERAGE_2026-11-08.md)。另外核对了 README、serve 文档、开发约定及恢复版本元数据。

范围不包含 `.pixi/` 中的第三方实现、模型权重、训练输出、数据中的目标代码，以及外部 benchmark 工程。没有把这些内容算作已审查的项目源码。没有执行日志中记录的代码。

执行过的有限验证：

- 80 个 Python 文件均通过 AST 语法解析；Shell 脚本通过 `bash -n`。
- 当前 12 个实验 YAML 全部通过 `load_config()` 的组合与 Pydantic 校验；这不代表数据、模型路径和训练资源可用。
- 完整测试入口在收集阶段因缺失 `tools.validate_trajectory` 失败。
- 排除该收集失败文件后，再运行测试，前三项因公共 fixture 引用已不存在的配置而失败，随后停止。
- 独立于这些旧配置路径的一组测试：**116 passed，29 deselected**。覆盖轨迹校验、清洗、服务命令构造、适配器提取、CPU 小模型损失/梯度及采样顺序等。
- 用户说明本机为开发机后，未再增加模型执行验证，只继续静态核对、标准库计算和报告整理。

没有启动完整 SFT/GRPO 实验、27B 权重加载、vLLM 服务或 benchmark；没有验证真实多卡训练、长上下文显存或完整评估结果。

## 一、会影响行为或实验可信度的问题

优先级：P1 表示应在继续实验前处理的启动或正确性问题；P2 表示特定条件下的可靠性问题。静态确认的风险不等于已经发生的数据污染或错误评估。

### R01 · P1：CLI 强行扩大数据筛选范围

位置：[cli.py:16](/home/ubuntu/3DHarnessGen/harness_training/cli.py:16)。

`execute()` 无条件向 `config.data.accepted_statuses` 追加 `UNKNOWN`，随后才创建 `Trainer`。即使 YAML 只允许 `OK`，通过 CLI 调用的 audit、preview、train 也会接受 `UNKNOWN`。

直接使用 `Trainer(config)` 则保留 YAML 的状态选择。因此，相同配置通过命令行和 Python 调用可能得到不同样本集合。警告信息不能替代用户配置；审查或训练数据的选择应该只有一个来源。

建议：CLI 只传递解析后的配置；需要扩大状态范围时，用显式配置或覆盖参数表达，并增加 CLI 与 Python 调用样本集合一致的验证。

### R02 · P1：不同实验的评估身份可能相同，导致结果目录冲突

位置：[inference.py:14](/home/ubuntu/3DHarnessGen/harness_training/inference.py:14)、[inference.py:47](/home/ubuntu/3DHarnessGen/harness_training/inference.py:47)、[inference.py:87](/home/ubuntu/3DHarnessGen/harness_training/inference.py:87)。

默认身份只取训练目录的最后一级名称和 checkpoint 名称，不包含实验完整归属。例如：

```text
outputs/Blender/run/checkpoints/checkpoint-20  → run-checkpoint-20
outputs/threejs/run/checkpoints/checkpoint-20 → run-checkpoint-20
```

`build_harness_command()` 用该身份构造输出目录；`build_mmsi_command()` 用它作为评估配置中的模型键，且默认追加 `--reuse`。共用评估输出根目录时，两次不同实验可能共用结果命名空间。当前配置中也存在不同任务采用同名输出叶目录的情况。

已有测试验证了“不同训练目录名”和“固定服务别名”的隔离，没有覆盖“不同上级实验、相同训练目录名和 checkpoint 号”。

建议：将服务请求名称与稳定的实验/权重身份分开；结果身份包含实验归属、checkpoint 及必要的评估配置身份。不能仅通过调用者自觉换目录保证正确性。

### R03 · P1：扩大数据集合后重新随机划分，可能污染后续验证集

位置：[trajectory.py:78](/home/ubuntu/3DHarnessGen/harness_training/data/trajectory.py:78)、[trajectory.py:151](/home/ubuntu/3DHarnessGen/harness_training/data/trajectory.py:151)，以及 `configs/exp/3DCodeVerse/sft_r2_9b.yaml`、`sft_r3_9b.yaml`、`sft_r4_9b.yaml`。

每轮都对本轮全部对象重新排序、打乱、按比例划分，没有读取前一轮保存的 `object_splits.json`。固定 seed 只能保证相同对象集合的划分一致，不能保证新增对象后原有对象的归属不变。

按当前算法用标准库计算：`Object0Factory` 至 `Object9Factory`、seed=42、验证/测试比例各 0.1 时，`Object4Factory` 属于训练集；扩展到 20 个对象后，它进入验证集。

当前分轮配置会从前一轮模型初始化并扩大数据来源。因此，先前被模型训练过的对象可能成为后续轮次的“验证对象”。单轮内部按对象划分正确，不足以保证多轮实验的验证集独立。

建议：明确跨轮共享的对象划分；为新增对象分配集合时保持已有归属，或在整个分轮计划开始前固定统一划分。已有测试只覆盖固定集合及场景组合，没有覆盖对象集合扩张。

### R04 · P1：旧导出器可覆盖正式轨迹，却输出训练器不接受的格式

位置：[sft_common.py:172](/home/ubuntu/3DHarnessGen/scripts/sft_common.py:172)、[sft_common.py:220](/home/ubuntu/3DHarnessGen/scripts/sft_common.py:220)、[build_trajectory.py:281](/home/ubuntu/3DHarnessGen/scripts/build_trajectory.py:281)、[export_sft_trajectories.py:615](/home/ubuntu/3DHarnessGen/scripts/export_sft_trajectories.py:615)。

`make_row()` 输出顶层 `messages/images/metadata`，没有当前 v3 要求的 `schema_version/model/tools`，消息及步骤结构也不同。`write_trajectory()` 随后通过 `os.replace()` 写入正式 `post_train_trajectory/trajectory.jsonl`，还会清理不再引用的图片。

当前训练读取器只接受 `data/contract.py` 定义的 v3。因而旧导出器运行成功，不代表新产物可以训练；对已有 v3 目录误用这些命令，还会用旧格式替换可用数据。`export_sft_trajectories.py` 还允许用文本占位替代缺失图片，和当前训练链路拒绝缺失图片的处理不一致。

建议：明确停用或迁移旧导出器；所有面向当前训练链路的生成结果在发布前必须使用同一份 v3 校验器校验。历史工具若保留，需要独立的输出位置和清楚的版本边界。当前修复前不要把它们当作通用数据入口。

### R05 · P1：校验器迁移未完成，脚本和测试还依赖不存在的 tools

位置：`pixi.toml:8`、`scripts/rebuild_astra_contract.py:24`、`scripts/audit_astra_contract.py:17`、`scripts/recover_interactive_contract.py:18`、`tests/harness_training/test_astra_contract_rebuild.py:8`、`test_trajectory_contract.py:201`。

实际契约定义位于 `harness_training/data/contract.py`，但上述入口仍导入或执行 `tools.validate_trajectory`；当前没有 `tools/` 目录。

已实际观察到 `pytest tests/harness_training` 在收集阶段报 `ModuleNotFoundError: No module named 'tools'`。Pixi 的 `validate-trajectory` 任务同样指向不存在的文件。

建议：统一复用包内契约定义，将命令行校验动作接到正式入口；同时更新脚本、任务和测试，避免重新复制一份校验器。

还不能只把 import 改到新模块就认定重建器恢复正常：`rebuild_astra_contract.py:148` 创建的中间记录包含 `tools_provenance`、`excluded_events`，`step()` 还添加 `tool_context_index`、`input_provenance`。其 `main()` 在调用 `audit()` 前未统一剥离这些字段，而当前 v3 模型禁止额外字段。`recover_interactive_contract.py:45` 有相应清理步骤，主重建路径没有同样收口。这是静态协议对照发现，需要在修复导入后以合成记录验证。

### R06 · P1：配置重命名尚未同步到默认入口、冒烟测试、公共 fixture 和文档

位置：[hydra_cli.py:25](/home/ubuntu/3DHarnessGen/harness_training/hydra_cli.py:25)、[smoke.py:17](/home/ubuntu/3DHarnessGen/harness_training/training/smoke.py:17)、[conftest.py:95](/home/ubuntu/3DHarnessGen/tests/harness_training/conftest.py:95) 及多处测试/README 示例。

当前实验已更名为 `sft_view_mix_r1_27b.yaml` 等文件；上述代码还引用已不存在的 `sft_view_mix_9b_fsdp.yaml`、`sft_interaction_mix_9b_fsdp.yaml` 等。

已实际观察到，排除 R05 的收集阻塞后，测试在公共配置 fixture 处报 `MissingConfigException`。同时，当前 12 个新 YAML 自身都可以组合和校验，说明这里的直接原因是引用未同步，不能归结为全部新配置都无效。

这是工作区正在进行的迁移状态，应由配置整理工作收尾；但默认入口、测试 fixture 的硬编码本身也使重命名容易漏改。README 中长度、轮数和旧路径还存在不同阶段的说明。

建议：同步默认入口、smoke、fixture、测试参数和文档；用轻量检查验证所有公开命令示例及配置引用存在。配置入口数量动态枚举不能替代对这些固定引用的检查。

### R07 · P2：自动恢复的“完整性检查”没有核对实际分片文件

位置：[checkpoint.py:54](/home/ubuntu/3DHarnessGen/harness_training/training/checkpoint.py:54)、[post_train.py:41](/home/ubuntu/3DHarnessGen/scripts/post_train.py:41)。

FSDP 分支只要求模型和优化器的 `.metadata` 文件存在且非空，没有解析其引用的数据分片，检查 `.distcp` 是否存在和完整。全参数模型的索引文件也可以在对应权重分片缺失时满足当前候选检查。

因此，目录编号、trainer_state、RNG 和元数据都已写出，但数据分片缺失的最新 checkpoint，可能被自动选择；真正的后端加载会在之后失败，而不是自动退到完整的旧 checkpoint。

建议：区分“候选文件检查”和“可恢复性检查”；按存储格式核对索引/元数据引用的载荷。无需在开发机加载完整模型来做这项检查。现有测试使用仅有元数据标记的假目录，未覆盖分片缺失。

### R08 · P2：恢复身份未完整覆盖基础权重和后端依赖

位置：[manifest.py:25](/home/ubuntu/3DHarnessGen/harness_training/training/manifest.py:25)、[manifest.py:39](/home/ubuntu/3DHarnessGen/harness_training/training/manifest.py:39)、[preflight.py:187](/home/ubuntu/3DHarnessGen/harness_training/training/preflight.py:187)。

基础模型来源主要记录模型目录下的 JSON/Jinja，不覆盖基础模型权重；初始化适配器/分片则另有记录。LoRA checkpoint 恢复仍依赖基础模型，如果同一路径下基础权重被替换，而配置及相关元数据保持不变，当前早期指纹无法识别这一变化。

此外，`backend_versions()` 只记录 torch、transformers、trl、pydantic-yaml，没有记录直接参与模型适配及分布式恢复的 PEFT、Accelerate 等版本。预检查缓存比恢复 manifest 多记录部分 Processor 相关来源，二者的身份边界也不完全相同。

这是恢复保证的覆盖缺口，不表示当前模型已经被修改。建议建立可验证的基础模型版本/权重身份，并定义相关依赖版本集合；大权重不必每次全量读取，可以使用受控的不可变制品清单。

### R09 · P2：恢复报告使用项目级固定文件，并行实验会互相覆盖

位置：[resume.py:21](/home/ubuntu/3DHarnessGen/harness_training/training/resume.py:21)。

所有恢复检查都写 `temp/resume_check.json`，临时文件名也相同。两个不同实验同时检查时，报告可能相互覆盖，原子替换使用的临时文件也可能冲突。报告路径不是实验身份的一部分。

建议：按实验及本次启动分配报告路径，继续保持每次启动只有主进程写入。

## 二、入口与职责不一致

### R10 · P2：三个启动路径不是同一个 CLI 的等价入口

位置：`pixi.toml:9`、`pyproject.toml:29`、`scripts/post_train.py:15`、`harness_training/hydra_cli.py:18`。

| 启动方式 | 实际执行路径 | 行为差异 |
| --- | --- | --- |
| `pixi run post-train` | `scripts.post_train → cli` | 预处理 `method.resume_from=auto` |
| 安装后的 `post-train` / `python -m harness_training.cli` | `cli` | 不进行上述自动 checkpoint 选择 |
| `pixi run post-train-hydra` | `hydra_cli → cli.execute` | 动作集合不同，有 multirun，未接入相同的 auto 选择 |

外层脚本解析参数、加载配置、寻找 checkpoint、改写 `sys.argv`，然后 CLI 再解析参数和加载配置。Hydra 的动作枚举没有普通 CLI 中的 `check-resume` 和 `smoke`。

因此，“入口只是薄包装”并不成立：换一个启动方式会换能力。建议让一个正式 CLI 拥有解析和分发职责；自动选择 checkpoint 是它调用的普通服务函数，返回配置/恢复计划，而不是回头修改命令行字符串。

### R11 · P2：serve 的核心实现位于 scripts，发布方式与训练不一致

位置：`scripts/serve_checkpoint.py`、两个 benchmark 启动脚本、`harness_training/inference.py`、`pyproject.toml:28`、`pyproject.toml:34`。

`serve_checkpoint.py` 有 273 行，承担 checkpoint 识别、LoRA 配置重建、FSDP 适配器提取、张量维度校验、Tokenizer 缓存、vLLM 参数和 CLI。它已经是业务模块，而不只是启动脚本。测试直接从 `scripts.serve_checkpoint` 导入这些函数。

训练/评估功能一部分在包内、一部分在 scripts；包只发布 `harness_training*`，console scripts 只注册两个训练入口。源码目录中的 serve/eval 命令没有与安装后的训练命令形成一致接口。`serve_blender_step1612.sh` 又固化了某一次实验路径和参数。

建议：将服务核心代码放回包内，作为统一 CLI 的 `serve` 动作；评估命令同理。保留薄脚本或实验启动示例可以，但它们不应持有额外业务能力。服务可在单独进程运行，不要求把服务进程和训练循环混在一起。

### R12 · P2：训练方法接口看似通用，调用者与配置仍直接依赖 SFT

位置：[trainer.py:161](/home/ubuntu/3DHarnessGen/harness_training/training/trainer.py:161)、[trainer.py:187](/home/ubuntu/3DHarnessGen/harness_training/training/trainer.py:187)、[config.py:368](/home/ubuntu/3DHarnessGen/harness_training/config.py:368)、`training/methods/base.py`。

`BaseTrainingMethod` 只声明 `run()`，但 `Trainer.run()` 多处把方法转换成 `SFTMethod` 并注入/读取 `resume_progress`、`resume_manifest`。这些是实际运行依赖，类型转换不能让它们变成通用接口。

`MethodConfig.name` 接受字符串，但 backend、trainer 的类型固定为 `SFTBackendConfig`、`SFTTrainerConfig`；大量顶层兼容性校验也依赖 SFT 的损失、采样和恢复设定。

`TrainingExample` 指向已有目标回复，`BaseTemplate.encode()` 的约定是生成监督标签。GRPO 需要的生成输入、候选结果、奖励和环境状态还没有明确的数据约定；当前代码没有 GRPO 实现。

建议：先约定训练方法真正共享的输入及恢复状态，把 SFT 专属参数/行为归还给 SFT；新方法通过明确分支接入。无需增加工厂、注册框架或空的强化学习实现。

### R13 · P3：配置与 JSON 轨迹存在重复 YAML 转换

位置：[config.py:505](/home/ubuntu/3DHarnessGen/harness_training/config.py:505)、[contract.py:414](/home/ubuntu/3DHarnessGen/harness_training/data/contract.py:414)、`models/qwen.py:59`、`inference.py:25`。

配置路径为 `OmegaConf → YAML 文本 → pydantic_yaml → TrainingConfig`；轨迹已经使用严格 JSON 读取器解析，还会 `json.dumps()` 后交给 YAML/Pydantic 读取器。显式 image_root 分支却直接 `model_validate()`。

Pydantic 的字段校验与 Hydra 的组合职责互补；多次文本往返增加了格式和类型语义需要维护的位置。`pydantic_yaml` 的使用也扩散到了快照与模型初始化读取，不能只删一个 import 就完成整理。

建议：配置直接从解析后的容器进入 Pydantic，JSON 保留严格解析后直接校验；为快照定义统一读写方式，并验证布尔样式字符串、路径、枚举及严格类型行为。这里是简化建议，不声称现有转换已造成数据错误。

### R14 · P2：checkpoint 与适配器协议在多个模块重复解释

涉及 `checkpoint.py`、`resume.py`、`training/methods/elastic.py`、`models/qwen.py`、`inference.py`、`scripts/serve_checkpoint.py`。

同一组目录布局分别承担候选检查、状态恢复、初始化权重、服务转换和评估身份推导；LoRA 参数构造在 QwenModel 和 serve 中也分别实现。checkpoint/最终产物/基础模型的支持范围及错误处理不同，新增格式时需要多处同步。

`BaseModel.save()` 声明并实现了保存接口，但实际训练使用 TRL 的 `save_model()`，分布式最终导出也在 SFTMethod 内处理。现有模型接口没有完整表达实际保存职责。

建议：共享 checkpoint 元数据、布局识别和适配器配置转换；训练时的 collective 恢复/保存、CPU 服务导出仍保留各自执行逻辑。不能为了“统一”把有不同执行条件的操作粗暴合并。

### R15 · P2：smoke 的名字、范围和成功条件不匹配

位置：`training/smoke.py:115`、`training/smoke.py:134`、`tests/harness_training/test_mix_distributed.py`、`test_distributed.py`。

smoke 除了真实样本/Processor 检查，还启动整个测试目录；目录包含多卡训练和四卡条件测试。任何跳过测试都被 smoke 判为失败，因此缺少对应硬件时，即便适用于当前机器的测试全通过，也无法得到 smoke 成功结论。

这可以作为“完整验证必须无跳过”的严格政策，但不适合作为开发机上的轻量冒烟入口。其固定场景配置映射还忽略了显式 `data.files` 的入口方式，进一步增加维护分支。

建议：区分开发机静态/单元检查、少量样本检查和目标机器集成验证，并分别报告执行、跳过及未验证的范围。用户已经明确本机不承担完整实验。

## 三、其他应收口的项目组织问题

- 配置组会跨顶层字段写入：distributed 修改 method/backend/model，quantization 修改精度，tracking 修改 report_to。不是错误，但必须提供最终生效配置，而不能把文件夹名当作互不影响的边界。
- 实验入口重复填写公共超参数，名称、目录标签、epoch 注释有历史残留；部分 `all` 名称的配置实际只选 geo 文件。以解析结果为准，命名整理时同时校对实验语义。
- 完整显式列出后端字段并检查版本接口，是当前已有开发约定；配置体积大不等于这些字段都可以直接删除。若简化，应明确替代的版本固定及可复现规则。
- `tests/lab01.py`、`lab03_multimodal_fusion.py` 在模块顶层加载大模型；lab02 顶层访问网络。它们不是常规单元测试，当前 pytest 的 testpaths/命名将其排除。宜将其归入明确的手工实验目录并加可控入口；本轮未执行。
- `experimental/test_nccl.py`、`test_vllm.py` 在导入时初始化通信或进入交互。不要把它们混入自动测试或常规包导入。
- 根目录下载器及 URDF 修复器各有独立 CLI、路径规则和依赖使用。下载器会对全部异常无限重试；修复器有明确 apply 开关和备份。它们应被清楚定位为数据运维工具，而非训练主流程。

## 四、应保留的实现与本轮没有确认的问题

1. `TrajectoryDataset` 在单次准备中按对象划分，再选择训练目标；混合场景对同一对象使用同一归属。R03 是跨轮集合变化的问题，不否定这个实现。
2. Step 校验禁止历史包含当前目标或未来消息；QwenTemplate 校验 prompt 与完整序列前缀一致，并只给目标 token 分配有效 labels。
3. 图片出现顺序与重复引用在读取、模板和批次组合中保留，已有对应测试。
4. 清洗器与旧导出器不同：使用 v3 校验，保留目标和来源映射，不覆盖已有清洗输出，并在发布前检查保留图片。
5. 分块损失和长上下文实现有数值/梯度对比测试；本轮通过的是其中的 CPU 小模型检查，不能推广为真实长上下文、多卡训练已验证。
6. 恢复流程已区分初始化权重与恢复优化器，并在加载权重前做兼容性检查；这些功能应在统一入口时保留。
7. `warmup_ratio → warmup_steps` 在安装的 Transformers 5.17.0 中合法：`warmup_steps < 1` 时按总步数比例解释。前面的梳理说明曾将其列为待核对，本轮已核对，不列为 bug。
8. 没有证据证明当前 SFT 数学实现整体错误；已有的 116 项通过检查也不能抵消全套测试当前无法启动的问题。

## 五、建议的整理顺序与分工

| 顺序 | 内容 | 负责人边界 | 可检查的完成条件 |
| --- | --- | --- | --- |
| 1 | 修复失效引用，明确旧导出器版本边界 | 配置合作者处理新入口引用；主流程侧处理包内校验动作和脚本调用 | 公开命令能解析；测试能收集；v3 产物统一校验 |
| 2 | 统一 CLI 与自动恢复入口，接入 serve/eval 动作 | 主流程；配置侧提供统一配置加载接口 | Pixi、console、模块启动经过同一分发和恢复选择；不再改写 sys.argv |
| 3 | 固定数据选择、跨轮划分和评估身份 | 主流程与数据逻辑；配置侧明确数据及评估参数 | 配置不被入口改写；已有对象归属稳定；不同实验结果不冲突 |
| 4 | 收口 checkpoint 元数据和运行状态 | 主流程 | 支持格式有共同识别规则；分片缺失被提前识别；报告按运行隔离 |
| 5 | 明确 SFT 与新训练方法的边界 | 用户负责主流程，合作者负责对应配置结构 | 调用者无需转换成 SFTMethod；新方法的数据、生成和奖励约定明确 |
| 6 | 在实验机器验证生成、评分和少量训练更新 | 实验机器上的后续任务 | 先验证最小闭环，再执行完整实验和多卡验证 |

配置整理不应只做文件重命名；主流程整理也不应只把 `scripts/` 中的全部内容原样搬进 `cli.py`。CLI 负责接收请求和分发，训练、服务、评估、恢复分别由包内明确的函数或类承担。

本报告是审查交付，不代表上述问题已修复。后续实现应按上述边界逐项推进，保留现有可用的训练与数据验证能力。
