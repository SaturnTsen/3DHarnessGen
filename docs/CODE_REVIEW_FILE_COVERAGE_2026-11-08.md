# 全量源码审查：逐文件阅读覆盖清单

对应报告：[审查报告](/home/ubuntu/3DHarnessGen/docs/CODE_REVIEW_2026-11-08.md)。基准为 2026-11-08 工作区，HEAD `4fd222c`，含未提交配置迁移。

所有下列文件均已通读；“保留项”或“—”不表示完整运行验证通过。测试执行范围以主报告为准。SHA-256 显示前 12 位，用于区分本次阅读版本。原始完整摘要及检查日志保存在 `temp/code_audit/`。

| 文件 | 行数 | SHA-256 前缀 | 阅读内容 | 对应结论 |
| --- | ---: | --- | --- | --- |
| `configs/command/default.yaml` | 14 | `cac702d115fd` | Hydra 动作和日志运行目录 | R06、R10、R15 |
| `configs/data/trajectory.yaml` | 13 | `166bf0af3411` | 轨迹默认来源、状态选择和对象划分比例 | R01、R03 |
| `configs/distributed/ddp.yaml` | 11 | `339d8a616d24` | 进程策略、后端及设备参数组合 | 跨字段配置说明 |
| `configs/distributed/fsdp.yaml` | 21 | `5159ae953050` | 进程策略、后端及设备参数组合 | 跨字段配置说明 |
| `configs/distributed/single.yaml` | 11 | `887930a46449` | 进程策略、后端及设备参数组合 | 跨字段配置说明 |
| `configs/exp/3DCodeVerse/sft_blender_all_9b.yaml` | 59 | `0706174c15ef` | 独立实验组合、来源、模型初始化与训练差异 | R06；命名与参数一致性说明 |
| `configs/exp/3DCodeVerse/sft_blender_all_9b_new.yaml` | 98 | `eabcba718d54` | 独立实验组合、来源、模型初始化与训练差异 | R06；命名与参数一致性说明 |
| `configs/exp/3DCodeVerse/sft_blender_geo_27b.yaml` | 69 | `4444d0ad1805` | 独立实验组合、来源、模型初始化与训练差异 | R06；命名与参数一致性说明 |
| `configs/exp/3DCodeVerse/sft_blender_geo_r1_9b.yaml` | 57 | `8691225cc485` | 独立实验组合、来源、模型初始化与训练差异 | R06；命名与参数一致性说明 |
| `configs/exp/3DCodeVerse/sft_cadquery_9b.yaml` | 68 | `40ad616eb422` | 独立实验组合、来源、模型初始化与训练差异 | R06；命名与参数一致性说明 |
| `configs/exp/3DCodeVerse/sft_r2_9b.yaml` | 71 | `6eb5dd6ecc19` | 独立实验组合、来源、模型初始化与训练差异 | R06；命名与参数一致性说明；R03 |
| `configs/exp/3DCodeVerse/sft_r3_9b.yaml` | 73 | `0678efc5b7e5` | 独立实验组合、来源、模型初始化与训练差异 | R06；命名与参数一致性说明；R03 |
| `configs/exp/3DCodeVerse/sft_r4_9b.yaml` | 77 | `4d7c71ee0e0e` | 独立实验组合、来源、模型初始化与训练差异 | R06；命名与参数一致性说明；R03 |
| `configs/exp/3DCodeVerse/sft_threejs_9b.yaml` | 56 | `54b91194ea60` | 独立实验组合、来源、模型初始化与训练差异 | R06；命名与参数一致性说明 |
| `configs/exp/3DHarnessBench/sft_interaciton_cleaned_mix_r2_27b.yaml` | 265 | `0263ce0af8b5` | 独立实验组合、来源、模型初始化与训练差异 | R06；命名与参数一致性说明 |
| `configs/exp/3DHarnessBench/sft_view_mix_r1_27b.yaml` | 64 | `1711c68f68cb` | 独立实验组合、来源、模型初始化与训练差异 | R06；命名与参数一致性说明 |
| `configs/exp/3DHarnessBench/sft_view_mix_r1_27b_new.yaml` | 96 | `14330e6d0c2c` | 独立实验组合、来源、模型初始化与训练差异 | R06；命名与参数一致性说明 |
| `configs/method/sft.yaml` | 151 | `ef4741a0138c` | 完整 SFT 参数和训练器入口选项 | R12、R13 |
| `configs/model/qwen.yaml` | 66 | `a8ccc329ef6e` | 基础模型、LoRA、加载与长度参数 | R08、R14 |
| `configs/model/qwen35_9b.yaml` | 5 | `bb4b8e09307a` | 基础模型、LoRA、加载与长度参数 | R08、R14 |
| `configs/model/qwen38_27b.yaml` | 5 | `ffbb5635cb45` | 基础模型、LoRA、加载与长度参数 | R08、R14 |
| `configs/quantization/4bit.yaml` | 15 | `9c81209c5bbc` | 量化和训练精度配置 | 跨字段配置说明 |
| `configs/quantization/none.yaml` | 3 | `856c9e691331` | 量化和训练精度配置 | 跨字段配置说明 |
| `configs/scene/active_visual.yaml` | 6 | `c9bfcec0c579` | 场景、种子与实验默认信息 | R03；跨字段配置说明 |
| `configs/scene/full_3d_interaction.yaml` | 6 | `986071fe8975` | 场景、种子与实验默认信息 | R03；跨字段配置说明 |
| `configs/scene/multi_view.yaml` | 6 | `0552c893cd84` | 场景、种子与实验默认信息 | R03；跨字段配置说明 |
| `configs/scene/single_view.yaml` | 6 | `bdb293129400` | 场景、种子与实验默认信息 | R03；跨字段配置说明 |
| `configs/template/qwen.yaml` | 19 | `f90ca6c6394d` | 图片预算、文本格式与目标标签规则 | 保留项 |
| `configs/tracking/disabled.yaml` | 10 | `320e55a17bf1` | 实验记录、后端回调和命名组合 | 跨字段配置说明 |
| `configs/tracking/wandb.yaml` | 28 | `6ea210f2cad6` | 实验记录、后端回调和命名组合 | 跨字段配置说明 |
| `configs/tuning/full.yaml` | 3 | `73aeabaaf156` | LoRA 或全参数训练选择 | R11、R14：支持范围不同 |
| `configs/tuning/lora.yaml` | 3 | `c968bbfbf90f` | LoRA 或全参数训练选择 | R11、R14：支持范围不同 |
| `download_3dcodeverse.py` | 413 | `0fd7e88d7939` | 数据下载、并发扫描、限流和重试 | 组织建议 |
| `experimental/pytorch_distributed.py` | 28 | `8c3630ba16c5` | 手工两进程发送/接收实验 | 组织建议 |
| `experimental/test_nccl.py` | 17 | `5b3212b26984` | 顶层 NCCL 初始化和集合通信实验 | 组织建议 |
| `experimental/test_vllm.py` | 40 | `823b288ecca2` | 顶层 API 客户端与交互循环 | 组织建议 |
| `fix_urdf_train.py` | 116 | `10dbcc335ded` | URDF 指令/代码围栏检查、备份与显式修复 | 组织建议 |
| `harness_training/__init__.py` | 1 | `fd24f25fb8f0` | 包说明及公开导出 | — |
| `harness_training/cli.py` | 76 | `9c3b10549b84` | 普通 CLI 参数解析和动作分发 | R01、R10 |
| `harness_training/config.py` | 529 | `8adaf9e46038` | 配置模型、组合、跨字段校验和路径解析 | R12、R13 |
| `harness_training/data/__init__.py` | 3 | `610595a454cd` | 包说明及公开导出 | — |
| `harness_training/data/base.py` | 88 | `3943e0bd852f` | 轨迹记录、训练样本、数据集合和准备接口 | R12 |
| `harness_training/data/contract.py` | 419 | `5cb8e52b11f3` | v3 数据结构、JSON 读取及消息/步骤关系校验 | R04、R05、R13 |
| `harness_training/data/elastic.py` | 70 | `f947e4e6f9e5` | 跨进程数稳定的样本顺序、补齐与批次采样 | 保留项 |
| `harness_training/data/trajectory.py` | 189 | `3b5b1d02fe8e` | 轨迹发现、读取、对象划分、图片检查和目标选择 | R01、R03 |
| `harness_training/hydra_cli.py` | 54 | `97072ca75a90` | Hydra 启动、动作校验和批量实验目录分配 | R06、R10 |
| `harness_training/inference.py` | 135 | `c7e0eacdb8d6` | checkpoint 身份、评估命令和客户端启动 | R02、R11、R13、R14 |
| `harness_training/models/__init__.py` | 3 | `496b46576bcf` | 包说明及公开导出 | — |
| `harness_training/models/base.py` | 33 | `5511feb0bf52` | 模型加载、Processor 和保存接口 | R14 |
| `harness_training/models/chunked_loss.py` | 47 | `a827fceaa611` | 有界 logits 的目标交叉熵及梯度重计算 | 保留项 |
| `harness_training/models/kernels.py` | 45 | `47dd7522ab3f` | 优化内核验证和 SDPA 后端作用域 | 保留项 |
| `harness_training/models/qwen.py` | 211 | `df36c587954b` | Qwen/量化/LoRA 加载、初始化检查及保存 | R08、R13、R14 |
| `harness_training/models/qwen_long_context.py` | 286 | `d805ab35d890` | 显式长上下文模型、视觉微批和分块损失 | 保留项 |
| `harness_training/templates/__init__.py` | 3 | `aecd40d50d4d` | 包说明及公开导出 | — |
| `harness_training/templates/base.py` | 36 | `b5ce97470fb3` | 渲染、编码和长度异常接口 | R12 |
| `harness_training/templates/qwen.py` | 120 | `b78bd5461db4` | Qwen 工具/消息转换、图片编码和目标标签 | 保留项；R12 |
| `harness_training/training/__init__.py` | 1 | `724c5e6d0179` | 包说明及公开导出 | — |
| `harness_training/training/checkpoint.py` | 68 | `4af03cc2eda6` | checkpoint 候选文件完整性检查 | R07、R14 |
| `harness_training/training/collator.py` | 57 | `e89651754f74` | 样本索引编码、序列补齐和图片批次合并 | 保留项 |
| `harness_training/training/elastic.py` | 125 | `831aa6b1d0a3` | 弹性恢复身份比较与数据游标重建 | R08、R14 |
| `harness_training/training/manifest.py` | 140 | `4855dc085c63` | 配置/代码/来源指纹、差异和摘要 | R08 |
| `harness_training/training/methods/__init__.py` | 3 | `bd08b3314766` | 包说明及公开导出 | — |
| `harness_training/training/methods/base.py` | 14 | `c78101bbb79f` | 训练方法执行接口 | R12 |
| `harness_training/training/methods/elastic.py` | 202 | `d7e07a0a4762` | 弹性 SFT 采样、恢复与可移植状态保存 | R14；保留项 |
| `harness_training/training/methods/long_context.py` | 33 | `ba90bb54e4c4` | 不返回完整 logits 的 SFT 损失与指标 | 保留项 |
| `harness_training/training/methods/sft.py` | 106 | `22cc269214d5` | SFT 参数映射、后端构建、训练及导出 | R12、R14；warmup 已核对 |
| `harness_training/training/preflight.py` | 258 | `a434310eca53` | 编码检查、缓存、分布式数据交换和进度 | R08 |
| `harness_training/training/progress.py` | 13 | `9099d3d54c7b` | 恢复后的训练进度与速度统计 | 保留项 |
| `harness_training/training/resume.py` | 121 | `020fc647c02e` | 加载权重前的恢复校验及报告 | R08、R09、R14 |
| `harness_training/training/smoke.py` | 142 | `7049894ae4d9` | 有界样本检查与全套测试调度 | R06、R15 |
| `harness_training/training/tracking.py` | 98 | `0b1649d9d9dc` | W&B 生命周期、身份与恢复转换记录 | 保留项 |
| `harness_training/training/trainer.py` | 211 | `3b9ca368e71c` | 实验组织、准备、恢复、指纹和报告 | R03、R12、R14 |
| `harness_training/types.py` | 118 | `eab7399b138b` | JSON、模型输入、报告及 Processor 类型约定 | — |
| `pixi.toml` | 55 | `733288656bf9` | 开发环境依赖、任务与启动器定义 | R05、R10、R11 |
| `pyproject.toml` | 47 | `5dedfd370b15` | 包元数据、安装命令、打包和测试配置 | R10、R11 |
| `scripts/audit_astra_contract.py` | 123 | `d1ea5dd2a6a2` | 轨迹来源、工具、图片和交付覆盖审计 | R05 |
| `scripts/build_trajectory.py` | 322 | `9b00302af7f8` | 旧迭代轨迹导出、图片与重试提示重建 | R04 |
| `scripts/clean_astra_trajectories.py` | 351 | `d5aa5b3b0ead` | v3 目标清洗、历史窗口及图片复制发布 | 保留项 |
| `scripts/eval_mmsi_checkpoint.py` | 7 | `91e4a60de30a` | MMSI 评估启动包装 | R11 |
| `scripts/export_astra_chat_completions.py` | 335 | `f8aad6c934c3` | 归档图片恢复和 Chat Completions 格式发布 | R04：版本边界；R05 关联链路 |
| `scripts/export_sft_trajectories.py` | 675 | `ff0de7ce7a5d` | 多提供方旧格式轨迹导出 | R04 |
| `scripts/post_train.py` | 62 | `dd834ac7354c` | CLI 外部的自动恢复解析和 argv 改写 | R07、R10 |
| `scripts/rebuild_astra_contract.py` | 490 | `9ac81b96e2fb` | 有来源证据的 v3 轨迹重建与发布 | R05；中间协议需同步 |
| `scripts/recover_interactive_contract.py` | 120 | `312b95e1f7fd` | 交互工具上下文恢复与中间字段清理 | R05 |
| `scripts/run_3dharness_checkpoint.py` | 7 | `59674acbd5e4` | 3DHarness 评估启动包装 | R11 |
| `scripts/serve_blender_step1612.sh` | 24 | `214ec20640eb` | 指定一次 Blender 实验的服务参数 | R11；组织建议 |
| `scripts/serve_checkpoint.py` | 273 | `341b4187a9a3` | 适配器识别/提取/校验、vLLM 命令与 CLI | R11、R14 |
| `scripts/sft_common.py` | 243 | `1b7f7c2c7da7` | 旧格式消息、图片存储和轨迹发布 | R04 |
| `tests/download_img.py` | 31 | `1f89a77bf4de` | 手工图片下载辅助函数 | 组织建议 |
| `tests/harness_training/conftest.py` | 112 | `5aeb4ca63842` | 轨迹、Processor、小模型与配置公共 fixture | R06 |
| `tests/harness_training/test_astra_contract_rebuild.py` | 150 | `c8dc08585d2d` | 归档重建、来源、环境和恢复协议测试 | R05 |
| `tests/harness_training/test_auto_resume.py` | 139 | `ca764f15582a` | 自动选择、完整性及实际恢复测试 | R06、R07、R10 |
| `tests/harness_training/test_checkpoint_transfer.py` | 150 | `1a054f1c098f` | 适配器初始化、跨阶段训练和恢复测试 | R06、R14 |
| `tests/harness_training/test_chunked_loss.py` | 56 | `88422bf5e93a` | 原生分块损失和多模态 LoRA 梯度对比 | R06；保留项 |
| `tests/harness_training/test_clean_astra_trajectories.py` | 226 | `a9da79d79224` | 清洗窗口、失败识别、图片和发布测试 | 保留项 |
| `tests/harness_training/test_config_data.py` | 400 | `c7d0880e3b94` | 配置严格性、读取、对象划分和图片测试 | R03、R06、R12 |
| `tests/harness_training/test_distributed.py` | 100 | `1022225212cc` | 真实小模型 GPU/多卡训练保存恢复矩阵 | R06、R15；本轮未执行 |
| `tests/harness_training/test_elastic_resume.py` | 341 | `064c33375ed1` | 跨拓扑采样、数据游标和优化器恢复 | R06、R14；部分纯逻辑已执行 |
| `tests/harness_training/test_hydra.py` | 187 | `6afde41dc7ee` | 组合、路径、参数覆盖和批量入口测试 | R06、R10 |
| `tests/harness_training/test_inference_launchers.py` | 130 | `59229c728137` | 评估名称、结果隔离及参数透传测试 | R02、R11；覆盖缺口已指出 |
| `tests/harness_training/test_long_context.py` | 214 | `9a654f049a45` | 分块损失、梯度、视觉微批及配置约束 | R06；CPU 部分已执行 |
| `tests/harness_training/test_mix_distributed.py` | 117 | `0b3e02c4d895` | 混合场景多卡训练/验证/恢复 | R06、R15；本轮未执行 |
| `tests/harness_training/test_preflight.py` | 278 | `dadeb24297d8` | 分摊检查、缓存、图片、拒绝样本及进度 | R06、R08 |
| `tests/harness_training/test_preflight_transport.py` | 138 | `50f0bd22a34a` | 大记录分块传输和进度异常清理 | R06；本轮未执行 |
| `tests/harness_training/test_progress.py` | 71 | `01182a4bef2d` | 恢复进度与真实训练显示测试 | R06；进度逻辑已执行 |
| `tests/harness_training/test_resume_hashing.py` | 174 | `53a9cd59d693` | 恢复来源分摊摘要与失败传播 | R06、R08、R09 |
| `tests/harness_training/test_resume_precheck.py` | 176 | `7d4d07280718` | 早期恢复拒绝、长度变化和来源一致性 | R06、R08、R09 |
| `tests/harness_training/test_serve_checkpoint.py` | 175 | `94f6fab6fdcb` | 适配器服务、维度校验、提取与 dry-run | R06、R11、R14；大部分已执行 |
| `tests/harness_training/test_sft.py` | 124 | `d91f02b93f8d` | 后端参数完整性和小模型 SFT 保存恢复 | R06、R12 |
| `tests/harness_training/test_templates.py` | 256 | `7dd87096f305` | 目标标签、历史边界、工具和重复图片 | R06；保留项 |
| `tests/harness_training/test_tracking.py` | 128 | `f9a88441de61` | W&B 生命周期和训练恢复身份 | R06 |
| `tests/harness_training/test_trajectory_contract.py` | 292 | `e1f01a6a241b` | v3 结构、索引、工具关系及校验 CLI | R05、R06；纯契约部分已执行 |
| `tests/lab01.py` | 82 | `38fe99f28b43` | 手工大模型文本输入与生成观察 | 组织建议；未执行 |
| `tests/lab02_vlm_processor.py` | 122 | `184567293ee1` | 手工多模态 Processor/图片 token 观察 | 组织建议；未执行 |
| `tests/lab03_multimodal_fusion.py` | 163 | `ce51993055f6` | 手工大模型视觉特征及融合观察 | 组织建议；未执行 |

合计：115 个文件，13,793 行。

补充核对（不计入上述源码行数）：README、serve 文档、AGENTS.md、.gitignore 和 legacy_code.json 的恢复版本记录结构。第三方依赖、外部 benchmark、原始数据、生成目标代码及权重不在本次项目源码阅读范围内。
