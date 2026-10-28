# Blender 9B 最长样本训练压力测试

2026-10-27，测试未通过：最长样本前向完成，但首步反向传播发生 CUDA OOM，未完成任何参数更新，没有生成训练检查点或最终适配器。未启动完整数据集训练。

## 数据与长度核对

使用 `configs/sft_blender_9b_fsdp.yaml`，最大序列长度为 163840。合并两份原始 JSONL，按对象划分后训练集 6460 条、验证集 50 条，数据和图片检查拒绝 0 条。

所有样本按 Qwen 模板渲染，使用真实 tokenizer 与图片尺寸对应的视觉 token 数计算长度。最长 10 条另通过正式模板的完整 Processor 编码，核对长度及非忽略标签与输入 token 一致。全部 6510 条均在长度预算内。

最长样本同时是训练集和整个数据集最长的样本：

- 对象：`3dcodebench/instances_geo/FanCoral_004_geo`
- 步骤：`step_2`
- 来源：`3dcodeverse/3dcodebench/instances_geo_llamafactory_img/trajectory.jsonl` 第 704 行
- 完整输入：90526 token
- 当前目标回复：89648 token
- 图片：4 张；保留原图片预算及出现顺序
- 未截断历史或目标脚本

## 测试配置与范围

硬件为双卡 NVIDIA RTX A6000，标称每卡 48GB。启动前 GPU 0 已有约 2.9GiB 设备显存占用，GPU 1 基本空闲；未终止其他进程。

沿用正式配置的本地 Qwen3.5-9B、NF4 四位量化与双重量化、BF16、LoRA rank=16/alpha=32/dropout=0.05、冻结视觉模块、FSDP2、SDPA、普通非重入梯度检查点，以及 `chunked_nll` 损失。未开启 CPU 或激活卸载，未添加分配器环境变量。

只在内存中选择最长一条原始样本，分布式数据加载为两个 rank 各提供该样本，进行一次更新的压力测试。测试覆盖配置为：

- `method.max_steps=1`
- `method.gradient_accumulation_steps=1`（正式配置为 8）
- `method.warmup_ratio=0`，确保若成功则首步实际更新参数
- `method.save_steps=1`
- `method.backend.eval_strategy=no`、`do_eval=false`
- `method.backend.train_sampling_strategy=sequential`
- `tracking=disabled`
- 输出写入独立的 `temp/blender_longest/run/`

没有修改正式 YAML 或训练实现，没有导出或改写原始训练样本。此次测试未验证正式的 8 次梯度累积；单个微批次已在反向时失败。

复现命令（输出目录存在时应先为复测指定新的独立目录）：

```bash
.pixi/envs/default/bin/python temp/blender_longest/preflight.py
.pixi/envs/default/bin/torchrun --standalone --nnodes=1 --nproc-per-node=2 temp/blender_longest/run_stress.py
```

## 失败结果

| 指标 | rank 0 | rank 1 |
| --- | --- | --- |
| PyTorch 峰值已分配显存 | 39.43 GiB | 44.62 GiB |
| PyTorch 峰值保留显存 | 42.76 GiB | 46.91 GiB |
| 最终失败的额外申请 | 1.38 GiB | 708 MiB |
| 失败时 GPU 空闲显存 | 1.19 GiB | 79.12 MiB |
| 从加载模型到异常的耗时 | 77.39 秒 | 78.38 秒 |

峰值来自 `torch.cuda.max_memory_allocated()` 和 `max_memory_reserved()`，不包含全部 CUDA/NCCL 开销或其他进程占用。两个 rank 均捕获 `OutOfMemoryError`：

- rank 0：`loss.backward` → 梯度检查点重计算 → Qwen3.5 自注意力 `q_norm`。
- rank 1：`loss.backward` → 梯度检查点重计算 → 自注意力 `o_proj` 的 LoRA 线性计算。

这表明当前配置在当前双卡环境中不能完成该最长样本的一次完整训练更新。提高允许的序列长度本身不能降低训练显存；本次未测试消除分配碎片或开启卸载后的结果，不能据此断言其他配置也必然失败。

证据保存在项目 `temp/blender_longest/`：`preflight.json`、`lengths.json`、`stress.log`，以及 `run/resolved_config.yaml`、`run/sample.json`、`run/result_rank0.json`、`run/result_rank1.json`。训练进程退出码为 1，GPU 已释放训练显存。`pixi lock --check` 通过，没有变更环境或依赖。
