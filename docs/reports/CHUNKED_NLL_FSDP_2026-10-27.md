# Single-view 分块损失与 FSDP2 测试

当前双卡 NVIDIA RTX A6000（每卡约 48 GB）下，27B BF16 LoRA 使用
`chunked_nll + FSDP2` 仍不能训练最长的真实样本。失败发生在模型内部激活计算，
不是完整词表分数矩阵的损失计算。本轮最长样本测试未完成任何 27B 参数更新，也未启动完整训练。
后续较短样本已完成真实 27B 更新，详见 [token 预算实测](FSDP_TOKEN_BUDGET_2026-10-27.md)。

## 条件与样本

- 使用项目现有 Pixi 环境，未变更依赖：PyTorch 2.13.0、Transformers 5.17.0、
  TRL 1.13.0、Accelerate 1.15.0、PEFT 0.21.2。
- 原 27B 模型目录的配置声明 `qwen3_5 / Qwen3_5ForConditionalGeneration`，64 层，
  词表大小 248,320。读取配置确认架构，不按目录名称推断。
- BF16 LoRA，不量化，冻结视觉模块；FSDP2、前向后重新分片、普通梯度检查点。
  每卡批次 1，梯度累积 1，关闭验证和在线记录。
- 先按对象划分数据。检查全部 320 条训练样本，长度范围为 2,471–26,749 token。
  在内存中按长度降序选择最长 6 条，使用顺序采样，计划更新 3 步。
  不导出新训练数据，不改原始轨迹，不截断，不缩小实际 27B 测试的图片。
- 最长 6 条的输入长度依次为 26,749、23,748、20,488、16,759、16,068、16,054；
  当前目标长度依次为 12,696、12,025、9,690、7,480、7,870、7,262。
  每条有 2 张图片，检查了图片引用与编码，非忽略标签与原 token 一致。

## 实测结果

| 配置 | 结果 | 失败位置 |
| --- | --- | --- |
| chunked NLL + FSDP2 | 首步前向 OOM，额外申请 628 MiB 失败 | `torch_chunk_gated_delta_rule` 线性注意力 |
| 同上，启用 `PYTORCH_ALLOC_CONF=expandable_segments:True` | 首步前向仍 OOM，额外申请 628 MiB 失败 | 同一位置 |
| 同上，再启用 `method.backend.activation_offloading=true` | 前向完成，首步反向 OOM，额外申请 890 MiB 失败 | 梯度检查点重算前馈网络的门控乘法 |

独占 GPU 的可扩展分配重测中，OOM 时 PyTorch 分配约 44.91 GiB，
保留但未使用约 247 MiB。激活卸载补测对应约 44.90 GiB 与 190 MiB。
减少碎片没有解决最长样本的显存容量问题。

环境缺少 `causal_conv1d` 和 `flash-linear-attention`，实际使用 Transformers 的
PyTorch 参考实现。上述结论只适用于这套已锁定环境和硬件。
尚未验证更换内核、4-bit 量化或其他激活管理方式能否解决这些长样本。

## 功能与回归验证

- 小模型分块损失的训练、验证、保存和恢复测试通过；CPU 回归测试 144 项通过。
- 双卡小模型用最长两条真实文本完成一次更新；小模型专用图片预算为 1,024 像素，
  因此输入为 26,239 和 23,238 token。此项仅验证功能，不代表原模型的显存表现。
- 两个进程各有 39 个分片参数；LoRA B 的非零元素总数为 160；
  最终导出的适配器含 4 个张量，LoRA B 参数非零，确认发生了参数更新。
- 新增 FSDP2 + chunked NLL 自动测试已单独运行通过（1 项），覆盖训练、保存和检查点恢复。

完整回归测试首次启动时包含 GPU 测试，与一次显存重测争用资源。
已停止这些进程，作废该轮重测；上表可扩展分配与激活卸载的结果来自后续独占 GPU 的运行。
CPU 回归重跑显式隐藏 GPU；不将被中止的完整测试集记为通过。

## 配置与记录

`configs/sft_single_view_fsdp_trial.yaml` 提供可复用的 3 步短训配置：

```bash
pixi run post-train-distributed train --config configs/sft_single_view_fsdp_trial.yaml
```

该入口使用正常的数据采样；本次最长样本压力测试另在内存中选择并排序样本，
因此不能把普通 3 步试跑与最长样本覆盖等同。配置保留用于复现，不能视为已解决 OOM。

最长样本测试的配置快照和样本索引分别保存在以下输出目录；这些失败运行没有最终模型：

- `outputs/post_training/Single-view/chunked_nll_fsdp_longest_v1`
- `outputs/post_training/Single-view/chunked_nll_fsdp_longest_v3`
- `outputs/post_training/Single-view/chunked_nll_fsdp_offload_longest_v1`

探索脚本、长度检查和原始错误日志保存在 `temp/chunked_fsdp/`。
