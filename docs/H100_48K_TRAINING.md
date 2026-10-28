# Qwen 27B 双卡 H100 48k 训练

目标硬件为两张 H100 80GB，无 NVLink 也可运行 FSDP2。长度预算为 **49,152 个编码后总 token**，包含历史、图片与当前回复。此实现尚未在 H100 上验证，不保证无 OOM；测试由目标服务器执行。

## 配置入口

| 数据范围 | 配置 |
| --- | --- |
| Single-view + Multi-view | `configs/sft_view_mix_27b_fsdp.yaml` |
| ActiveVisual + Full3DInteraction | `configs/sft_interaction_mix_27b_fsdp.yaml` |

两组任务独立训练、分别保存，均复用 `model: qwen38_27b` 的基础模型路径和 49,152 长度预算。优化只在这两份配置中启用：模型参数由各入口的 `model` 段覆盖，重算和分块参数放在 `method.long_context`。`model/` 和四个单场景 YAML 不启用本次优化。

NF4 四位 QLoRA、BF16 计算和量化存储、冻结视觉模块、双卡 FSDP2、每卡批次 1、梯度累积 8、右侧补齐、不拼接样本、不截断。参数与激活都不卸载到 CPU；数据读取和模型初始化仍使用主机内存。

`~/models/Qwen3.8-27B` 只作为路径使用，加载前检查其配置必须为 `qwen3_5` 多模态架构。原始模型文件不会被转换或覆盖。本实现不涉及 FP8。

## GPU 内存优化

- `require_optimized_kernels: true`：训练前导入 FlashAttention-2、FLA、causal-conv1d，并核对 Transformers 实际选择的线性注意力及卷积实现；发现参考路径或不可导入的扩展直接报错。`audit` 和 `preview` 不要求这些训练内核。
- `activation_dtype: bfloat16`：嵌入输出和传入文本解码器的隐藏状态显式使用 BF16，避免 PEFT 对未量化权重的 FP32 转换将残差流也提升为 FP32。FP32 权重保留原训练语义。
- `checkpoint_group_size: 8`：外层每 8 个解码层一组重算，组内保留 Transformers 的逐层非重入检查点。设置 `null` 可关闭外层分组以做对照。不会切断历史梯度。
- `mlp_chunk_size: 2048`：仅将逐词元的前馈网络分块，训练时对每块重算；注意力仍看到完整历史。
- `vision_batch_size: 2`：冻结视觉塔每次处理至多两张图片，所有图片特征均保留在 GPU，保持图片顺序、重复次数和网格。视频帧网格不在单个条目内部切开。
- `loss_chunk_size: 256`：只为当前目标的有效标签计算分块交叉熵，包含正确的下一词元移位；训练和验证均不返回完整 logits。损失归一化接收 Trainer 的累计有效目标词元数，保留梯度累积与跨卡加权。

这些优化由明确的 Qwen3.5 子类实现，参数路径与基础模型一致。没有给模型或上游类动态替换 `forward`。因此 `method.backend.loss_type` **必须填写 `nll`**：分块 NLL 由模型实现，不能再启用会替换 `forward` 的 TRL `chunked_nll`。指标由显式 `LongContextSFTTrainer` 子类汇总计数。此路径针对 Transformers **5.17.0**，升级版本后须先审查模型实现。

FSDP 只逐层包装 `LongContextQwenDecoderLayer`。视觉权重属于外层 FSDP 单元，不能逐块包装视觉塔：不同卡的图片批次数可能不同，逐块包装会引入不一致的通信次数。配置校验会拒绝不兼容的视觉包装设置。

## 在目标服务器安装与检查

使用现有 Pixi 环境。服务器需要与锁定 PyTorch 相容的 CUDA Toolkit、`nvcc` 和 C++ 编译器；两个 CUDA 扩展按锁文件构建，不运行 pip 安装。

```bash
MAX_JOBS=4 pixi install --locked
pixi lock --check

# 检查本机拓扑与当前占用。
nvidia-smi
nvidia-smi topo -m

# 只检查 CUDA 和内核分派，不加载基础模型权重或启动训练。
pixi run post-train check-runtime --config configs/sft_interaction_mix_27b_fsdp.yaml

# 数据和图片检查，不加载 Processor 或权重。
pixi run post-train audit --config configs/sft_interaction_mix_27b_fsdp.yaml

# 只加载 Processor；显示文字、图片引用和标签长度，可能包含很长的历史。
pixi run post-train preview --config configs/sft_interaction_mix_27b_fsdp.yaml --limit 1
```

本次只更新了依赖声明与锁文件，未在开发机安装新加入的 FlashAttention 扩展。

## 由目标服务器执行的测试

核心对照不需要下载模型，使用随机初始化的小 Qwen 检查分块损失、梯度、分组重算及图片网格；CUDA 可用时另执行 BF16 混合注意力对照并检查原生内核。保存恢复用例复用已有本地 Processor 和小模型 fixture；模型目录缺失时该项会跳过，不能把跳过视为验证通过。

```bash
mkdir -p temp/h100_48k_tests
pixi run test-post-training tests/harness_training/test_long_context.py -v -ra \
  --basetemp=temp/h100_48k_tests/pytest

# 新增的双卡 NF4 + 长上下文小模型用例也验证保存和恢复。
pixi run test-post-training tests/harness_training/test_distributed.py -v -ra -k long_context

# 再检查已有模板、标签、量化、多卡保存和恢复行为。
pixi run test-post-training
```

CPU 对照中的模型全部为小模型；CPU/float32 配置明确关闭混合精度及原生内核要求。不要将这些结果当作真实 27B 或 H100 显存验证。

## 真实模型试跑和正式训练

先用独立目录试跑，关闭在线记录与验证，验证保存；正常随机采样的三步不保证覆盖最长或图片最多样本，应另外覆盖这些样本。

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 \
  pixi run post-train-distributed train \
  --config configs/sft_interaction_mix_27b_fsdp.yaml \
  tracking=disabled method.max_steps=3 method.save_steps=3 \
  method.backend.eval_strategy=no method.backend.do_eval=false \
  method.backend.skip_memory_metrics=false \
  experiment.output_dir=../outputs/post_training/Interaction-mix/qwen38_27b_qlora_fsdp_trial_v1
```

试跑时同时记录两卡整卡占用、PyTorch 分配/保留峰值和主机内存；不要把不同显存口径直接相加。验收需包括预算内最长输入、最多图片和最长目标，连续至少三次更新、验证、checkpoint 保存、恢复和最终适配器加载。目标是保留约 8–10 GiB 整卡余量；当前没有已验证的峰值。

正式配置开启原有验证和 W&B，使用独立输出目录；已有训练不会被覆盖：

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 \
  pixi run post-train-distributed train \
  --config configs/sft_interaction_mix_27b_fsdp.yaml
```

恢复时使用原配置、原输出目录，并指定其 `method.resume_from`；配置和代码指纹仍须一致。与旧模型一样，推理加载原基础模型加最终 LoRA 即可；优化子类保持原权重名称，优化参数记录在训练配置快照中。

## 数据覆盖与未验证事项

超长记录会完整跳过并保存拒绝原因，不删除图片、不截断脚本。既有两个交互场景的合计长度统计中，49,152 预算覆盖 2,860/6,941 条；这只是跨场景汇总，对应交互混合任务；实际运行数量以检查和编码报告为准。

本次仅完成代码、配置、依赖锁与静态检查；未在开发机运行新增测试、加载大模型或执行 GPU 训练。目标服务器仍需验证 CUDA 扩展构建、混合注意力正确性、双卡通信、峰值显存和恢复行为。上述试跑命令不等于已验证可训练。
