# 使用 vLLM 加载训练 checkpoint

快速切换 9B / 27B checkpoint 并启动两个 benchmark，使用[三个通用脚本](CHECKPOINT_BENCHMARKS.md)。
服务脚本支持 `--model-name` 作为 `--adapter-name` 的同义参数；省略时客户端自动使用相同命名规则。

通过现有 Pixi 环境运行 `scripts/serve_checkpoint.py`。脚本接收 checkpoint 路径，加载对应基础模型与 LoRA；不合并、修改原始模型或 checkpoint，不加载优化器或训练 RNG 状态。

```bash
CUDA_VISIBLE_DEVICES=3 \
  pixi run python scripts/serve_checkpoint.py \
  --checkpoint-path ~/3DHarnessGen/outputs/post_training/Blender/qwen35_9b_qlora_fsdp_mix_epoch1_v1/checkpoints/checkpoint-1612 \
  --base-model-path ~/models/Qwen3.5-9B \
  --adapter-name blender-9b-step1612 \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 73728 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.92 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":8,"video":0}' \
  --host 0.0.0.0 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --enable-sleep-mode \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

环境变量 `VLLM_API_KEY` 由 vLLM 原生读取，无需把密钥作为参数传入。其他环境变量（例如 `HF_HUB_CACHE`）同样保持正常 vLLM 用法。以上长度、并发和显存设置需在目标 GPU 上验证；脚本不会自动改变它们。

## 脚本参数

- `--checkpoint-path`：必填，接受本项目训练保存的 checkpoint 或 `final` 目录。
- `--base-model-path`：可选，指定本地基础模型目录，优先于适配器和训练快照记录的旧路径。迁移服务器后可使用它修正路径；模型目录必须直接包含 `config.json` 和 safetensors 权重或索引。
- `--adapter-name`：可选，客户端请求的模型名称。不填时，checkpoint 使用 `<训练目录名>-checkpoint-<步数>`，final 使用 `<训练目录名>-final`。启动时打印名称。
- `--dry-run`：检查适配器、准备必要的小文件并打印 vLLM 参数，不启动服务、不加载基础模型张量。打印时隐藏 `--api-key` 的值。

其余参数原样传给 `vllm serve`，不需要分隔符 `--`，也支持使用该分隔符。脚本管理基础模型位置、`--enable-lora` 和 `--lora-modules`；这些参数不能用来换成其他模型或适配器。`--max-lora-rank` 自动根据 checkpoint 设置；显式传入时必须不小于实际 rank。

服务同时列出基础模型和适配器。**请求必须使用适配器名称**，例如：

```python
response = client.chat.completions.create(
    model="blender-9b-step1612",
    messages=[{"role": "user", "content": "生成一个 Blender 脚本。"}],
)
```

不要从 `/v1/models` 中直接取第一项作为训练后模型，它可能是基础模型。

## 两种 checkpoint 格式

1. **标准 PEFT**：已有 `adapter_config.json` 和 `adapter_model.safetensors` 时直接加载。指定的 `checkpoint-1612` 属于这种格式。
2. **FSDP 分片**：包含 `pytorch_model_fsdp_0/.metadata` 时，只读取 LoRA A/B 张量，导出到项目 `temp/vllm_adapters/` 下按源文件指纹区分的目录。适配器配置优先读取 checkpoint 自身；缺少时从同一训练目录的 `resolved_config.yaml` 重建。不会聚合基础权重或优化器，原始分片保留。

当前支持本项目 `qwen3_5` 多模态架构的标准 LoRA/QLoRA（`bias: none`、无 `modules_to_save`）。DoRA、aLoRA、BDLoRA、逐模块 rank/alpha 模式及全参数 checkpoint 会明确拒绝。标准适配器仅接受 safetensors。

检查器读取基础权重的 safetensors 形状头信息，核对每组 LoRA A/B 的维度和 rank；不加载基础模型张量。形状一致不能证明两个目录的基础权重内容相同，迁移时仍应使用原始基础模型的同一版本。

## Tokenizer、图片与量化

checkpoint 有 tokenizer 时，脚本在 `temp/vllm_adapters/` 准备一份小文件目录，配上基础模型的 `config.json`，供 vLLM 探测使用；不会复制模型权重。聊天模板优先使用 checkpoint 的 `chat_template.jinja`。图片尺寸预算从保存的 `processor_config.json` 提取为 `--mm-processor-kwargs`。用户显式提供 `--tokenizer`、`--chat-template`、`--mm-processor-kwargs` 时，相应参数优先。

训练时的 NF4 是训练加载方式，适配器本身不包含四位基础模型。默认推理加载原始基础权重，精度由 vLLM 和基础模型配置决定；推理量化须另外显式传入并验证。

`method.long_context` 中的重算、MLP 分块和训练损失不参与 vLLM 推理。它们保持原参数名称，无需加载训练子类。

## 并行与额外服务参数

例如三张可见 GPU 各运行一个完整模型副本：

```bash
CUDA_VISIBLE_DEVICES=0,1,3 pixi run python scripts/serve_checkpoint.py \
  --checkpoint-path /path/to/checkpoint \
  --base-model-path /path/to/base-model \
  --tensor-parallel-size 1 --data-parallel-size 3 --data-parallel-backend mp \
  --max-model-len 73728 --port 8000
```

这沿用 vLLM 原生并行规则；训练使用的卡数与推理副本数独立。长上下文、多图片与并发数是否适合显存，需要在实际硬件上验证。

`--middleware` 等额外参数原样传入。示例中的 `system.vllm_responses_compat.ResponsesCompatibilityMiddleware` 不在当前仓库内；只有服务环境能导入该模块时才能加上。工具解析器及 `/v1/responses` 的具体兼容行为由 vLLM 与所选 middleware 负责。

## 已验证范围

两个 benchmark 共用的服务预设为 `scripts/serve_blender_step1612.sh`：指定该 checkpoint 与基础模型，监听本机 8000 端口，关闭 thinking，允许每题 10 张图片。使用 `CUDA_VISIBLE_DEVICES=3 bash ~/3DHarnessGen/scripts/serve_blender_step1612.sh` 启动，可在末尾追加覆盖参数或 `--dry-run`。

客户端别名均为 `blender-9b-step1612-local`：`~/3DHarnessBench` 使用 `pixi run single-view --agent blender-9b-step1612-local ...`，`~/spatial_benchs/MMSI-Bench` 使用 `pixi run eval-blender-9b-step1612-local`。详细两步命令分别保存在两个仓库的 `configs/BLENDER_CHECKPOINT.md` 中。

实际 `checkpoint-1612` 已使用 vLLM 0.30.0、原始 BF16 Qwen3.5-9B 和单张 RTX A6000 启动。
实测设置为 `max-model-len=4096`、单并发、`enforce-eager`：适配器注册、文本、图片、指定函数工具调用及原生 `/v1/responses` 文本请求均成功。同一提示下，基础模型与适配器的 token 概率不同，确认训练权重参与推理；这不等于质量评估。

7 项自动化测试覆盖参数透传、迁移基础模型路径、形状不兼容拒绝、两种 FSDP 张量命名、从训练快照重建适配器配置、源文件不变和密钥输出隐藏；类型检查通过。三副本并行、长上下文、sleep mode 和外部兼容 middleware 尚未实测。
