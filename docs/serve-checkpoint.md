# checkpoint 服务与评估

直接使用 `pixi run serve` 和 `pixi run eval <harness|mmsi>`，分别执行 `harness_training/cli/serve.py` 与 `harness_training/cli/evaluate.py`。

## 服务

先在实验机设置 checkpoint 路径：

```bash
CKPT=~/3DHarnessGen/outputs/post_training/View-mix/qwen38_27b_qlora_fsdp_epoch1_v1/checkpoints/checkpoint-40
```

校验并显示命令：

```bash
pixi run serve --checkpoint-path "$CKPT" --base-model-path ~/models/Qwen3.8-27B --dry-run
```

启动服务：

```bash
CUDA_VISIBLE_DEVICES=0,1 pixi run serve \
  --checkpoint-path "$CKPT" \
  --base-model-path ~/models/Qwen3.8-27B \
  --tensor-parallel-size 2 --max-model-len 49152 \
  --host 127.0.0.1 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

GPU 数量、显存与长度预算需要按实验机条件设置；本次重构没有验证上述服务的硬件容量。

当前支持标准 LoRA/QLoRA 适配器和 FSDP 适配器导出。基础模型可以从适配器或训练快照确定，迁移机器时可通过 `--base-model-path` 覆盖。
`--model-name` / `--adapter-name` 指定 API 请求模型名。其余参数传给 vLLM。

`--dry-run` 会准备适配器/Tokenizer 缓存并校验维度，但不会加载基础权重到 GPU 或启动服务。临时产物位于 `temp/vllm_adapters/`，不修改 checkpoint。

## 评估

服务启动后，在另一个终端使用同一 checkpoint 和服务名称：

```bash
pixi run eval harness --checkpoint-path "$CKPT" --setting single-view --dry-run
pixi run eval mmsi --checkpoint-path "$CKPT" --dry-run
```

去掉 `--dry-run` 才会执行 benchmark。`--benchmark-root` 指定外部工程位置，使用那个工程已有的 Pixi 环境；`--base-url` 默认 `http://127.0.0.1:8000/v1`。

交互场景支持 `--context-window`。默认读取训练快照中的长度；如果服务的 `--max-model-len` 与训练预算不同，应显式传入与服务一致的预算。
MMSI 支持 `--dataset MMSI_Bench_Circular`、`--max-tokens`；其余参数透传给外部 benchmark。

launcher 的 `--dry-run` 只生成命令。需要透传原生 dry-run 时，使用 `-- --dry-run`，且外部 benchmark 必须支持该参数。

默认身份规则仍为 `<训练目录名>-<checkpoint名>`。不同上级实验采用同名训练目录和 checkpoint 时可能冲突（审查报告 R02，不属于本次 R10–R15 重构）。当前可通过分别指定 `--output-root` 隔离结果，不能假定任意两份 checkpoint 都自动隔离。

## 实现边界

- `cli.py`：参数解析和动作分发。
- `serving.py`：适配器导出/校验、vLLM 参数构造及服务启动。
- `inference.py`：评估参数构造与外部客户端启动。
- `checkpoints.py`：统一 checkpoint 布局、训练快照和适配器配置读取。
- `models/adapters.py`：训练和服务共用的 PEFT 参数转换。

训练后端负责分布式保存/恢复，服务导出保持 CPU 路径；两者共用元数据协议，不混合执行过程。
