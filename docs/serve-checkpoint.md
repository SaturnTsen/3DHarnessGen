# checkpoint 服务

使用 `pixi run serve`，执行 `harness_training/cli/serve.py`。本项目负责 checkpoint 适配器准备、校验和 vLLM 服务启动；评估由各 benchmark 工程自行运行。

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

## 客户端连接

服务启动命令会输出 `Client request model: <名称>`。客户端使用该名称和配置的服务地址调用 OpenAI 兼容 API；例如上述服务对应 `http://127.0.0.1:8000/v1`。

默认模型名称为 `<训练目录名>-<checkpoint名>`，也可以通过 `--model-name` / `--adapter-name` 显式指定。不同实验使用同名训练目录和 checkpoint 时默认名称可能相同；需要区分时应显式指定名称。

本项目不提供评估命令，不读取 MMSI 或 3DHarnessBench 的评估配置，也不启动这些工程。评估数据、参数和输出目录由相应 benchmark 工程管理。

## 实现边界

- `cli/serve.py`：服务参数解析和进程启动。
- `serving.py`：适配器导出/校验、vLLM 参数构造及服务启动。
- 不设置公共 checkpoint 模块；服务在使用处直接读取适配器配置或训练快照，并复用模型侧 LoRA 参数转换。
- `config.py` 的 `LoRAConfig.to_peft()`：按现有配置构造 PEFT 的 LoRA 配置对象。

训练后端负责分布式保存，服务导出保持 CPU 路径；两者共用 checkpoint 布局和适配器元数据。
