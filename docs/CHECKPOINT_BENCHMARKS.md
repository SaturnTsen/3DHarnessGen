# 切换 checkpoint 并运行两个 benchmark

三个入口都支持 `--checkpoint-path`；默认模型名统一按 `<训练目录名>-<checkpoint目录名>` 生成。换 checkpoint 不需要新增 agent、编辑 JSON 或抄服务打印的名称。服务端选择的基础模型必须与 checkpoint 匹配。

本机目录为 `~/3DHarnessBench` 和 `~/spatial_benchs/MMSI-Bench`。通过各自已有的 Pixi 环境执行 benchmark，vLLM 始终使用 `3DHarnessGen` 的现有环境。脚本可从任意工作目录启动。

## 1. 启动 vLLM

以 View-mix 27B checkpoint-40 为例。在服务终端先设置：

```bash
CKPT=~/3DHarnessGen/outputs/post_training/View-mix/qwen38_27b_qlora_fsdp_epoch1_v1/checkpoints/checkpoint-40
```

然后：

```bash
CUDA_VISIBLE_DEVICES=0,1 pixi run --manifest-path ~/3DHarnessGen/pixi.toml \
  python ~/3DHarnessGen/scripts/serve_checkpoint.py \
  --checkpoint-path "$CKPT" \
  --base-model-path ~/models/Qwen3.8-27B \
  --tensor-parallel-size 2 --data-parallel-size 1 \
  --max-model-len 49152 --max-num-seqs 4 \
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":10,"video":0}' \
  --host 127.0.0.1 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

其余参数遵循正常的 vLLM 用法。`--model-name` 是 `--adapter-name` 的同义参数；不填时本例自动得到 `qwen38_27b_qlora_fsdp_epoch1_v1-checkpoint-40`。三份脚本的默认规则一致。

`--dry-run` 只读取 checkpoint、准备必要的小文件并显示命令，不加载基础模型到 GPU。真实 checkpoint-40 已通过 FSDP 适配器提取与基础权重维度检查；这里的 27B 服务、49k 长度和四并发尚未在目标 GPU 上实测。

## 2. 运行 3DHarnessBench inference

另开终端，设置同一个 `CKPT` 和与服务端相同的 `VLLM_API_KEY`，然后：

```bash
bash ~/3DHarnessBench/scripts/infer_checkpoint.sh \
  --checkpoint-path "$CKPT" \
  --setting single-view \
  --data-path ~/3DHarnessGen/3DHarnessBench/benchmark \
  --iterations 3 --num-parallel 4
```

`--setting` 可选 `single-view`、`multi-view`、`active-visual`、`full-3d-interaction`，默认 `single-view`。后两种没有 `--iterations` 参数。它们从训练目录的 `resolved_config.yaml` 读取总长度预算；如果服务的 `--max-model-len` 与训练预算不同，请显式传 `--context-window`：

```bash
bash ~/3DHarnessBench/scripts/infer_checkpoint.sh \
  --checkpoint-path "$CKPT" --setting active-visual \
  --context-window 49152 \
  --data-path ~/3DHarnessGen/3DHarnessBench/benchmark --num-parallel 4
```

图像场景允许通过原生 `--gen-max-tokens` 指定输出预算，默认 8192。`--base-url` 可覆盖本地服务地址；`--model-name` 可指定服务的自定义适配器名。通用 agent 为 `qwen-checkpoint-local`，关闭 thinking，不发送 reasoning effort。

当前机器的 `~/3DHarnessBench/data/benchmark` 是失效软链接，示例用了已存在的 benchmark 目录。其他机器可替换 `--data-path`，不要指向训练 `runs/`。

## 3. 运行 MMSI-Bench evaluation

另开终端，同样设置 `CKPT` 和 API key：

```bash
bash ~/spatial_benchs/MMSI-Bench/scripts/eval_checkpoint.sh \
  --checkpoint-path "$CKPT" --api-nproc 4
```

自动创建对应的 API 模型配置，运行原有 1000 题推理和本地选项匹配评分。只推理加 `--mode infer`；Circular 评测加 `--dataset MMSI_Bench_Circular`。`--max-tokens` 默认 8192，可显式调整。

MMSI 有 9～10 图样本，服务启动时须允许 10 张图。MMSI 环境不安装或加载 vLLM，复用现有流式多图 HTTP 客户端。导出的 `VLLM_API_KEY` 优先于 benchmark 的 `.env`。

## 参数与结果隔离

两个 benchmark 脚本支持：

- `--checkpoint-path`：必填，用于统一默认服务名及结果身份。
- `--model-name`（或 `--adapter-name`）：可选，必须与启动服务时指定的名字相同。
- `--base-url`：默认 `http://127.0.0.1:8000/v1`。
- `--benchmark-root`：覆盖 benchmark 仓库位置；在各仓库的 shell 入口中默认就是该入口所在仓库。
- `--output-root`：覆盖结果根目录；结果仍按 checkpoint 身份隔离。
- `--dry-run`：只显示执行目录与命令。MMSI 会准备配置 JSON，不发模型请求。

其余参数原样传给 benchmark。结果路径不以自定义服务别名为唯一身份：即使把所有 checkpoint 都服务为 `sft`，换 checkpoint 后也不会复用上一份结果。

```text
3DHarnessBench/outputs/checkpoints/<训练目录名>-checkpoint-N/<setting>/...
MMSI-Bench/outputs/checkpoints/<训练目录名>-checkpoint-N/T<日期>_G<commit>/...
```

MMSI 的运行 JSON 保存在 `3DHarnessGen/temp/benchmark_configs/`，与源配置和 checkpoint 分离。两个脚本都不加载 checkpoint 权重，只生成客户端参数；FSDP 适配器提取由服务脚本负责。

若需要 benchmark 自身的 dry-run，可用 `-- --dry-run` 把该参数传给 3DHarnessBench。例如在正常 inference 命令末尾加 `--limit 1 -- --dry-run`。MMSI 原生入口没有 dry-run 或 limit；不要把 launcher 的 dry-run 当作完整评测通过。

## 换回 Blender 9B

在各终端改成：

```bash
CKPT=~/3DHarnessGen/outputs/post_training/Blender/qwen35_9b_qlora_fsdp_mix_epoch1_v1/checkpoints/checkpoint-1612
```

服务的 `--base-model-path` 改成 `~/models/Qwen3.5-9B`，并按硬件调整 GPU、并行和总长度；两个客户端命令继续使用同样接口，无需修改代码或新增固定别名。更换 checkpoint 需要停止旧服务并用新 checkpoint 重启。

两个 shell 入口通过 `HARNESS_GEN_ROOT` 指定共享代码仓库，默认 `~/3DHarnessGen`；如果服务代码放在其他目录，可在调用前导出该变量。通用 Python 入口也可直接使用：

```bash
pixi run --manifest-path ~/3DHarnessGen/pixi.toml python ~/3DHarnessGen/scripts/run_3dharness_checkpoint.py --help
pixi run --manifest-path ~/3DHarnessGen/pixi.toml python ~/3DHarnessGen/scripts/eval_mmsi_checkpoint.py --help
```
