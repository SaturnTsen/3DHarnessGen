#!/usr/bin/env bash
# Shared local service for 3DHarnessBench and MMSI-Bench.
set -euo pipefail

gen_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec pixi run --manifest-path "$gen_root/pixi.toml" python "$gen_root/scripts/serve_checkpoint.py" \
  --checkpoint-path "$gen_root/outputs/post_training/Blender/qwen35_9b_qlora_fsdp_mix_epoch1_v1/checkpoints/checkpoint-1612" \
  --base-model-path "$HOME/models/Qwen3.5-9B" \
  --adapter-name blender-9b-step1612 \
  --tensor-parallel-size 1 \
  --data-parallel-size 2 \
  --max-model-len 73728 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":10,"video":0}' \
  --host 127.0.0.1 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  "$@"
