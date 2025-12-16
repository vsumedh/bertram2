#!/bin/bash
# Start vLLM server with Qwen2.5-7B-Instruct
# Requirements: ~14GB GPU VRAM (fits in 20GB with KV cache)

set -e

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"

echo "Starting vLLM server..."
echo "  Model: $MODEL"
echo "  Host: $HOST:$PORT"
echo "  Max model length: $MAX_MODEL_LEN"
echo "  GPU utilization: $GPU_UTIL"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --dtype float16



