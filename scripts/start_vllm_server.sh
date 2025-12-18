#!/bin/bash
# Start vLLM server with Qwen2.5-14B-Instruct-GPTQ-Int4 (pre-quantized)
# Requirements: ~9GB GPU VRAM (fits comfortably in 23GB L4)
# Hardware: Any NVIDIA GPU with sufficient VRAM

set -e

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
QUANTIZATION="${VLLM_QUANTIZATION:-gptq_marlin}"
KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-auto}"

echo "Starting vLLM server..."
echo "  Model: $MODEL"
echo "  Host: $HOST:$PORT"
echo "  Max model length: $MAX_MODEL_LEN"
echo "  GPU utilization: $GPU_UTIL"
echo "  Quantization: $QUANTIZATION"
echo "  KV Cache dtype: $KV_CACHE_DTYPE"

# Build command with optional quantization
CMD="python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --host $HOST \
    --port $PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_UTIL \
    --dtype float16 \
    --enable-prefix-caching \
    --disable-log-requests"

# Add quantization if specified (fp8, gptq, awq, etc.)
if [ "$QUANTIZATION" != "none" ] && [ -n "$QUANTIZATION" ]; then
    CMD="$CMD --quantization $QUANTIZATION"
fi

# Add KV cache dtype if specified
if [ "$KV_CACHE_DTYPE" != "auto" ] && [ -n "$KV_CACHE_DTYPE" ]; then
    CMD="$CMD --kv-cache-dtype $KV_CACHE_DTYPE"
fi

echo "Running: $CMD"
eval $CMD
