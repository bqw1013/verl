#!/bin/bash
set -e
GPU_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPU_NUM: $GPU_NUM"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_NUM - 1)))
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tp)
        TP="$2"
        shift 2
        ;;
        --dp)
        DP="$2"
        shift 2
        ;;
        --model-path)
        MODEL_PATH="$2"
        shift 2
        ;;
        *)
        echo "未知参数: $1"
        exit 1
        ;;
    esac
done

TP=${TP:-1}
DP=${DP:-$((GPU_NUM / TP))}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B}

set -x
vllm serve "$MODEL_PATH" \
    --port 8000 \
    --host 0.0.0.0 \
    --tensor-parallel-size $TP \
    --data-parallel-size $DP \
    --seed 42 \
    --trust-remote-code \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.75 \
    --enable-prefix-caching \
    --max-model-len 8192 \
    --max-num-seqs 256 \
    --served-model-name "Qwen"
set +x