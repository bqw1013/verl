#!/bin/bash
set -e

MODELS=(
    "/root/autodl-fs/models/qwen2.5_0.5b_countdown_sft_e1_grpo/step_500"
    "/root/autodl-fs/models/qwen2.5_0.5b_countdown_sft_e1_grpo/step_400"
    "/root/autodl-fs/models/qwen2.5_0.5b_countdown_sft_e1_grpo/step_300"
    "/root/autodl-fs/models/qwen2.5_0.5b_countdown_sft_e1_grpo/step_200"
    "/root/autodl-fs/models/qwen2.5_0.5b_countdown_sft_e1_grpo/step_100"
)

for MODEL_PATH in "${MODELS[@]}"; do
    echo "Evaluating $MODEL_PATH..."

    echo "Starting server..."
    
    (bash eval/run_vllm_server.sh \
        --model-path $MODEL_PATH \
        --tp 1) & SERVER_SCRIPT_PID=$!

    echo "Server script PID: $SERVER_SCRIPT_PID"

    echo "Waiting for server to start..."
    while ! timeout 1 bash -c "</dev/tcp/localhost/8000" 2>/dev/null; do
        sleep 1
    done
    
    echo "Running evaluation..."
    set -x
    python eval/eval_countdown.py \
        --model_path $MODEL_PATH \
        --dataset_path /root/autodl-tmp/code/verl/datasets/countdown/test.parquet \
        --prompt_key question \
        --ground_truth_key reward_model \
        --template qwen \
        --n 1 \
        --temperature 0.7 \
        --max_tokens 2048 \
        --top_p 0.9 \
        --output_file sft_e1_grpo_countdown_0.5b_results.jsonl
    set +x

    echo "Stopping server..."
    kill $(lsof -ti :8000)
    wait $SERVER_SCRIPT_PID 2>/dev/null || true

    echo "Evaluation completed for $MODEL_PATH"
done
