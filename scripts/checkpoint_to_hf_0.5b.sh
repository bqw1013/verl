STEPS=(
    "1605"
    "1284"
    "963"
    "642"
    "321"
)

for step in "${STEPS[@]}"; do
    python scripts/legacy_model_merger.py merge \
        --backend fsdp \
        --hf_model_path /root/autodl-fs/models/Qwen/Qwen2.5-0.5B \
        --local_dir /root/autodl-tmp/code/verl/checkpoints/miromind_sft/global_step_${step} \
        --target_dir /root/autodl-fs/models/qwen2.5_0.5b_miromind_sft/step_${step}
done
