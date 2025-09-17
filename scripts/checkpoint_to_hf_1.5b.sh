STEPS=(
    "240"
)

for step in "${STEPS[@]}"; do
    python scripts/legacy_model_merger.py merge \
        --backend fsdp \
        --hf_model_path /root/autodl-fs/models/Qwen/Qwen2.5-1.5B \
        --local_dir /root/autodl-tmp/code/verl/checkpoints/off_policy/dare/global_step_${step}/actor \
        --target_dir /root/autodl-fs/models/qwen2.5_1.5b_dare_simplerl/step_${step}
done
