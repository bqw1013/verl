STEPS=(
    "320"
    "280"
    "240"
    "200"
    "160"
    # "120"
    # "80"
    # "40"
)

for step in "${STEPS[@]}"; do
    python scripts/legacy_model_merger.py merge \
        --backend fsdp \
        --hf_model_path /root/autodl-fs/models/Qwen/Qwen2.5-3B \
        --local_dir /root/autodl-tmp/code/verl/checkpoints/off_policy/dare_simplerl_dynamic_neg_backup/global_step_${step}/actor \
        --target_dir /root/autodl-fs/models/qwen2.5_3b_simplerl_dare_neg_backup_inter_200/step_${step}
done
