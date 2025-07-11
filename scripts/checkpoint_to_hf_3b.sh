python scripts/model_merger.py merge \
    --backend fsdp \
    --hf_model_path /root/autodl-fs/models/Qwen/Qwen2.5-3B \
    --local_dir /root/autodl-tmp/code/verl/checkpoints/qwen2.5_3b_data_dapo_entropy/baseline_grpo/global_step_160/actor \
    --target_dir /root/autodl-fs/models/qwen2.5_3b_baseline_grpo/step_160
