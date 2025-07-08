python scripts/model_merger.py merge \
    --backend fsdp \
    --hf_model_path /root/autodl-fs/models/Qwen/Qwen2___5-Math-1___5B \
    --local_dir /root/autodl-tmp/code/verl/checkpoints/golden_trajectory/baseline_kl_cov/global_step_240/actor \
    --target_dir /root/autodl-fs/models/qwen2.5_math_1.5b_golden_trajectory_baseline_kl_cov/step_240
