python scripts/model_merger.py merge \
    --backend fsdp \
    --hf_model_path /root/autodl-fs/models/Qwen/Qwen2___5-Math-1___5B \
    --local_dir /root/autodl-tmp/code/verl/checkpoints/verl_algorithem_rebase_8_4090D_autodl/qwen2.5_math_1.5b_dapo_kl_cov/global_step_50/actor \
    --target_dir /root/autodl-fs/models/qwen2.5_math1.5b_dapo_kl_cov/step_50
