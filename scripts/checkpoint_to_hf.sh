python scripts/model_merger.py merge \
    --backend fsdp \
    --hf_model_path /root/autodl-fs/models/Qwen/Qwen2___5-Math-1___5B-Instruct\
    --local_dir /root/autodl-fs/models/verl/checkpoints/qwen2_1.5b_dapo_simplerl_lr1e-5/global_step_100/actor \
    --target_dir /root/autodl-fs/models/verl/dapo_lr1e-5/step_100
