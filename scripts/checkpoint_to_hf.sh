python scripts/model_merger.py merge \
    --backend fsdp \
    --hf_model_path /root/autodl-fs/models/Qwen/Qwen2___5-Math-1___5B-Instruct \
    --local_dir /root/autodl-fs/models/verl_checkpoints/qwen2.5_math_1.5b_re++/global_step_160/actor \
    --target_dir /root/autodl-fs/models/verl/qwen2.5_math_1.5b_re++/step_160
