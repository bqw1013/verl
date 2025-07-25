python scripts/model_merger.py merge \
    --backend fsdp \
    --hf_model_path /root/autodl-fs/models/Qwen/Qwen2.5-3B \
    --local_dir /root/autodl-tmp/code/verl/checkpoints/qwen2.5_3b_data_dapo_passk/pkpo_4_pos_rho5_neg_mean/global_step_40/actor \
    --target_dir /root/autodl-fs/models/qwen2.5_3b_pkpo_4_pos_rho5_neg_mean/step_40
