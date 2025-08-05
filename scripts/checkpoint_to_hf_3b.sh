python scripts/model_merger.py merge \
    --backend fsdp \
    --hf_model_path /root/autodl-fs/models/Qwen/Qwen2.5-3B \
    --local_dir /root/autodl-tmp/code/verl/checkpoints/qwen2.5_3b_data_simplerl_passk/pkpo_4_neg_rho_lr2e-6/global_step_400/actor \
    --target_dir /root/autodl-fs/models/qwen2.5_3b_simplerl_pkpo_4_neg_rho_lr2e-6/step_400
