set -x
###
 # @Author: Qiangwei Bai
 # @Date: 2025-04-17 13:00:21
 # @LastEditTime: 2025-04-26 02:50:30
 # @LastEditors: Qiangwei Bai
 # @FilePath: /verlx/scripts/train_simplerl_qwen2-1.5b_math.sh
 # @Description: 
### 

export VLLM_ATTENTION_BACKEND=XFORMERS
HOME=/root/autodl-tmp/code/verl/datasets/math/SimpleRL/simplerl_qwen_level3to5

python3 -m recipe.dapo.main_dapo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/train.parquet \
    data.val_files=$HOME/test.parquet \
    data.truncation='left' \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/root/autodl-fs/models/Qwen/Qwen2___5-Math-1___5B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    algorithm.filter_groups.enable=False \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_mode=kl_cov \
    actor_rollout_ref.actor.k_percent=0.2 \
    actor_rollout_ref.actor.ppo_kl_coef=1.0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=True \
    reward_model.overlong_buffer.len=1024 \
    reward_model.overlong_buffer.penalty_factor=1.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_algorithem_rebase_8_4090D_autodl' \
    trainer.experiment_name='qwen2.5_math_1.5b_dapo_kl_cov' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    trainer.val_before_train=True
