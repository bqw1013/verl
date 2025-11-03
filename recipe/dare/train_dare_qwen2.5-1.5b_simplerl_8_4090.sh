set -x

# export CUDA_VISIBLE_DEVICES=0,1

export VLLM_ATTENTION_BACKEND=XFORMERS
HOME=/root/autodl-tmp/code/verl

python3 -m recipe.dare.main_dare \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/datasets/math/SimpleRL/simplerl_qwen_level3to5/train2nonmath.parquet \
    data.val_files=$HOME/datasets/math/SimpleRL/simplerl_qwen_level3to5/test2nonmath.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/root/autodl-fs/models/Qwen/Qwen2.5-1.5B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_liger=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.service.base_url=https://u66551-baab-24341aca.cqa1.seetacloud.com:8443/v1 \
    actor_rollout_ref.rollout.service.model_name=Qwen2.5-7B-Instruct \
    actor_rollout_ref.rollout.service.api_key=none \
    actor_rollout_ref.rollout.service.temperature=1.0 \
    actor_rollout_ref.rollout.service.top_p=0.8 \
    actor_rollout_ref.rollout.service.concurrency_limit=512 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='off_policy' \
    trainer.experiment_name='dare_simplerl_dynamic_20_start' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=5 \
    trainer.total_training_steps=320 \
    trainer.relay_schedule="'[(21, 120, 1.0, 0.2)]'" \
    trainer.val_before_train=False
