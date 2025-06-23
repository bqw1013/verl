set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
HOME=/root/autodl-tmp/code/verl

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=pkpo \
    data.train_files=$HOME/datasets/math/SimpleRL/simplerl_qwen_level3to5/train.parquet \
    data.val_files=$HOME/datasets/math/SimpleRL/simplerl_qwen_level3to5/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.oversampling_ratio=2.0 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/root/autodl-fs/models/Qwen/Qwen2___5-Math-1___5B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.pkpo.k=4 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='develop' \
    trainer.experiment_name='pkpo' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    trainer.val_before_train=False