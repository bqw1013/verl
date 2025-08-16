set -x

HOME=/root/autodl-tmp/code/verl

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/datasets/countdown/train_with_responses.parquet  \
    data.val_files=$HOME/datasets/countdown/test_with_responses.parquet \
    data.prompt_key=question \
    data.response_key=responses \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=4 \
    data.max_length=2048 \
    model.partial_pretrain=/root/autodl-fs/models/Qwen/Qwen2.5-0.5B \
    model.fsdp_config.offload_params=True \
    model.use_liger=True \
    trainer.default_local_dir=./checkpoints/countdown_sft \
    trainer.project_name=sft \
    trainer.experiment_name=countdown-sft-qwen2.5-0.5b \
    trainer.total_epochs=10 \
    trainer.save_freq=80 \
    trainer.logger='["console","swanlab"]' $@
