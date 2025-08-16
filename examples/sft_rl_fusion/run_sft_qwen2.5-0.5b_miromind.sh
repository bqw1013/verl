set -x

HOME=/root/autodl-tmp/code/verl

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/datasets/math/miromind-m1-sft-rl-fusion/train.parquet  \
    data.val_files=$HOME/datasets/math/miromind-m1-sft-rl-fusion/test.parquet \
    data.prompt_key=question \
    data.response_key=response \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=16384 \
    data.truncation=right \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=True \
    model.partial_pretrain=/root/autodl-fs/models/Qwen/Qwen2.5-0.5B \
    model.fsdp_config.offload_params=True \
    model.fsdp_config.model_dtype=bf16 \
    model.use_liger=True \
    trainer.default_local_dir=./checkpoints/miromind_sft \
    trainer.project_name=sft \
    trainer.experiment_name=miromind-sft-qwen2.5-0.5b \
    trainer.total_epochs=5 \
    trainer.save_freq=321 \
    trainer.logger='["console","swanlab"]' $@
