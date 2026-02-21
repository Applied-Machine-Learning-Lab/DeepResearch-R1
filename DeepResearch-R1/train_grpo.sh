export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export DATA_DIR=''

WAND_PROJECT=''

# export BASE_MODEL='meta-llama/Llama-3.2-3B'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.2-3b-em
# export BASE_MODEL='meta-llama/Llama-3.2-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.2-3b-it-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.1-8b-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-llama3.1-8b-it-em

export BASE_MODEL=''
export EXPERIMENT_NAME=""
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-3b-it-em
# export BASE_MODEL='Qwen/Qwen2.5-7B'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-7b-em
# export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-7b-it-em

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=16 \
    data.train_data_num=7168 \
    data.val_data_num=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=256 \
    data.max_start_length=512 \
    data.max_obs_length=512 \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.grad_clip=0.5 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    ++actor_rollout_ref.model.torch_dtype=bfloat16 \
    ++actor_rollout_ref.rollout.dtype=bfloat16 \
    ++actor_rollout_ref.rollout.temperature=0.7 \
    ++actor_rollout_ref.rollout.top_p=0.9 \
    critic.model.path=$BASE_MODEL \
    critic.ppo_mini_batch_size=32 \
    critic.ppo_micro_batch_size=16 \
    critic.optim.lr=5e-6 \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=true \
    ++critic.model.torch_dtype=bfloat16 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    +trainer.val_before_train=false \
    trainer.logger="[console,wandb]" \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=3 \
    trainer.save_freq=28 \
    trainer.test_freq=28 \
    trainer.default_hdfs_dir=null \
    max_turns=2 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    reward_model.final_format_score=0.2 \
    reward_model.retrieval_score=0.1 \
    +reward_model.correct_score=1.0 \
    2>&1 | tee ${EXPERIMENT_NAME}.log