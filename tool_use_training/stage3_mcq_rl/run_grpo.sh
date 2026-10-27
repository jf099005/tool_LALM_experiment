#!/usr/bin/env bash
# Stage-3: GSPO/GRPO on audio MCQ with real multi-turn audio-tool use (Qwen3-Omni thinker, LoRA).
# Adapted from verl-omni examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_avqa_v1.sh;
# differences: custom agent loop (agent_loop.yaml), stage-3 reward, longer response budget for
# tool turns that carry audio, stage-1-merged init checkpoint.
#
# Do not call directly: train.sbatch / eval.sbatch run it through launch_with_tool_pool.py
# (the agent loop refuses to start without the heavy-tool servers). Extra args = hydra overrides.
set -x

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
export VERL_USE_EXTERNAL_MODULES=verl_omni
# Slurm exports ROCR_VISIBLE_DEVICES next to CUDA_VISIBLE_DEVICES on GPU nodes; verl's Worker
# refuses to start when both are set (verl/single_controller/base/worker.py).
unset ROCR_VISIBLE_DEVICES
# vllm must be the +cu129 wheel (GPU nodes run driver 550, CUDA <= 12.4 + minor-version compat);
# the default PyPI vllm 0.27.0 wheel is a CUDA-13 build and fails on these nodes.
# Ray workers import audio_tool_agent_loop (hydra _target_) and the project's tools/interface.
export PYTHONPATH="${HERE}:${REPO_ROOT}:${PYTHONPATH}"

MODEL_PATH=${MODEL_PATH:-/work/u1501463/stage3_mcq_rl/Qwen3-Omni-30B-A3B-stage1-merged}
DATA_DIR=${DATA_DIR:-/work/u1501463/stage3_mcq_rl/data}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.parquet}
VAL_FILES=${VAL_FILES:-"['${DATA_DIR}/val.parquet']"}
OUT_ROOT=${OUT_ROOT:-/work/u1501463/stage3_mcq_rl}
EXP_NAME=${EXP_NAME:-qwen3omni_stage3_mcq_tools_v1}
NGPUS=${NGPUS:-4}
ROLLOUT_TP=${ROLLOUT_TP:-2}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.5}      # leave room for the heavy-tool models on HEAVY_TOOL_GPU
LOGGER=${LOGGER:-'["console","tensorboard"]'}
# Actor/ref attention. flash-attn has no build for torch 2.13 (GitHub wheels stop at torch 2.8,
# the HF kernels hub at 2.12), so default to sdpa. verl's use_remove_padding packs sequences and
# relies on flash-attn varlen (_flash_attention_forward monkey patch); with sdpa packed samples
# would attend to each other, so remove-padding is tied to flash_attention_2.
ATTN_IMPL=${ATTN_IMPL:-sdpa}
if [ "${ATTN_IMPL}" = "flash_attention_2" ]; then REMOVE_PADDING=True; else REMOVE_PADDING=False; fi
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${OUT_ROOT}/tensorboard/${EXP_NAME}}

python3 -m verl_omni.trainer.main_omni \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.omni_rl_datasets \
    data.custom_cls.name=QwenOmniRLHFDataset \
    +data.mm_processor_kwargs.sampling_rate=16000 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=${REMOVE_PADDING} \
    +actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPL} \
    actor_rollout_ref.model.exclude_modules=".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*" \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj']" \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.prompt_length=4160 \
    actor_rollout_ref.rollout.response_length=4096 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${HERE}/agent_loop.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_loop=audio_tool_mcq_agent \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode="ar" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="qwen3_omni_moe" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=256 \
    actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,32,64,128,256] \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path="${HERE}/reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.val_before_train=true \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger="${LOGGER}" \
    trainer.project_name=stage3_mcq_tool_rl \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${OUT_ROOT}/ckpt/${EXP_NAME}" \
    trainer.validation_data_dir="${OUT_ROOT}/val_generations/${EXP_NAME}" \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.max_actor_ckpt_to_keep=4 \
    trainer.test_freq=25 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${TOTAL_STEPS:-300} \
    "$@"
