#!/usr/bin/env bash
set -euo pipefail

swift rlhf \
    --rlhf_type grpo \
    --model /work/u1501463/model_qwen_25 \
    --adapters ../stage1_training/v5-20260719-005455/checkpoint-1800 \
    --template qwen2_5_omni \
    --target_modules all-linear \
    --lora_rank 8 \
    --lora_alpha 32 \
    --dataset /work/u1501463/stage1_RL/train_rl_singleturn.jsonl \
    --external_plugins reward.py \
    --reward_funcs tool_format tool_accuracy \
    --reward_weights 0.2 0.8 \
    --num_generations 4 \
    --generation_batch_size 4 \
    --max_length 4096