#!/usr/bin/env bash
set -euo pipefail

swift rlhf \
    --rlhf_type grpo \
    --model /home/u1501463/tool_use_LALM/output/v13-20260626-142749/checkpoint-39-merged \
    --template qwen2_5_omni \
    --target_modules all-linear \
    --lora_rank 8 \
    --lora_alpha 32 \
    --dataset /home/u1501463/tool_use_LALM/tool_use_training/tool_use_RL/grpo_data.jsonl \
    --external_plugins tool_use_training/tool_use_RL/reward.py \
    --reward_funcs tool_format tool_accuracy \
    --reward_weights 0.2 0.8 \
    --num_generations 4 \
    --generation_batch_size 4 \
    --max_length 4096 \
    --output_dir output/grpo-tool-use-v13
