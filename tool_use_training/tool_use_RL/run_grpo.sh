#!/usr/bin/env bash
set -euo pipefail

# --dataset below must be regenerated (build_rl_dataset.py) whenever
# gen_1st_stage_data's raw JSON changes -- e.g. after moving the tool
# catalogue into the system prompt:
#   python build_rl_dataset.py \
#       --input /work/u1501463/stage1_RL/tool_usage_qa_audioset.json \
#       --output /work/u1501463/stage1_RL/train_rl_singleturn.jsonl \
#       --system-prompt-model-dir /work/u1501463/model_qwen_25 \
#       --system-prompt-model-type qwen2_5_omni

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