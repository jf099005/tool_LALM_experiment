#!/usr/bin/env bash
set -euo pipefail

# --dataset below must be regenerated (build_rl_dataset.py) whenever
# gen_1st_stage_data's raw JSON changes -- e.g. after moving the tool
# catalogue into the system prompt:
#   python build_rl_dataset.py \
#       --input /work/u1501463/stage1_qwen3_omni/tool_usage_qa_audioset.json \
#       --output /work/u1501463/stage1_qwen3_omni/train_rl_singleturn.jsonl \
#       --system-prompt-model-dir /work/u1501463/model_qwen3_omni \
#       --system-prompt-model-type qwen3_omni

# Qwen3-Omni-Instruct weights -- point this at the cluster's real materialized
# path (or an HF hub id), same convention as ../submit_python.sh.
model_dir=/work/u1501463/model_qwen3_omni/

# SFT LoRA checkpoint to initialize the RL policy from (produced by
# ../train.py against the same model_dir above). "none" (default) = start
# GRPO straight from the base model, no --adapters passed at all.
adapters=none
# adapters=/work/u1501463/stage1_training/<qwen3-omni-run>/checkpoint-XXX/

# `swift` CLI auto-launches via torchrun when this is set (see
# swift.cli.main) -- match ./submit_python.sh's --gpus-per-node.
export NPROC_PER_NODE=${NPROC_PER_NODE:-2}

adapter_args=()
if [[ "$adapters" != "none" ]]; then
    adapter_args=(--adapters "$adapters")
fi

swift rlhf \
    --rlhf_type grpo \
    --model "${model_dir}" \
    --template qwen3_omni \
    "${adapter_args[@]}" \
    --deepspeed zero3 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --dataset /work/u1501463/stage1_qwen3_omni/train_rl_singleturn.jsonl \
    --external_plugins reward.py \
    --reward_funcs tool_format tool_accuracy tool_audio_closeness \
    --reward_weights 0.2 0.5 0.3 \
    --num_generations 4 \
    --generation_batch_size 4 \
    --max_length 4096 \
    --save_steps 100
