#!/bin/bash
#SBATCH --job-name=s3_merge_stage1
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --account=MST113234
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1                       # merge itself is light; 1 GPU just to load the 30B model fast
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH -o ./logs/merge_stage1_%j.log
#SBATCH -e ./logs/merge_stage1_%j.err

# Step 1 of stage-3: merge the stage-1 tool-use LoRA into Qwen3-Omni so verl-omni can start RL
# from it (verl trains a fresh LoRA on top; it can't resume an ms-swift adapter).
# Submit from this directory:  cd tool_use_training/stage3_mcq_rl && sbatch merge_stage1.sbatch

ADAPTER=${ADAPTER:-/work/u1501463/stage1_model_full_toolset/v1-20261003-113849/checkpoint-100}
BASE=${BASE:-/work/u1501463/Qwen3-Omni-30B-A3B-Instruct}
OUT=${OUT:-/work/u1501463/stage3_mcq_rl/Qwen3-Omni-30B-A3B-stage1-merged}

ml load miniconda3/24.11.1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate swift-qwen3
mkdir -p logs "$(dirname "$OUT")"
echo "Start: $(date)  Node: $(hostname)"

swift export \
    --model "$BASE" \
    --adapters "$ADAPTER" \
    --merge_lora true \
    --output_dir "$OUT" \
    --safe_serialization true

# verl-omni / vLLM-Omni read the processor + chat template from the model dir; make sure the
# merged dir carries the base model's copies (cp -n never overwrites what swift wrote).
for f in chat_template.json preprocessor_config.json tokenizer_config.json vocab.json merges.txt generation_config.json; do
    cp -n "$BASE/$f" "$OUT/$f" 2>/dev/null || true
done
ls -la "$OUT"
echo "End: $(date)"
