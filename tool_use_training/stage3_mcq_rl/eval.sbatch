#!/bin/bash
#SBATCH --job-name=s3_mcq_tool_eval
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --account=MST113234
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH -o ./logs/eval_%j.log
#SBATCH -e ./logs/eval_%j.err

# Full MMAU test-mini (1000) + MMAR (1000) with the same tool-using agent loop, greedy decoding.
# Reuses run_grpo.sh with trainer.val_only=true, so prompts/tools/reward are identical to training.
#
#   sbatch eval.sbatch                                    # stage-1 merged init (baseline)
#   CKPT=/work/u1501463/stage3_mcq_rl/ckpt/<exp>/global_step_300 sbatch eval.sbatch   # an RL checkpoint
#   MODEL_PATH=/work/u1501463/Qwen3-Omni-30B-A3B-Instruct sbatch eval.sbatch          # raw base model
#
# Per-sample generations (incl. tool usage) land in /work/u1501463/stage3_mcq_rl/val_generations/<EXP_NAME>;
# headline numbers are the val-core/<data_source>/acc lines in the log.

ml load miniconda3/24.11.1
module load cuda/12.4
export HEAVY_TOOL_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
module unload cuda/12.4
source $(conda info --base)/etc/profile.d/conda.sh
conda activate verl-omni
mkdir -p logs
echo "Start: $(date)  Node: $(hostname)"

export HEAVY_TOOL_GPU=${HEAVY_TOOL_GPU:-3}
export STAGE3_TOOL_WORKDIR=${STAGE3_TOOL_WORKDIR:-/work/u1501463/stage3_mcq_rl/tool_work/${SLURM_JOB_ID}}
DATA_DIR=${DATA_DIR:-/work/u1501463/stage3_mcq_rl/data}
export VAL_FILES="['${DATA_DIR}/eval_mmau_test_mini.parquet','${DATA_DIR}/eval_mmar.parquet']"

EXTRA=()
if [ -n "${CKPT}" ]; then
    export EXP_NAME=${EXP_NAME:-eval_$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")}
    EXTRA+=(trainer.resume_mode=resume_path trainer.resume_from_path="${CKPT}")
else
    export EXP_NAME=${EXP_NAME:-eval_$(basename "${MODEL_PATH:-stage1_merged}")}
    EXTRA+=(trainer.resume_mode=disable)
fi

python launch_with_tool_pool.py -- bash run_grpo.sh \
    trainer.val_only=true trainer.val_before_train=true \
    data.max_prompt_length=6144 actor_rollout_ref.rollout.prompt_length=6208 \
    "${EXTRA[@]}" "$@"
STATUS=$?
rm -rf "$STAGE3_TOOL_WORKDIR"
echo "End: $(date)  exit=${STATUS}"
exit ${STATUS}
