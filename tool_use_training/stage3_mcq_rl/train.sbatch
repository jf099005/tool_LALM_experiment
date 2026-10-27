#!/bin/bash
#SBATCH --job-name=s3_mcq_tool_grpo
#SBATCH --partition=normal
#SBATCH --time=48:00:00                         # 300 steps x (rollout w/ real tool calls + update); check s/step in the log and adjust
#SBATCH --account=MST113234
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4                       # 4x H100 80G: FSDP actor + vLLM-Omni rollout (TP=2 x 2 replicas), colocated
#SBATCH --cpus-per-task=32                      # light tools (librosa/scipy) run on CPU threads inside the agent-loop workers
#SBATCH --ntasks-per-node=1
#SBATCH -o ./logs/train_%j.log
#SBATCH -e ./logs/train_%j.err

# Step 3 of stage-3: GSPO RL on audio MCQ with multi-turn audio-tool use.
# Submit from this directory:  cd tool_use_training/stage3_mcq_rl && sbatch train.sbatch [hydra overrides]
# Prereqs: merge_stage1.sbatch and build_data.sbatch finished; HF access to facebook/sam-audio-large.

ml load miniconda3/24.11.1
# sam_audio's torchcodec needs the system CUDA 12.4 NPP libs -- but only the tool servers get
# them (HEAVY_TOOL_LD_LIBRARY_PATH); verl-omni (torch cu129 / flashinfer JIT) keeps a clean path.
module load cuda/12.4
export HEAVY_TOOL_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
module unload cuda/12.4

source $(conda info --base)/etc/profile.d/conda.sh
conda activate verl-omni

mkdir -p logs
echo "Start: $(date)  Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Heavy tools (SAM-Audio-large / AudioSR / DeepFilterNet) live on the last GPU next to one
# rollout shard; GPU_MEM_UTIL=0.5 in run_grpo.sh leaves them ~25 GB there.
export HEAVY_TOOL_GPU=${HEAVY_TOOL_GPU:-3}
export STAGE3_TOOL_WORKDIR=${STAGE3_TOOL_WORKDIR:-/work/u1501463/stage3_mcq_rl/tool_work/${SLURM_JOB_ID}}

(
while true; do
    echo "===== $(date) ====="
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    sleep 300  # NCHC: polling at <30 s for >10 min counts as a system attack
done
) > logs/gpu_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!

python launch_with_tool_pool.py -- bash run_grpo.sh "$@"
STATUS=$?

kill ${GPU_MONITOR_PID} 2>/dev/null
rm -rf "$STAGE3_TOOL_WORKDIR"
echo "End: $(date)  exit=${STATUS}"
exit ${STATUS}
