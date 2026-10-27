i#!/bin/bash
#SBATCH --job-name=qwen3_omni_grpo_dualgpu
#SBATCH --partition=normal
#SBATCH --time=04:00:00                        # 正式訓練拉長時間上限,先給 4hr 觀察,穩定後再依實際需求調整
#SBATCH --account=MST113234
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2                      # 雙卡
#SBATCH --cpus-per-task=16                     # 跟著 GPU 數等比例從 8 拉到 16
#SBATCH --ntasks-per-node=1
#SBATCH -o ./logs/training_%j.log
#SBATCH -e ./logs/training_%j.err
#SBATCH --exclude=hgpn17

ml load miniconda3/24.11.1
module load cuda/12.4
CONDA_DEFAULT_ENV="verl-omni"                  # 對齊你目前實際使用的 conda 環境名稱

mkdir -p logs
echo "Start: $(date)"
echo "Node: $(hostname)"

source $(conda info --base)/etc/profile.d/conda.sh

(
while true; do
    echo "===== $(date) ====="
    nvidia-smi \
        --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total \
        --format=csv,noheader,nounits
    sleep 300  # NCHC: polling at <30 s for >10 min counts as a system attack
done
) > logs/gpu_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!

conda activate ${CONDA_DEFAULT_ENV}

bash run_grpo.sh

kill ${GPU_MONITOR_PID} 2>/dev/null

echo "after conda"
