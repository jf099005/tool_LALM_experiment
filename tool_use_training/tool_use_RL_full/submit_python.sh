#!/bin/bash
#SBATCH --job-name=qwen3_omni_grpo_smoketest    # 工作名稱
#SBATCH --partition=normal                         # 使用的 partition (請根據你的系統修改)
#SBATCH --time=01:00:00                         # 小規模測試上限 -- GRPO 還要跑 rollout 生成，比 SFT 更吃記憶體，比起 SFT 更可能在 2 張卡上 OOM，這正是這次測試要驗證的地方
#SBATCH --account=MST113234     ####### 請記得換成您的計畫代碼 #######
#SBATCH --nodes=1                               # (-N) Maximum number of nodes to be allocated
#SBATCH --gpus-per-node=1                       # 2x H100 80G -- 需與 run_grpo.sh 的 NPROC_PER_NODE 預設值一致
#SBATCH --cpus-per-task=8                       # (-c) Number of cores per MPI task -- 依 GPU 數等比例縮放
#SBATCH --ntasks-per-node=1                     # Maximum number of tasks on each node
#SBATCH -o ./logs/training_%j.log               # output file (%j expands to jobId)
#SBATCH -e ./logs/training_%j.err               # output file (%j expands to jobId)
#SBATCH --exclude=hgpn17

ml load miniconda3/24.11.1
module load cuda/12.4
CONDA_DEFAULT_ENV="RL" # 你的 conda 環境名稱


mkdir -p logs
echo "Start: $(date)"
echo "Node: $(hostname)"


echo "before conda"

# conda run -n "$CONDA_DEFAULT_ENV" python -u train.py
source $(conda info --base)/etc/profile.d/conda.sh

# 每5分鐘記錄一次GPU資訊（NCHC 規定：間隔 <30 秒且持續 >10 分鐘的輪詢視為系統攻擊）
(
while true; do
    echo "===== $(date) ====="
    nvidia-smi \
        --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total \
        --format=csv,noheader,nounits
    sleep 300
done
) > logs/gpu_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!


conda activate ${CONDA_DEFAULT_ENV}

#bash run_grpo.sh
bash grpo_test.sh

echo "after conda"
