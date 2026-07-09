#!/bin/bash
#SBATCH --job-name=training                     # 工作名稱
#SBATCH --partition=dev                         # 使用的 partition (請根據你的系統修改)
#SBATCH --time=00:05:00                         # 執行時間上限 (小時:分鐘:秒)
#SBATCH --account=MST113025     ####### 請記得換成您的計畫代碼 #######
#SBATCH --nodes=1                               # (-N) Maximum number of nodes to be allocated
#SBATCH --gpus-per-node=1                       # Gpus per node
#SBATCH --cpus-per-task=4                       # (-c) Number of cores per MPI task
#SBATCH --ntasks-per-node=1                     # Maximum number of tasks on each node
#SBATCH -o ./ckpt/training_%j.log               # output file (%j expands to jobId)
#SBATCH -e ./ckpt/training_%j.err               # output file (%j expands to jobId)

ml load miniconda3/24.11.1
CONDA_DEFAULT_ENV="swift" # 你的 conda 環境名稱


mkdir -p logs

echo "Start: $(date)"
echo "Node: $(hostname)"

# 每5秒記錄一次GPU資訊
(
while true; do
    echo "===== $(date) ====="
    nvidia-smi \
        --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total \
        --format=csv,noheader,nounits
    sleep 5
done
) > logs/gpu_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!

conda run -n "$CONDA_DEFAULT_ENV" bash train.sh



# 結束GPU監控
kill $GPU_MONITOR_PID
echo "End: $(date)"