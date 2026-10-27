#!/bin/bash
#SBATCH --job-name=plot_loss                    # 工作名稱
#SBATCH --partition=normal                      # 使用的 partition（依系統修改，若有專屬 CPU partition 請改用該分區，例如 cpu / ctest）
#SBATCH --time=00:15:00                         # 純畫圖，通常幾秒到幾分鐘內完成
#SBATCH --account=MST113234      ####### 請記得換成您的計畫代碼 #######
#SBATCH --nodes=1                               # 只需要 1 個節點
#SBATCH --ntasks-per-node=1                     # 只需要 1 個 task
#SBATCH --cpus-per-task=1                       # 純畫圖不需要多核心
#SBATCH --gpus-per-node=1
#SBATCH -o ./logs/plot_loss_%j.log              # output file (%j expands to jobId)
#SBATCH -e ./logs/plot_loss_%j.err              # output file (%j expands to jobId)
# 注意：本腳本刻意不加 --gpus-per-node，僅要求 CPU 資源

ml load miniconda3/24.11.1
CONDA_DEFAULT_ENV="swift-qwen3"   # 你的 conda 環境名稱（需已安裝 matplotlib）

mkdir -p logs

echo "Start: $(date)"
echo "Node: $(hostname)"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$CONDA_DEFAULT_ENV"

# ---- 輸入/參數由呼叫端在 sbatch 時提供 ----
# $1: 必填，logging.jsonl 的路徑
# $2: 選填，輸出圖檔路徑（不給則預設存到輸入檔同目錄下的 loss_curve.png）
INPUT_JSONL="$1"
OUTPUT_PNG="$2"

if [ -z "${INPUT_JSONL}" ]; then
    echo "錯誤: 未指定輸入的 logging.jsonl 路徑" >&2
    echo "用法: sbatch submit_plot_loss.slurm <logging.jsonl路徑> [輸出圖檔路徑]" >&2
    exit 1
fi

if [ -n "${OUTPUT_PNG}" ]; then
    python -u plot_loss.py "${INPUT_JSONL}" --output "${OUTPUT_PNG}"
else
    python -u plot_loss.py "${INPUT_JSONL}"
fi

echo "End: $(date)"

