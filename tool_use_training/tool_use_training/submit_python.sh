#!/bin/bash
#SBATCH --job-name=qwen3_omni_sft_smoketest     # 工作名稱
#SBATCH --partition=normal                         # 使用的 partition (請根據你的系統修改)
#SBATCH --time=01:00:00                         # 小規模測試上限；正式跑請視實際 step 耗時再拉長
#SBATCH --account=MST113234     ####### 請記得換成您的計畫代碼 #######
#SBATCH --nodes=1                               # (-N) Maximum number of nodes to be allocated
#SBATCH --gpus-per-node=2                       # 2x H100 80G -- 見 run.sh/train.py 旁的評估：bf16 權重約 70GB，靠 deepspeed=zero3 切成兩張才有機會塞下，這正是這次小規模測試要驗證的地方
#SBATCH --cpus-per-task=8                       # (-c) Number of cores per MPI task -- 依 GPU 數等比例縮放
#SBATCH --ntasks-per-node=1                     # Maximum number of tasks on each node
#SBATCH -o ./logs/training_%j.log               # output file (%j expands to jobId)
#SBATCH -e ./logs/training_%j.err               # output file (%j expands to jobId)

ml load miniconda3/24.11.1
module load cuda/12.4
CONDA_DEFAULT_ENV="swift" # 你的 conda 環境名稱

# Qwen3-Omni-Instruct 權重路徑 -- 請改成叢集上實際的 materialize 路徑（或 HF hub id，
# 若該節點可連外網且有足夠磁碟自動下載）。
model_dir=/work/u1501463/model_qwen3_omni/

# 由 gen_1st_stage_data/run.sh 產生（--system-prompt-model-type qwen3_omni）的訓練資料，
# 尚未跟 stage1_RL/ 下的 Qwen2.5 版本共用，需重新產生一份。
dataset_path=/work/u1501463/stage1_qwen3_omni/train.jsonl

# LoRA checkpoint 續練用；"none"（預設）= 從 --model 從頭訓練，不接任何既有 adapter。
resume_from_checkpoint=none
# resume_from_checkpoint=/work/u1501463/stage1_training/<qwen3-omni-run>/checkpoint-XXX/

mkdir -p logs
echo "Start: $(date)"
echo "Node: $(hostname)"


echo "before conda"

# conda run -n "$CONDA_DEFAULT_ENV" python -u train.py
source $(conda info --base)/etc/profile.d/conda.sh

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


conda activate swift

# 多卡改用 torchrun 啟動（train.py 內用 ms-swift 的 Python API sft_main，跟 CLI 的
# `swift sft` 一樣仰賴 torchrun/accelerate 設好的 RANK/WORLD_SIZE/LOCAL_RANK 環境變數
# 才會走分散式路徑；單卡 `python -u train.py` 不會自動切成多卡）。
torchrun --nproc_per_node=${SLURM_GPUS_ON_NODE:-2} -u train.py \
        --model "${model_dir}" \
        --dataset "${dataset_path}" \
        --output_dir ./stage1_training \
        --resume_from_checkpoint "${resume_from_checkpoint}"

echo "after conda"
