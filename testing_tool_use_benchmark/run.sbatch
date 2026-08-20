#!/bin/bash
#SBATCH --job-name=qwen3_omni_tool_use_eval     # 工作名稱
#SBATCH --partition=normal                      # 使用的 partition (請根據你的系統修改)
#SBATCH --time=02:00:00                         # 依 benchmark 規模調整
#SBATCH --account=MST113234     ####### 請記得換成您的計畫代碼 #######
#SBATCH --nodes=1                               # (-N) Maximum number of nodes to be allocated
#SBATCH --gpus-per-node=1                       # 1x H100 80G -- 單一推論行程未用 torchrun 切分，若模型+adapter 超出 80G 請自行調高
#SBATCH --cpus-per-task=8                       # (-c) Number of cores per MPI task
#SBATCH --ntasks-per-node=1                     # Maximum number of tasks on each node
#SBATCH -o ./logs/eval_%j.log                   # output file (%j expands to jobId)
#SBATCH -e ./logs/eval_%j.err                   # output file (%j expands to jobId)

ml load miniconda3/24.11.1
module load cuda/12.4
CONDA_DEFAULT_ENV="swift-qwen3" # 你的 conda 環境名稱

model_path=/work/u1501463/Qwen3-Omni-30B-A3B-Instruct/
adapter_path=/work/u1501463/stage1_training/v0-20260814-011059/checkpoint-100/
# 同時在 GPU 上跑幾筆 sample (run_eval_batch.py 每個 turn 把這些 sample 疊在一起
# 送進同一次 engine.infer，取代逐筆生成) -- 數字越大 GPU 利用率越高，但 KV cache
# (每個 in-flight 對話都掛著好幾段 audio) 也吃更多顯存；30B 模型在 float16 下單卡
# 80G 本來就吃緊 (見上面 --gpus-per-node 的註解)，先從小的開始試，OOM 再往下調、
# 沒滿再往上調。
batch_size=4

mkdir -p logs
echo "Start: $(date)"
echo "Node: $(hostname)"

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

conda activate "$CONDA_DEFAULT_ENV"

python run_eval_batch.py \
    --benchmark-file benchmark.json \
    --model "$model_path" \
    --adapter-dir "$adapter_path" \
    --model-type qwen3_omni_moe \
    --system-prompt-model-type qwen3_omni \
    --output-file results_qwen3.json \
    --dtype float16 \
    --batch-size "$batch_size"

echo "End: $(date)"
