#!/bin/bash
#SBATCH --job-name=qwen3_omni_tool_use_eval     # 工作名稱
#SBATCH --partition=normal                      # 使用的 partition (請根據你的系統修改)
#SBATCH --time=02:00:00                         # 依 benchmark 規模調整
#SBATCH --account=MST113234     ####### 請記得換成您的計畫代碼 #######
#SBATCH --nodes=1                               # (-N) Maximum number of nodes to be allocated
#SBATCH --gpus-per-node=2                       # 1x H100 80G -- 單一推論行程未用 torchrun 切分，若模型+adapter 超出 80G 請自行調高
#SBATCH --cpus-per-task=8                       # (-c) Number of cores per MPI task
#SBATCH --ntasks-per-node=1                     # Maximum number of tasks on each node
#SBATCH -o ./logs/eval_%j.log                   # output file (%j expands to jobId)
#SBATCH -e ./logs/eval_%j.err                   # output file (%j expands to jobId)

ml load miniconda3/24.11.1
module load cuda/12.4
CONDA_DEFAULT_ENV="swift-qwen3" # 你的 conda 環境名稱

model_path=/work/u1501463/Qwen3-Omni-30B-A3B-Instruct/
#adapter_path=/work/u1501463/stage1_training/v6-20260815-112315/checkpoint-100/
adapter_path=/work/u1501463/stage1_model_full_toolset/v1-20261003-113849/checkpoint-100/
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

conda activate "$CONDA_DEFAULT_ENV"
trap 'kill $GPU_MONITOR_PID 2>/dev/null' EXIT

# GPU 健康檢查：在載入 30B 模型前先確認每張分配到的 GPU 都能正常初始化。
# (job 369600 在 hgpn17 上 GPU 1 卡在 100% utilization、0 記憶體，torch 讀不到
# GPU 數量，直到載入 adapter 才以 "device=1, num_gpus=" 的 assert 失敗。)
# 若這裡失敗，請排除該節點重新提交：sbatch --exclude=<node> run.sbatch
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
python - <<'EOF' || { echo "GPU preflight FAILED on $(hostname) -- resubmit with: sbatch --exclude=$(hostname) run.sbatch" >&2; exit 1; }
import os, sys, torch
expected = int(os.environ.get("SLURM_GPUS_ON_NODE", "0") or 0)
n = torch.cuda.device_count()
print(f"torch sees {n} GPU(s), SLURM allocated {expected}", flush=True)
if n == 0 or (expected and n != expected):
    sys.exit(f"GPU count mismatch: torch={n}, slurm={expected}")
for i in range(n):
    name = torch.cuda.get_device_name(i)
    torch.ones(1024, device=f"cuda:{i}").sum().item()  # actually run a kernel on it
    print(f"  cuda:{i} OK ({name})", flush=True)
EOF

python run_eval_parallel.py \
    --benchmark-file benchmark.json \
    --model "$model_path" \
    --adapter-dir "$adapter_path" \
    --model-type qwen3_omni_moe \
    --system-prompt-model-type qwen3_omni \
    --output-file results_qwen3_200.json \
    --dtype float16 \
    --batch-size "$batch_size"
status=$?

echo "End: $(date) (exit code $status)"
exit $status
