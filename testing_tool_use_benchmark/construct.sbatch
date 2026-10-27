#!/bin/bash
#SBATCH --job-name=build_tool_use_benchmark     # 工作名稱
#SBATCH --partition=normal                      # 使用的 partition (請根據你的系統修改)
#SBATCH --time=02:00:00                         # 依 benchmark 規模調整
#SBATCH --account=MST113234     ####### 請記得換成您的計畫代碼 #######
#SBATCH --nodes=1                               # (-N) Maximum number of nodes to be allocated
#SBATCH --cpus-per-task=8                       # (-c) Number of cores per MPI task
#SBATCH --gpus-per-node=1                       # 1x H100 80G -- 單一推論行程未用 torchrun 切分，若模型+adapter 超出 80G 請自行調高
#SBATCH --ntasks-per-node=1                     # Maximum number of tasks on each node
#SBATCH -o ./logs/construct_%j.log              # output file (%j expands to jobId)
#SBATCH -e ./logs/construct_%j.err              # output file (%j expands to jobId)

# tools/tools_registry.py 現在包含 DL 工具 (human_voice_enhance / super_resolution /
# extract_target / remove_target)。build_benchmark.py 會先替 deepfilternet / audiosr / sam_audio
# 三個 conda env 各啟一個常駐 server (模型只載入一次、GPU 逐筆處理)，所以這個 job 需要 GPU。
# 前置條件：SAM-Audio 權重是 HuggingFace gated repo (facebook/sam-audio-large)，需先
# `hf auth login` 或設定 HF_TOKEN；否則抽到 extract_target/remove_target 的 sample 會失敗並重抽。
# 注意：下面沒指定 --output-file，會覆寫 testing_tool_use_benchmark/benchmark.json
# (含 DL 工具的新 benchmark 與舊的不可直接比較，要保留舊的請先備份或加 --output-file)。

ml load miniconda3/24.11.1
# sam_audio 的 torchcodec (cu126) 需要系統 CUDA 12 的 NPP 函式庫 (libnppicc.so.12 等)，
# 不載入的話 extract_target/remove_target 會在 import sam_audio 時失敗 (同 tools/smoke_test_dl_envs.sbatch)
module load cuda/12.4
CONDA_DEFAULT_ENV="swift-qwen3" # 你的 conda 環境名稱 (build_benchmark.py 需要在此環境下執行)

mkdir -p logs
echo "Start: $(date)"
echo "Node: $(hostname)"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$CONDA_DEFAULT_ENV"

python build_benchmark.py \
    --num-samples 100 \
    --min-tools 1 \
    --max-tools 1 \
    --exclude /work/u1501463/stage1_full_toolset/train.jsonl
echo "End: $(date)"

