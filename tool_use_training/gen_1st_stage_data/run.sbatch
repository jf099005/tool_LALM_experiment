#!/bin/bash
#SBATCH --job-name=gen_stage1_data              # 工作名稱
#SBATCH --partition=normal                      # 使用的 partition (請根據你的系統修改)
#SBATCH --time=08:00:00                         # AudioSR (diffusion) 是瓶頸且逐筆序列處理；先用少量 --num-samples 試跑量單筆耗時再調整
#SBATCH --account=MST113234     ####### 請記得換成您的計畫代碼 #######
#SBATCH --nodes=1                               # (-N) Maximum number of nodes to be allocated
#SBATCH --gpus-per-node=1                       # 1x H100 80G -- DL 工具的常駐 server (DeepFilterNet/AudioSR/SAM-Audio-large fp16) 加總遠低於 80G
#SBATCH --cpus-per-task=12                      # 對應 --workers (輕量 CPU 工具平行度)
#SBATCH --ntasks-per-node=1                     # Maximum number of tasks on each node
#SBATCH -o ./logs/gen_stage1_%j.log             # output file (%j expands to jobId)
#SBATCH -e ./logs/gen_stage1_%j.err             # output file (%j expands to jobId)

# Generates the stage-1 tool-use training set including the heavy DL tools
# (human_voice_enhance / super_resolution / extract_target / remove_target) that
# tools/tools_registry.py now registers. build_dataset.py starts one persistent server per DL
# conda env (deepfilternet / audiosr / sam_audio) so each model is loaded once; the servers need
# the GPU, hence sbatch instead of plain run.sh.
#
# Submit from this directory:  cd tool_use_training/gen_1st_stage_data && sbatch run.sbatch
#
# Prerequisites (see tools/smoke_test_dl_envs.sbatch): SAM-Audio weights are gated on HuggingFace
# (facebook/sam-audio-large) -- `hf auth login` / HF_TOKEN with granted access must already be
# in place, or every sample that draws extract_target/remove_target fails and gets redrawn.

ml load miniconda3/24.11.1
# sam_audio 的 torchcodec (cu126) 需要系統 CUDA 12 的 NPP 函式庫 (libnppicc.so.12 等)，
# 不載入的話 extract_target/remove_target 會在 import sam_audio 時失敗 (同 tools/smoke_test_dl_envs.sbatch)
module load cuda/12.4
CONDA_DEFAULT_ENV="swift-qwen3" # build_dataset.py 本身執行的環境 (librosa/soundfile/swift)

mkdir -p logs
echo "Start: $(date)"
echo "Node: $(hostname)"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$CONDA_DEFAULT_ENV"

# stage1_dir=/work/u1501463/stage1/
# stage1_dir=/work/u1501463/stage1_RL/
stage1_dir=/work/u1501463/stage1_full_toolset
# stage1_dir=./exp/

# Base model this data is being generated for -- used to auto-detect its own
# official default system message (chat_template.json/tokenizer_config.json)
# as the base of the training system prompt. Point this at whichever model
# dir stage1_training/ actually trains from; swap it when training a
# different model.
base_model_dir=/home/u1501463/.cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695/

python build_dataset.py \
    --sources audioset \
    --num-samples 5000   \
    --min-tools 1 \
    --max-tools 1   \
    --seed 0   \
    --workers 24   \
    --output-dir ${stage1_dir}/audio   \
    --output-file ${stage1_dir}/tool_usage_qa_audioset.json \
    --swift-output-file ${stage1_dir}/train.jsonl \
    --system-prompt-model-dir ${base_model_dir} \
    --system-prompt-model-type qwen3_omni\
    --use-cot

echo "End: $(date)"
