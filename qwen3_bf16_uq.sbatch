#!/bin/bash
#SBATCH --job-name=qwen3_bf16_uq
#SBATCH --partition=normal
#SBATCH --account=MST113234
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH -o ./logs/qwen3_bf16_uq_%j.log
#SBATCH -e ./logs/qwen3_bf16_uq_%j.err

# Full-precision (bf16, vLLM) version of the PC's quantized Qwen3-Omni UQ run
# (run_qwen3_llamacpp_local.sh: llama.cpp IQ2_M, `qwen3_uq_benchmark.py --backend llamacpp`):
# same script, same prompt / P(True) protocol, vLLM backend instead of llama-server.
#
# Input: an unfiltered random sample exported on the PC by export_random_subset.py and uploaded
# with push_exports_to_nano5.sh (default SET: 10k of the 50k AudioMCQ tool-applied set).
#   SMOKE=1 sbatch --time=00:20:00 qwen3_bf16_uq.sbatch   # first: 32 items, separate output dir
#   sbatch qwen3_bf16_uq.sbatch                            # then the full set (resumable)
# Overrides: SET, OUT_NAME, MODEL, SAMPLES (default 10; 0 = P(True) + greedy only, like the PC run --
# the samples don't change the greedy answer or P(True), they only add the entropies), BATCH_SIZE.
#
# One GPU, not two: bf16 weights are ~62 GiB, so one H100 (79 GiB) fits them plus ~9 GiB of KV
# cache (~100k tokens) at 0.92 utilisation -- 4-5x what 128 concurrent sequences of <=700 tokens
# need. The earlier TP=2 runs (e.g. job 481760) left ~97% of their 2 x 39 GiB KV cache unused,
# i.e. paid for a second GPU they didn't need. If engine start-up OOMs, lower GPU_MEM_UTIL to 0.90
# or MAX_NUM_SEQS to 64 before going back to 2 GPUs.
#
# If the export carries the quantized run (reference/), compare_quant_uq.py runs at the end:
# bf16 (reference) vs quantized (candidate): same option, P(True) and ΔP(True) correlation.

ml load miniconda3/24.11.1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate verl-omni
unset ROCR_VISIBLE_DEVICES                 # Slurm exports it; some GPU stacks reject it
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0       # no nvcc on the GPU nodes for FlashInfer's JIT sampler (job 427150)
mkdir -p logs
source ./cluster_paths.sh
echo "Start: $(date)  Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

SET=${SET:-audiomcq50k_random10k_s0}
SUBSET_DIR=${REMOTE_EXPORT_ROOT}/${SET}
[ -f "${SUBSET_DIR}/original_subset.json" ] || {
    echo "No export at ${SUBSET_DIR}: run export_random_subset.py + push_exports_to_nano5.sh on the PC first" >&2
    exit 1; }
MODEL=${MODEL:-/work/u1501463/Qwen3-Omni-30B-A3B-Instruct}
SAMPLES=${SAMPLES:-10}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.92}
EXTRA=()
if [ "${SMOKE:-0}" = 1 ]; then
    OUT_NAME=${OUT_NAME:-${SET}_bf16_smoke}
    EXTRA+=(--limit 32 --no-cont)
else
    OUT_NAME=${OUT_NAME:-${SET}_bf16}
fi
OUTPUT_DIR=${REMOTE_QWEN3_OUTPUT_ROOT}/${OUT_NAME}
echo "SET=${SET}  OUTPUT=${OUTPUT_DIR}  samples=${SAMPLES}  batch=${BATCH_SIZE}  max_num_seqs=${MAX_NUM_SEQS}  gpu_mem=${GPU_MEM_UTIL}"

python qwen3_uq_benchmark.py --backend vllm \
    --subset_dir "${SUBSET_DIR}" --output_dir "${OUTPUT_DIR}" \
    --model "${MODEL}" --tensor_parallel_size 1 --dtype bfloat16 \
    --gpu_memory_utilization "${GPU_MEM_UTIL}" --max_model_len 4096 --max_num_seqs "${MAX_NUM_SEQS}" \
    --no_image_video --download_dir "${REMOTE_HF_CACHE}" \
    --batch_size "${BATCH_SIZE}" \
    --uq_num_samples "${SAMPLES}" --uq_sample_temperature 1.0 --uq_p_true --no-uq_semantic_entropy \
    "${EXTRA[@]}"
STATUS=$?

# bf16 vs the PC's quantized run on the same items (CPU only; numpy/scipy are in verl-omni).
if [ ${STATUS} -eq 0 ] && [ -d "${SUBSET_DIR}/reference" ]; then
    echo "===== bf16 (reference) vs quantized (candidate) ====="
    python compare_quant_uq.py "${OUTPUT_DIR}" "${SUBSET_DIR}/reference" \
        || echo "comparison failed (results are fine; rerun compare_quant_uq.py by hand)" >&2
fi
echo "End: $(date)  exit=${STATUS}"
exit ${STATUS}
