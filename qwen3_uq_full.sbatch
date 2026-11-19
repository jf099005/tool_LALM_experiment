#!/bin/bash
#SBATCH --job-name=qwen3_uq_full
#SBATCH --partition=normal
#SBATCH --account=MST113234
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH -o ./logs/qwen3_uq_full_%j.log
#SBATCH -e ./logs/qwen3_uq_full_%j.err

# Qwen3-Omni UQ benchmark, full run over an exported set (QWEN3_UQ_BENCHMARK.md §4b).
# Submit from the repo root, only after qwen3_uq_smoke.sbatch looked right:
#   sbatch qwen3_uq_full.sbatch
# Resumable: if it hits the time limit, resubmit the same command (progress is in *.partial.jsonl).
# Timing: 469 items x 2 sides took ~3 min incl. start-up (job 428363). Scale --time for bigger sets.
# OUT_NAME names the output dir; default <SET>_n<item count>, read from the subset at run time.
# If an export is re-generated under the same SET name with the same item count, pass a new OUT_NAME,
# otherwise --cont would reuse results computed on the old audio.
# Default: audiomcq50k_lowp_ptrue_lt0.8_dp_ge0.2 (exported 2026-11-19 13:27): 4533 tool variants
# (3 rounds, one random tool each) of 2849 questions, selected on a quantized Qwen3-Omni (IQ2_M,
# llama.cpp on a PC): base P(True) < 0.8 and ΔP(True) >= 0.2. ~9k entries -> ~15-20 min.
# If the set carries the quantized run's values (filter.json items + base_ptrue_*.json),
# compare_quant_export_uq.py runs afterwards and writes <tag>_quant_comparison.json next to the results.
# Other sets: SET=audiomcq500k_p_true_delta_ge0.2 / p_true_delta_ge0.2 / p_true_delta_ge0.1.
# Previous results:
#   qwen3_uq/audiomcq500k_p_true_delta_ge0.2_n754 = 500k Qwen2.5 run, Δp_true >= 0.2
#   qwen3_uq/p_true_delta_ge0.2_n62   = 50k run, Δp_true >= 0.2 (job 428380)
#   qwen3_uq/p_true_delta_ge0.1       = old 147-item export (job 427219)
#   qwen3_uq/p_true_delta_ge0.1_n469  = 469-item export (job 428363); SET=p_true_delta_ge0.1 maps here
# Overrides: SET=<exported set> OUT_NAME=<output dir name> MODEL=<model dir> sbatch qwen3_uq_full.sbatch

ml load miniconda3/24.11.1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate verl-omni
unset ROCR_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0   # no nvcc on GPU nodes for FlashInfer's JIT sampler (see smoke sbatch)
mkdir -p logs
source ./cluster_paths.sh
echo "Start: $(date)  Node: $(hostname)"

SET=${SET:-audiomcq50k_lowp_ptrue_lt0.8_dp_ge0.2}
N_ITEMS=$(python -c "import json, sys; print(len(json.load(open(sys.argv[1]))))" \
    "${REMOTE_EXPORT_ROOT}/${SET}/original_subset.json") || exit 1
OUT_NAME=${OUT_NAME:-${SET}_n${N_ITEMS}}
MODEL=${MODEL:-/work/u1501463/Qwen3-Omni-30B-A3B-Instruct}
echo "SET=${SET}  items=${N_ITEMS}  OUTPUT=${REMOTE_QWEN3_OUTPUT_ROOT}/${OUT_NAME}"

python qwen3_uq_benchmark.py \
    --subset_dir "${REMOTE_EXPORT_ROOT}/${SET}" \
    --output_dir "${REMOTE_QWEN3_OUTPUT_ROOT}/${OUT_NAME}" \
    --model "${MODEL}" --tensor_parallel_size 2 --dtype bfloat16 \
    --max_model_len 8192 \
    --download_dir "${REMOTE_HF_CACHE}" \
    --batch_size 16 \
    --uq_num_samples 10 --uq_sample_temperature 1.0 --uq_p_true --no-uq_semantic_entropy
STATUS=$?

# Quantized (PC) vs bf16 comparison, CPU only, when the export carries the quantized run's values.
if [ ${STATUS} -eq 0 ] && ls "${REMOTE_EXPORT_ROOT}/${SET}"/base_ptrue_*.json >/dev/null 2>&1; then
    echo "===== quantized vs bf16 comparison ====="
    python compare_quant_export_uq.py \
        --subset_dir "${REMOTE_EXPORT_ROOT}/${SET}" \
        --results_dir "${REMOTE_QWEN3_OUTPUT_ROOT}/${OUT_NAME}" \
        || echo "comparison failed (results are fine; rerun compare_quant_export_uq.py by hand)" >&2
fi
echo "End: $(date)  exit=${STATUS}"
exit ${STATUS}
