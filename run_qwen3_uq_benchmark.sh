#!/bin/bash
# Qwen3-Omni UQ benchmark on an exported item set: no tool (original audio) vs tool (tool-applied).
#
# 1. On the local PC: export the items to benchmark, then copy them to the cluster:
#      python uq_threshold_confusion.py --on delta --metric p_true --threshold 0.1 \
#          --export-dir "$DATA_ROOT/exports/p_true_delta_ge0.1"
#      ./push_exports_to_nano5.sh p_true_delta_ge0.1
#    (lands in $REMOTE_EXPORT_ROOT/p_true_delta_ge0.1, see cluster_paths.sh)
# 2. On the cluster: run this script (Qwen3-Omni-30B-A3B in fp16 needs ~60 GB: 2x V100-32GB
#    or more). Interrupt and rerun freely -- finished items are skipped.
#
# Paths default to cluster_paths.sh. Env overrides: SET (exported set name), SUBSET_DIR,
# OUTPUT_DIR, MODEL, TP, DTYPE, BATCH_SIZE, PY.
cd "$(dirname "$0")"
source ./cluster_paths.sh

SET=${SET:-p_true_delta_ge0.1}
SUBSET_DIR=${SUBSET_DIR:-${REMOTE_EXPORT_ROOT}/${SET}}
OUTPUT_DIR=${OUTPUT_DIR:-${REMOTE_QWEN3_OUTPUT_ROOT}/$(basename "${SUBSET_DIR}")}
MODEL=${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}
TP=${TP:-2}
DTYPE=${DTYPE:-float16}          # V100 has no bf16; use bfloat16 on A100/H100
BATCH_SIZE=${BATCH_SIZE:-16}
PY=${PY:-python}

${PY} qwen3_uq_benchmark.py \
    --subset_dir "${SUBSET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --tensor_parallel_size ${TP} \
    --dtype ${DTYPE} \
    --batch_size ${BATCH_SIZE} \
    --download_dir "${REMOTE_HF_CACHE}" \
    --uq_num_samples 10 \
    --uq_sample_temperature 1.0 \
    --uq_p_true \
    --no-uq_semantic_entropy

# Re-print/recompute the summary later without loading the model:
#   ${PY} qwen3_uq_benchmark.py --subset_dir "${SUBSET_DIR}" --output_dir "${OUTPUT_DIR}" --summary_only
# Confusion figure / further filtering on the Qwen3 results:
#   python uq_threshold_confusion.py --threshold 0.1 --on delta \
#       --original "${OUTPUT_DIR}/Qwen3-Omni-30B-A3B-Instruct_original_uq_results_no_tool_chain_results.json" \
#       --tool-applied "${OUTPUT_DIR}/Qwen3-Omni-30B-A3B-Instruct_tool_applied_uq_results_no_tool_chain_results.json" \
#       --manifest "${SUBSET_DIR}/manifest.json" --run-tag qwen3
