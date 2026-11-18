# Paths on the NCHC cluster (nano5), shared by push_exports_to_nano5.sh (runs locally) and
# run_qwen3_uq_benchmark.sh (runs on the cluster). Edit here, or override any of them with an
# environment variable of the same name.
export REMOTE_USER=${REMOTE_USER:-u1501463}
export REMOTE_HOST=${REMOTE_HOST:-nano5.nchc.org.tw}
export REMOTE_WORK=${REMOTE_WORK:-/work/u1501463}

# Exported item sets (uq_threshold_confusion.py --export-dir) are copied here, one dir each.
export REMOTE_EXPORT_ROOT=${REMOTE_EXPORT_ROOT:-${REMOTE_WORK}/LALM_UQ/exports}
# Qwen3-Omni UQ benchmark results, one dir per exported set.
export REMOTE_QWEN3_OUTPUT_ROOT=${REMOTE_QWEN3_OUTPUT_ROOT:-${REMOTE_WORK}/LALM_UQ/qwen3_uq}
# Model download cache (same as qwen3_with_tool_chain_evaluation.py's CACHE_DIR).
export REMOTE_HF_CACHE=${REMOTE_HF_CACHE:-${REMOTE_WORK}/hf_cache}
