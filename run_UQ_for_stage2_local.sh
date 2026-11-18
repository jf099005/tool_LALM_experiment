#!/bin/bash
# Local-PC version of run_UQ_for_stage2.sh (RTX 4070, 12 GB).
# Runs UQ on an (original, tool_applied) subset pair made by gen_2nd_2_data/build_dataset.py.
#
# Safe to interrupt (Ctrl+C / kill) at any time: progress is appended after every batch,
# and rerunning the same command skips items already done. Pass FRESH=1 to start over.
#
# Env overrides:
#   SUBSET_DIR   dir with original_subset.json + tool_applied_subset.json (default: $STAGE2_2_DIR)
#   RUN_TAG      prefix for the result files (default: model name)
#   MODEL, BATCH_SIZE, GPU_MEM_UTIL, MAX_MODEL_LEN, MAX_BATCHED_TOKENS, MAX_NUM_SEQS
set -e
cd "$(dirname "$0")"
source ./local_env.sh

# --- Uncertainty quantification controls -----------------------------------
uq_num_samples=10
uq_sample_temperature=1.0
uq_predictive_entropy=true             # H_pred, requires uq_num_samples > 0
uq_length_normalized_entropy=true      # H_norm_tok, requires uq_num_samples > 0
uq_discrete_semantic_entropy=true      # H_disc + answer distribution, requires uq_num_samples > 0
uq_semantic_entropy=false              # H_sem via NLI entailment clustering (loads a cross-encoder model)
uq_p_true=true                         # P(True) self-verification (one extra generation per item)

# --- GPU memory / throughput controls (12 GB card) --------------------------
# Tested with Qwen2.5-Omni-3B: weights 8.8 GiB, ~1.2 GiB KV cache at 0.90.
# batch 16 x 10 samples runs ~0.6 s/item (vs ~1.8 s/item one item at a time).
# 0.92 with 32 sequences OOMs in vLLM's sampler warm-up.
model=${MODEL:-$UQ_MODEL}
batch_size=${BATCH_SIZE:-16}
gpu_memory_utilization=${GPU_MEM_UTIL:-0.90}
max_model_len=${MAX_MODEL_LEN:-4096}
max_num_batched_tokens=${MAX_BATCHED_TOKENS:-2048}
max_num_seqs=${MAX_NUM_SEQS:-32}
# -----------------------------------------------------------------------------

subset_dir=${SUBSET_DIR:-$STAGE2_2_DIR}
run_tag=${RUN_TAG:-$(basename "${model}")}

mkdir -p "${UQ_RESULT_DIR}"
resume_flag="--cont"
[ "${FRESH:-0}" = 1 ] && resume_flag="--no-cont"

to_flag() {
    if [ "$2" = true ]; then echo "--$1"; else echo "--no-$1"; fi
}

run_uq() {
    # $1 = subset json, $2 = output json, optional $3/$4 = paired subset/output (run in lockstep)
    local paired=()
    [ -n "$3" ] && paired=(--paired_subset_path "$3" --paired_output_path "$4")
    $PY qwen25_with_tool_chain_evaluation.py \
        --subset_path "$1" \
        --output_path "$2" \
        "${paired[@]}" \
        --model "${model}" \
        --batch_size ${batch_size} \
        --gpu_memory_utilization ${gpu_memory_utilization} \
        --max_model_len ${max_model_len} \
        --max_num_batched_tokens ${max_num_batched_tokens} \
        --max_num_seqs ${max_num_seqs} \
        --max_audios_per_prompt 1 \
        ${resume_flag} \
        --compute_uncertainty \
        --uq_num_samples ${uq_num_samples} \
        --uq_sample_temperature ${uq_sample_temperature} \
        $(to_flag uq_predictive_entropy ${uq_predictive_entropy}) \
        $(to_flag uq_length_normalized_entropy ${uq_length_normalized_entropy}) \
        $(to_flag uq_discrete_semantic_entropy ${uq_discrete_semantic_entropy}) \
        $(to_flag uq_semantic_entropy ${uq_semantic_entropy}) \
        $(to_flag uq_p_true ${uq_p_true})
}

# Both subsets in one process, interleaved item by item (PAIRED=0 runs them one after the
# other instead): the same items are always done on both sides, ready to compare.
orig_out="${UQ_RESULT_DIR}/${run_tag}_original_uq_results.json"
tool_out="${UQ_RESULT_DIR}/${run_tag}_tool_applied_uq_results.json"
if [ "${PAIRED:-1}" = 1 ]; then
    echo "Running uncertainty quantification on original + tool-applied subsets in lockstep (${subset_dir})..."
    run_uq "${subset_dir}/original_subset.json" "${orig_out}" "${subset_dir}/tool_applied_subset.json" "${tool_out}"
else
    echo "Running uncertainty quantification on original subset (${subset_dir})..."
    run_uq "${subset_dir}/original_subset.json" "${orig_out}"
    echo "Running uncertainty quantification on tool-applied subset (${subset_dir})..."
    run_uq "${subset_dir}/tool_applied_subset.json" "${tool_out}"
fi
