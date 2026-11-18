#!/bin/bash
# 500k tool-applied UQ build, in slices (local PC). Every step leaves a marker and is
# resumable, so rerunning the script continues where it stopped. Start it as a service:
#     ./run_500k_pipeline.sh service      # systemd user unit, restarts itself after a crash/OOM
#     ./run_500k_pipeline.sh status       # progress
#     ./run_500k_pipeline.sh stop
#
# Steps:
#   0. plan     500k (question, tool) variants over the 94k AudioMCQ Clotho+AudioCaps questions:
#               each question reused with several *different* tools, none it already had in the
#               50k build; super_resolution disabled. -> $OUT/plan/slice_XX/input.json
#   1. pool UQ  UQ of every question's ORIGINAL audio, once (all variants compare against it).
#   per slice:
#   2. build    apply the planned tool to each item (original audio is not copied)
#   3. UQ       UQ of the slice's tool-applied audio
#   4. prune    delete tool-applied audio whose P(True) went down vs the original; records,
#               manifests and UQ text results are kept (see prune_slice.py for rebuilding)
#
# UQ: Qwen2.5-Omni-3B with FP8 weights (~1.6x faster than bf16 on the RTX 4070; tested
# 2026-11-02: same option chosen on 97.5% of items vs 99.4% for a bf16 rerun). Its numbers are
# therefore not mixed with the bf16 50k results -- originals are recomputed in step 1.
set -e
cd "$(dirname "$0")"
source ./local_env.sh

N_SLICES=${N_SLICES:-10}
TOTAL=${TOTAL:-500000}
OUT=${OUT:-${DATA_ROOT}/datasets/audiomcq500k}
POOL=${POOL:-${DATA_ROOT}/datasets/audiomcq_all_94k/uq_subset.json}
CFG=tool_use_training/gen_2nd_2_data/tool_config_local_ml_nosr.json
PREV_50K=${DATA_ROOT}/datasets/audiomcq_strong_50k_tool_applied_ml40
PREV_50K_SUBSET=${DATA_ROOT}/datasets/audiomcq_strong_50k/uq_subset.json
RES=${UQ_RESULT_DIR}/audiomcq500k
TAG=Qwen2.5-Omni-3B-fp8
UNIT=lalm-500k
LOG=${DATA_ROOT}/logs/${UNIT}.log
UQ_ARGS=(--model "${UQ_MODEL}" --quantization fp8 --batch_size 32 --max_num_seqs 96
         --max_num_batched_tokens 4096 --gpu_memory_utilization 0.82 --max_model_len 4096
         --max_audios_per_prompt 1 --compute_uncertainty --uq_num_samples 10
         --uq_sample_temperature 1.0 --cont)

# Wait until the GPU is free (e.g. the build's SAM-Audio server has released its memory):
# vLLM sizes itself from free memory at start-up and OOMs in sampler warm-up otherwise.
wait_gpu_free() {
    for _ in $(seq 1 60); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        [ "${used:-0}" -lt 1500 ] && return 0
        sleep 5
    done
    echo "warning: GPU still has ${used} MiB in use" >&2
}

case "${1:-run}" in
service)
    systemctl --user reset-failed "$UNIT" 2>/dev/null || true
    systemd-run --user --unit="$UNIT" --working-directory="$PWD" \
        -p Restart=on-failure -p RestartSec=60 -p StartLimitIntervalSec=3600 -p StartLimitBurst=10 \
        -p MemoryMax=29G \
        /bin/bash -c "echo \"===== \$(date) (re)started =====\" >> '$LOG'; exec ./run_500k_pipeline.sh run >> '$LOG' 2>&1"
    echo "log: $LOG"; exit 0 ;;
stop)   systemctl --user stop "$UNIT"; exit 0 ;;
status)
    echo "service : $(systemctl --user is-active "$UNIT" 2>/dev/null)  (restarts: $(systemctl --user show "$UNIT" -p NRestarts --value 2>/dev/null))"
    echo "step    : $(tr '\r' '\n' < "$LOG" 2>/dev/null | grep -E '^== ' | tail -n 1)"
    errs=$(tr '\r' '\n' < "$LOG" 2>/dev/null | grep -cE 'Traceback|OutOfMemory|Giving up' || true)
    echo "errors  : ${errs} line(s) with Traceback/OutOfMemory/Giving up in the log"
    for d in "$OUT"/slice_*; do
        [ -d "$d" ] || continue
        st=""; for m in built uq_done pruned; do [ -f "$d/.$m" ] && st="$st $m"; done
        echo "  $(basename "$d"):${st:- in progress}$( [ -f "$d/prune_summary.json" ] && echo "  $(cat "$d/prune_summary.json" | tr -d '\n ')")"
    done
    echo "progress: $(tr '\r' '\n' < "$LOG" 2>/dev/null | grep -E '/[0-9]+ \[' | tail -n 1 | sed 's/^ *//')"
    df -h "$DATA_ROOT" | tail -1
    exit 0 ;;
run) ;;
*) echo "usage: $0 run|service|status|stop" >&2; exit 2 ;;
esac

mkdir -p "$OUT" "$RES"

if [ ! -f "$OUT/plan/plan_summary.json" ]; then
    echo "== 0. plan"
    $PY tool_use_training/gen_2nd_2_data/plan_tool_variants.py --pool "$POOL" --tool-config "$CFG" \
        --exclude-dataset "$PREV_50K" --exclude-uq-subset "$PREV_50K_SUBSET" \
        --total "$TOTAL" --n-slices "$N_SLICES" --seed 1 --output-dir "$OUT/plan"
fi

if [ ! -f "$RES/.pool_original_done" ]; then
    echo "== 1. UQ of the question pool (original audio)"
    wait_gpu_free
    $PY qwen25_with_tool_chain_evaluation.py --subset_path "$POOL" \
        --output_path "$RES/${TAG}_pool_original_uq_results.json" "${UQ_ARGS[@]}"
    touch "$RES/.pool_original_done"
fi
POOL_UQ="$RES/${TAG}_pool_original_uq_results_no_tool_chain_results.json"

for IN in "$OUT"/plan/slice_*/input.json; do
    NAME=$(basename "$(dirname "$IN")")
    S="$OUT/$NAME"
    mkdir -p "$S"
    if [ ! -f "$S/.built" ]; then
        echo "== 2. build $NAME"
        N=$($PY -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$IN")
        $PY tool_use_training/gen_2nd_2_data/build_dataset.py --tool-config "$CFG" --input-json "$IN" \
            --num-samples "$N" --min-tools 1 --max-tools 1 --seed 1 --workers 12 \
            --no-copy-original --output-dir "$S"
        touch "$S/.built"
    fi
    TOOL_UQ="$RES/${TAG}_${NAME}_tool_applied_uq_results.json"
    if [ ! -f "$S/.uq_done" ]; then
        echo "== 3. UQ $NAME"
        wait_gpu_free
        $PY qwen25_with_tool_chain_evaluation.py --subset_path "$S/tool_applied_subset.json" \
            --output_path "$TOOL_UQ" "${UQ_ARGS[@]}"
        touch "$S/.uq_done"
    fi
    if [ ! -f "$S/.pruned" ]; then
        echo "== 4. prune $NAME"
        $PY tool_use_training/gen_2nd_2_data/prune_slice.py --slice-dir "$S" \
            --tool-uq "${TOOL_UQ%.json}_no_tool_chain_results.json" --original-uq "$POOL_UQ" --metric p_true
        touch "$S/.pruned"
    fi
done
echo "== all slices done $(date)"
