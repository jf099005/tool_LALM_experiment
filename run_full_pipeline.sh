#!/bin/bash
# End-to-end: build the tool-applied dataset + Qwen2.5 UQ  ->  export the items whose UQ
# metric changed enough  ->  upload the export (problem jsons + audio) to the cluster.
#
#   ./run_full_pipeline.sh run        # all steps in this terminal (upload asks for the cluster password)
#   ./run_full_pipeline.sh service    # build + export as a systemd user service (auto-restart after
#                                     # crash/OOM); upload is left for you: ./run_full_pipeline.sh push
#   ./run_full_pipeline.sh push       # upload only (DRY_RUN=1: list, copy nothing)
#   ./run_full_pipeline.sh status     # which steps are done, service state, build progress
#   ./run_full_pipeline.sh stop       # stop the service
#
# Env:
#   PIPELINE   500k (default) | 50k
#                500k: run_500k_pipeline.sh run  ->  export_500k_uq_filtered.py
#                50k : stage3_mcq_rl/run_local_uq.sh  ->  uq_threshold_confusion.py --export-dir
#   METRIC     p_true (default) | predictive_entropy | discrete_semantic_entropy | ...
#   ON         delta (default; tool − original) | increase | abs_delta | ...  (uq_threshold_confusion.py)
#   THRESHOLD  0.2 (default)
#   STEPS      subset of "build export push" (default: all for `run`, "build export" for `service`)
#   REEXPORT=1 rebuild the export even if it is up to date
#
# Every step is resumable: the build steps keep their own progress markers, the export is
# skipped when its filter.json is newer than every UQ result file, and rsync only sends what
# is missing. On the cluster afterwards:  SET=<set name> ./run_qwen3_uq_benchmark.sh
set -e
cd "$(dirname "$0")"
source ./local_env.sh

PIPELINE=${PIPELINE:-500k}
METRIC=${METRIC:-p_true}
ON=${ON:-delta}
THRESHOLD=${THRESHOLD:-0.2}
UNIT=lalm-full-${PIPELINE}
LOG=${DATA_ROOT}/logs/${UNIT}.log

case "$PIPELINE" in
500k)
    SET="audiomcq500k_${METRIC}_${ON}_ge${THRESHOLD}"
    RESULTS=("${UQ_RESULT_DIR}/audiomcq500k/"*_no_tool_chain_results.json)
    BUILD_UNIT=lalm-500k ;;
50k)
    SET="${METRIC}_${ON}_ge${THRESHOLD}"
    RUN_TAG=$(basename "${UQ_MODEL}")_audiomcq_strong_50k_ml40
    RESULTS=("${UQ_RESULT_DIR}/${RUN_TAG}_original_uq_results_no_tool_chain_results.json"
             "${UQ_RESULT_DIR}/${RUN_TAG}_tool_applied_uq_results_no_tool_chain_results.json")
    BUILD_UNIT=lalm-uq ;;
*) echo "PIPELINE must be 500k or 50k" >&2; exit 2 ;;
esac
EXPORT_DIR="${DATA_ROOT}/exports/${SET}"

build_done() {
    if [ "$PIPELINE" = 500k ]; then
        local s n=0
        for s in "${DATA_ROOT}/datasets/audiomcq500k"/plan/slice_*/; do
            [ -f "${DATA_ROOT}/datasets/audiomcq500k/$(basename "$s")/.pruned" ] || return 1; n=$((n + 1))
        done
        [ "$n" -gt 0 ]
    else
        [ -f "${RESULTS[0]}" ] && [ -f "${RESULTS[1]}" ]
    fi
}

export_up_to_date() {
    [ -f "${EXPORT_DIR}/filter.json" ] || return 1
    local r
    for r in "${RESULTS[@]}"; do [ "$r" -nt "${EXPORT_DIR}/filter.json" ] && return 1; done
    return 0
}

step_build() {
    if systemctl --user is-active --quiet "$BUILD_UNIT"; then
        echo "The build service ${BUILD_UNIT} is already running; stop it first or wait for it." >&2; exit 1
    fi
    if build_done; then echo "== build: done (skipped)"; return; fi
    echo "== build (${PIPELINE}) $(date)"
    if [ "$PIPELINE" = 500k ]; then
        ./run_500k_pipeline.sh run
    else
        STEPS="audio mcq tools uq" tool_use_training/stage3_mcq_rl/run_local_uq.sh
    fi
    build_done || { echo "build finished without all its outputs" >&2; exit 1; }
}

step_export() {
    if export_up_to_date && [ "${REEXPORT:-0}" != 1 ]; then
        echo "== export: up to date -> ${EXPORT_DIR} (REEXPORT=1 to rebuild)"; return
    fi
    echo "== export: ${METRIC} (${ON}) >= ${THRESHOLD} -> ${EXPORT_DIR}  $(date)"
    if [ "$PIPELINE" = 500k ]; then
        $PY export_500k_uq_filtered.py --metric "$METRIC" --on "$ON" --threshold "$THRESHOLD" \
            --export-dir "$EXPORT_DIR"
    else
        $PY uq_threshold_confusion.py --metric "$METRIC" --on "$ON" --threshold "$THRESHOLD" \
            --run-tag "$RUN_TAG" --export-dir "$EXPORT_DIR" --out "$EXPORT_DIR/confusion.png"
    fi
}

step_push() {
    [ -f "${EXPORT_DIR}/filter.json" ] || { echo "No export at ${EXPORT_DIR}; run the export step first." >&2; exit 1; }
    if [ ! -t 0 ]; then
        echo "== push: skipped (no terminal for the cluster password). Run yourself:"
        echo "     PIPELINE=${PIPELINE} METRIC=${METRIC} ON=${ON} THRESHOLD=${THRESHOLD} ./run_full_pipeline.sh push"
        return
    fi
    echo "== push"
    ./push_exports_to_nano5.sh "$EXPORT_DIR"
}

run_steps() {
    local s
    for s in $1; do
        case "$s" in build) step_build ;; export) step_export ;; push) step_push ;;
        *) echo "unknown step: $s" >&2; exit 2 ;; esac
    done
    echo "== finished: ${1}  $(date)"
}

case "${1:-status}" in
run)  run_steps "${STEPS:-build export push}" ;;
push) step_push ;;
service)
    systemctl --user reset-failed "$UNIT" 2>/dev/null || true
    systemd-run --user --unit="$UNIT" --working-directory="$PWD" \
        -p Restart=on-failure -p RestartSec=60 -p StartLimitIntervalSec=3600 -p StartLimitBurst=10 \
        -p MemoryMax=29G \
        -E PIPELINE="$PIPELINE" -E METRIC="$METRIC" -E ON="$ON" -E THRESHOLD="$THRESHOLD" \
        -E REEXPORT="${REEXPORT:-0}" -E STEPS="${STEPS:-build export}" \
        /bin/bash -c "echo \"===== \$(date) (re)started =====\" >> '$LOG'; exec ./run_full_pipeline.sh _service >> '$LOG' 2>&1"
    echo "service: ${UNIT}   log: ${LOG}"
    echo "when it is done:  PIPELINE=${PIPELINE} METRIC=${METRIC} ON=${ON} THRESHOLD=${THRESHOLD} ./run_full_pipeline.sh push" ;;
_service) run_steps "${STEPS:-build export}" ;;
stop) systemctl --user stop "$UNIT" ;;
status)
    echo "pipeline: ${PIPELINE}   set: ${SET}"
    echo "service : ${UNIT} $(systemctl --user is-active "$UNIT" 2>/dev/null)"
    echo "build   : $(build_done && echo done || echo 'not done')"
    echo "export  : $(export_up_to_date && echo "up to date ($(du -sh "$EXPORT_DIR" | cut -f1))" || echo 'missing or stale')  ${EXPORT_DIR}"
    [ -f "$LOG" ] && echo "last    : $(tr '\r' '\n' < "$LOG" | grep -E '^== ' | tail -n 1)"
    if [ "$PIPELINE" = 500k ] && ! build_done; then ./run_500k_pipeline.sh status | sed 's/^/  /'; fi ;;
*) echo "usage: $0 run|service|push|status|stop" >&2; exit 2 ;;
esac
