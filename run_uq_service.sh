#!/bin/bash
# Run the local UQ pipeline step as a user systemd service that resumes on its own.
#
#   ./run_uq_service.sh start     # start (or restart) the service
#   ./run_uq_service.sh stop      # stop it (no automatic restart after a manual stop)
#   ./run_uq_service.sh status    # service state + progress
#   ./run_uq_service.sh log       # follow the log
#
# Why a service and not a retry loop in a script: when memory runs out, systemd-oomd (or the
# kernel OOM killer) kills the whole cgroup, loop included. systemd's Restart=on-failure
# restarts the unit after any abnormal exit -- OOM kill, crash, CUDA OOM -- and the pipeline
# resumes from its saved progress (finished items are skipped). A clean finish (exit 0) or
# `stop` does not restart.
#
# MemoryMax caps this service's RAM, so if it ever runs away, only this service is killed
# (then restarted), not the desktop session/terminal it was started from.
#
# Env overrides: STEPS (default "uq"; e.g. "tools uq"), UQ_MEMORY_MAX (default 24G),
# UNIT (default lalm-uq). Other run_local_uq.sh env vars (TOTAL_N, BATCH_SIZE, ...) pass through.
set -e
cd "$(dirname "$0")"
source ./local_env.sh

UNIT=${UNIT:-lalm-uq}
STEPS=${STEPS:-uq}
# UQ alone uses ~8 GB. The tools step needs more: the SAM-Audio server briefly takes ~26 GB
# while loading its checkpoint.
if [[ " $STEPS " == *" tools "* ]]; then MEM_MAX=${UQ_MEMORY_MAX:-29G}; else MEM_MAX=${UQ_MEMORY_MAX:-24G}; fi
LOG="${DATA_ROOT}/logs/${UNIT}.log"
STAGE3="$PWD/tool_use_training/stage3_mcq_rl"

pass_env=()
for v in TOTAL_N CONTRIBUTION SEED WORKERS TOOL_CONFIG PAIR_SUFFIX MODEL BATCH_SIZE GPU_MEM_UTIL \
         MAX_MODEL_LEN MAX_BATCHED_TOKENS MAX_NUM_SEQS PAIRED FRESH; do
    [ -n "${!v}" ] && pass_env+=(-E "$v=${!v}")
done

case "${1:-status}" in
start)
    if systemctl --user is-active --quiet "$UNIT"; then
        echo "$UNIT is already running; stopping it first (progress is kept)"
        systemctl --user stop "$UNIT"
    fi
    systemctl --user reset-failed "$UNIT" 2>/dev/null || true
    echo "===== $(date) start (STEPS=$STEPS) =====" >> "$LOG"
    systemd-run --user --unit="$UNIT" \
        --working-directory="$STAGE3" \
        -p Restart=on-failure -p RestartSec=60 \
        -p StartLimitIntervalSec=3600 -p StartLimitBurst=10 \
        -p MemoryMax="$MEM_MAX" -p MemorySwapMax=0 \
        -E STEPS="$STEPS" "${pass_env[@]}" \
        /bin/bash -c "echo \"===== \$(date) (re)started =====\" >> '$LOG'; exec ./run_local_uq.sh >> '$LOG' 2>&1"
    echo "log: $LOG"
    ;;
stop)
    systemctl --user stop "$UNIT"
    ;;
status)
    systemctl --user status "$UNIT" --no-pager 2>&1 | head -n 6 || true
    echo "restarts: $(systemctl --user show "$UNIT" -p NRestarts --value 2>/dev/null)"
    tr '\r' '\n' < "$LOG" 2>/dev/null | grep -E "/[0-9]+ \[" | tail -n 1
    for f in "${UQ_RESULT_DIR}"/*.partial.jsonl; do
        [ -f "$f" ] && echo "$(wc -l < "$f")  $(basename "$f")"
    done
    ;;
log)
    tail -f "$LOG"
    ;;
*)
    echo "usage: $0 start|stop|status|log" >&2; exit 2
    ;;
esac
