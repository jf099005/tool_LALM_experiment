#!/bin/bash
# Qwen3-Omni-30B-A3B UQ (P(True)) on the local 12 GB RTX 4070 via llama.cpp.
#
# Default model: IQ2_M (imatrix, quantised from the ggml-org Q8_0), fully on the GPU, with an
# audio-only mmproj (vision tower stripped: frees ~0.6 GB). Measured 2026-11-18 on 150 random
# items of the 50k set: 0.50 s per (item, side), ~10x faster than Q4_K_M (5.4 s); vs Q4_K_M,
# chosen option agrees 90-94%, P(True) Spearman 0.83, dP(True) Pearson 0.66.
#
#   ./run_qwen3_llamacpp_local.sh server          # start llama-server only (unit qwen3-llama)
#   ./run_qwen3_llamacpp_local.sh stop            # stop server and run
#   SUBSET_DIR=... OUTPUT_DIR=... ./run_qwen3_llamacpp_local.sh service   # long run (unit qwen3-run)
#   ./run_qwen3_llamacpp_local.sh status
#
# The run is resumable (progress appended per batch); the service restarts the server and
# continues after a crash. Env: QUANT (IQ2_M | IQ2_XS | Q2_K | IQ3_XXS | Q4_K_M), NCMOE (MoE
# layers kept on CPU; IQ2_M/IQ2_XS: 0), PARALLEL, SAMPLES (0 = P(True) only).
cd "$(dirname "$0")"
source ./local_env.sh || exit 1
GGUF_DIR=${GGUF_DIR:-$DATA_ROOT/models/Qwen3-Omni-30B-A3B-Instruct-GGUF}
QUANT=${QUANT:-IQ2_M}
if [ "$QUANT" = Q4_K_M ]; then MODEL_FILE=$GGUF_DIR/Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf
else MODEL_FILE=$GGUF_DIR/requant/q8im-$QUANT.gguf; fi
MMPROJ=${MMPROJ:-$GGUF_DIR/mmproj-Qwen3-Omni-30B-A3B-Instruct-Q8_0-audio-only.gguf}
NCMOE=${NCMOE:-0}
PARALLEL=${PARALLEL:-4}
BIN=$DATA_ROOT/llama.cpp/build/bin
SRV_LOG=$DATA_ROOT/logs/qwen3_llama_server.log
TAG=Qwen3-Omni-30B-A3B-Instruct-$QUANT
SUBSET_DIR=${SUBSET_DIR:-$DATA_ROOT/exports/p_true_delta_ge0.1}
OUTPUT_DIR=${OUTPUT_DIR:-$DATA_ROOT/exp/qwen3_llamacpp/$(basename "$SUBSET_DIR")_$QUANT}
RUN_LOG=$OUTPUT_DIR/run.log

server_up() { curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; }

start_server() {
    systemctl --user stop qwen3-llama 2>/dev/null
    for i in $(seq 1 30); do [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
    # 1024 tokens per slot: the longest prompt (30 s clip) is ~470 tokens.
    systemd-run --user --unit=qwen3-llama --collect -E LD_LIBRARY_PATH="$BIN" \
        "$BIN/llama-server" -m "$MODEL_FILE" --mmproj "$MMPROJ" \
        -ngl 99 --n-cpu-moe "$NCMOE" -c $((1536 * PARALLEL)) --parallel "$PARALLEL" -ub 256 \
        -ctk q8_0 -ctv q8_0 -fa on --host 127.0.0.1 --port 8080 --log-file "$SRV_LOG" >/dev/null
    for i in $(seq 1 300); do
        server_up && return 0
        systemctl --user is-active -q qwen3-llama || { echo "server died on start"; return 1; }
        sleep 2
    done; return 1
}

run_once() {
    "$PY" qwen3_uq_benchmark.py --backend llamacpp --concurrency "$PARALLEL" --batch_size 16 \
        --subset_dir "$SUBSET_DIR" --output_dir "$OUTPUT_DIR" --model "$TAG" \
        --uq_num_samples "${SAMPLES:-0}" --uq_p_true --no-uq_semantic_entropy
}

case "${1:-}" in
server) start_server && echo "server ready ($QUANT)" ;;
stop) systemctl --user stop qwen3-run qwen3-llama 2>/dev/null; echo stopped ;;
status)
    systemctl --user is-active qwen3-run qwen3-llama
    tr '\r' '\n' < "$RUN_LOG" 2>/dev/null | grep -E "it/s\]|s/it\]" | tail -1
    grep -E "restart|done|FAILED" "$RUN_LOG" 2>/dev/null | tail -3 ;;
service)
    mkdir -p "$OUTPUT_DIR"
    systemd-run --user --unit=qwen3-run --collect \
        -E SUBSET_DIR="$SUBSET_DIR" -E OUTPUT_DIR="$OUTPUT_DIR" -E QUANT="$QUANT" -E NCMOE="$NCMOE" \
        -E PARALLEL="$PARALLEL" -E SAMPLES="${SAMPLES:-0}" "$PWD/$(basename "$0")" _loop
    echo "started qwen3-run -> $OUTPUT_DIR (log: $RUN_LOG)" ;;
_loop)
    exec >>"$RUN_LOG" 2>&1
    for attempt in $(seq 1 50); do
        echo "== $(date) attempt $attempt ($QUANT, ncmoe $NCMOE, parallel $PARALLEL)"
        server_up || start_server || { sleep 30; continue; }
        run_once && { echo "== $(date) done"; systemctl --user stop qwen3-llama; exit 0; }
        echo "== $(date) run failed, restarting server"; systemctl --user stop qwen3-llama; sleep 10
    done; echo "== $(date) FAILED after 50 attempts"; exit 1 ;;
*) echo "usage: $0 server|stop|status|service"; exit 1 ;;
esac
