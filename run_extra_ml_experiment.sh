#!/bin/bash
# Small UQ experiment for the extra ML tools (tool_config_extra_ml.json: music_source_separation,
# speech_enhance, vad_keep_speech, speech_restore_voicefixer, dereverb_wpe): N_Q pool questions x
# every tool -> build -> FP8 UQ -> per-tool analysis vs the question's original-audio UQ (reused
# from the 500k run: same model, same FP8 settings).
# Resumable; start as a service with `./run_extra_ml_experiment.sh service`.
set -e
cd "$(dirname "$0")"
source ./local_env.sh

# SET / N_Q / SEED pick the experiment; the input (N_Q random pool questions x every tool in the
# config) is created on first run. Defaults: the 2000-question, 5-tool run of 2026-11-08.
SET=${SET:-extra_ml_2000}
N_Q=${N_Q:-2000}
SEED=${SEED:-8}
CFG=tool_use_training/gen_2nd_2_data/tool_config_extra_ml.json
X=${DATA_ROOT}/datasets/${SET}
RES=${UQ_RESULT_DIR}/${SET}
POOL_UQ=${UQ_RESULT_DIR}/audiomcq500k/Qwen2.5-Omni-3B-fp8_pool_original_uq_results_no_tool_chain_results.json
TAG=Qwen2.5-Omni-3B-fp8
UNIT=lalm-extra-ml-${SET//_/-}
LOG=${DATA_ROOT}/logs/${UNIT}.log

if [ "${1:-run}" = service ]; then
    systemctl --user reset-failed "$UNIT" 2>/dev/null || true
    systemd-run --user --unit="$UNIT" --working-directory="$PWD" -p MemoryMax=28G \
        /bin/bash -c "exec ./run_extra_ml_experiment.sh run >> '$LOG' 2>&1"
    echo "log: $LOG"; exit 0
fi

mkdir -p "$RES" "$X"
if [ ! -f "$X/input.json" ]; then
    $PY - "$X/input.json" "$CFG" "$N_Q" "$SEED" "$DATA_ROOT/datasets/audiomcq_all_94k/uq_subset.json" <<'PYEOF'
import json, random, sys
out, cfg, n_q, seed, pool = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
tools = [t for t, c in json.load(open(cfg))["tools"].items() if c.get("enabled")]
qs = random.Random(seed).sample(json.load(open(pool)), n_q)
items = [{**q, "id": f"{q['id']}__x{t}", "base_id": q["id"], "force_tools": [t]} for q in qs for t in tools]
json.dump(items, open(out, "w"), ensure_ascii=False)
print(f"input: {n_q} questions x {len(tools)} tools = {len(items)} items")
PYEOF
fi
if [ ! -f "$X/.built" ]; then
    echo "== build"
    $PY tool_use_training/gen_2nd_2_data/build_dataset.py \
        --tool-config "$CFG" \
        --input-json "$X/input.json" --num-samples "$($PY -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$X/input.json")" \
        --min-tools 1 --max-tools 1 --seed "$SEED" \
        --workers 6 --no-copy-original --max-attempts-per-sample 1 --output-dir "$X/build"
    touch "$X/.built"
fi
if [ ! -f "$X/.uq_done" ]; then
    echo "== UQ"
    for _ in $(seq 1 60); do  # wait for the tool server to release the GPU
        [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1500 ] && break; sleep 5
    done
    $PY qwen25_with_tool_chain_evaluation.py --subset_path "$X/build/tool_applied_subset.json" \
        --output_path "$RES/${TAG}_tool_applied_uq_results.json" \
        --model "${UQ_MODEL}" --quantization fp8 --batch_size 32 --max_num_seqs 96 \
        --max_num_batched_tokens 4096 --gpu_memory_utilization 0.82 --max_model_len 4096 \
        --max_audios_per_prompt 1 --compute_uncertainty --uq_num_samples 10 --uq_sample_temperature 1.0 --cont
    touch "$X/.uq_done"
fi
echo "== analysis"
$PY analyze_tool_uq.py --tool-uq "$RES/${TAG}_tool_applied_uq_results_no_tool_chain_results.json" \
    --manifest "$X/build/manifest.json" --original-uq "$POOL_UQ" \
    --baseline-slices "${DATA_ROOT}/datasets/audiomcq500k" \
    --baseline-uq-glob "${UQ_RESULT_DIR}/audiomcq500k/${TAG}_slice_*_tool_applied_uq_results_no_tool_chain_results.json" \
    --out "$RES/summary.json"
echo "== done $(date)"
