#!/bin/bash
# Local-PC version of run.sh: same build_dataset.py call, with all data on the data disk.
# Safe to interrupt (Ctrl+C) and rerun -- each step resumes where it stopped.
set -e
cd "$(dirname "$0")"
source ../../local_env.sh

# 0. Unpack the DCASE 2025 train audio (HF parquet mirror) and point dcase_7481.json at it.
$PY prepare_dcase_audio.py \
    --parquet-dir "${DATA_ROOT}/datasets/dcase2025-audio-qa-parquet/data" \
    --audio-root "${DATA_ROOT}/datasets/2025_DCASE_AudioQA_Official" \
    --qa-json ./dcase_7481.json \
    --output-json "${DCASE_QA_JSON}"

# 1. Sample real audio, apply a random tool per sample, write original/tool_applied subsets.
$PY build_dataset.py \
    --tool-config ./tool_config.json \
    --input-json "${DCASE_QA_JSON}" \
    --num-samples 50 \
    --min-tools 1 \
    --max-tools 1 \
    --output-dir "${STAGE2_2_DIR}" \
    --seed 0
