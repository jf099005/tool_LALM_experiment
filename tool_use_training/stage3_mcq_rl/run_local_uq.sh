#!/bin/bash
# Local PC (RTX 4070, 12 GB): AudioMCQ Clotho + AudioCaps MCQs -> random tool applied
# (gen_2nd_2_data) -> UQ on the original and tool-applied audio.
# Every step is safe to interrupt and rerun; finished work is skipped.
#
#   TOTAL_N=50000 ./run_local_uq.sh            # default 50k
#   CONTRIBUTION=all ./run_local_uq.sh         # strong (default) | weak | all
#   STEPS="uq" ./run_local_uq.sh               # run only some of: audio mcq tools uq
set -e
cd "$(dirname "$0")"
source ../../local_env.sh

TOTAL_N=${TOTAL_N:-50000}
CONTRIBUTION=${CONTRIBUTION:-strong}
STEPS=${STEPS:-"audio mcq tools uq"}
SEED=${SEED:-0}
WORKERS=${WORKERS:-12}          # CPU processes for the tool-application step
# Tool set for the tool-application step. Default: light tools + ML tools (DeepFilterNet,
# AudioSR, SAM-Audio) at 40% total (`heavy_tool_probability`); needs the envs from
# tools/setup_local_heavy_envs.sh and an HF login with access to facebook/sam-audio-*.
TOOL_CONFIG=${TOOL_CONFIG:-../gen_2nd_2_data/tool_config_local_ml.json}
PAIR_SUFFIX=${PAIR_SUFFIX:-_ml40}

D="${DATA_ROOT}/datasets"
AUDIOMCQ_DIR="${D}/AudioMCQ/datasets"
CLOTHO_AUDIO="${D}/Clotho/audio"
AUDIOCAPS_AUDIO="${D}/AudioCaps/audio"
TAG="audiomcq_${CONTRIBUTION}_$((TOTAL_N / 1000))k"
MCQ_DIR="${D}/${TAG}"
PAIR_DIR="${D}/${TAG}_tool_applied${PAIR_SUFFIX}"

has() { [[ " ${STEPS} " == *" $1 "* ]]; }

if has audio; then
    echo "== [1/4] Clotho / AudioCaps audio"
    $PY prepare_audiomcq_audio.py \
        --audiomcq-dir "${AUDIOMCQ_DIR}" \
        --clotho-parquet-dir "${D}/Clotho/parquet/data" --clotho-out "${CLOTHO_AUDIO}" \
        --audiocaps-tar-dir "${D}/AudioCaps/raw" --audiocaps-out "${AUDIOCAPS_AUDIO}"
fi

if has mcq; then
    echo "== [2/4] ${TOTAL_N} MCQs (audio-contribution=${CONTRIBUTION})"
    $PY build_mcq_dataset.py --format uq \
        --output-dir "${MCQ_DIR}" --seed ${SEED} \
        --compa-r-n 0 --dcase-n 0 --audioset-n 0 \
        --clotho-n -1 --audiocaps-n -1 --total-n ${TOTAL_N} \
        --audiomcq-contribution ${CONTRIBUTION} \
        --audiomcq-dir "${AUDIOMCQ_DIR}" \
        --clotho-audio-dir "${CLOTHO_AUDIO}" --audiocaps-audio-dir "${AUDIOCAPS_AUDIO}"
fi

if has tools; then
    echo "== [3/4] apply one random tool per item"
    # Fail fast if SAM-Audio (gated) is enabled but its weights can't be fetched; otherwise
    # every extract/remove_target sample would fail one by one and be dropped.
    if grep -q 'sam_audio/bin/python' "${TOOL_CONFIG}"; then
        $PY -c "
import os, sys
from huggingface_hub import get_hf_file_metadata, hf_hub_url
repo = os.environ.get('SAM_AUDIO_MODEL', 'facebook/sam-audio-large')
try:
    get_hf_file_metadata(hf_hub_url(repo, 'config.json')); print('SAM-Audio access OK:', repo)
except Exception as e:
    sys.exit(f'Cannot access {repo} ({type(e).__name__}). Log in first: ~/anaconda3/bin/huggingface-cli login')
"
    fi
    $PY ../gen_2nd_2_data/build_dataset.py \
        --tool-config "${TOOL_CONFIG}" \
        --input-json "${MCQ_DIR}/uq_subset.json" \
        --num-samples ${TOTAL_N} --min-tools 1 --max-tools 1 \
        --output-dir "${PAIR_DIR}" --seed ${SEED} --workers ${WORKERS}
fi

if has uq; then
    echo "== [4/4] UQ"
    SUBSET_DIR="${PAIR_DIR}" RUN_TAG="$(basename "${UQ_MODEL}")_${TAG}${PAIR_SUFFIX}" ../../run_UQ_for_stage2_local.sh
fi
