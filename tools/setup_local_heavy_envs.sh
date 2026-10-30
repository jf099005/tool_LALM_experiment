#!/bin/bash
# Build the three heavy-tool conda envs on the local PC's data disk (one env per tool, as on
# the cluster; never imported into the main env -- they run as tools/heavy_tool_server.py).
#
#   tools/setup_local_heavy_envs.sh [deepfilternet] [audiosr] [sam_audio]   (default: all)
#
# Version notes:
#   deepfilternet 0.5.6 imports torchaudio.backend.common (removed in torchaudio 2.1) -> torch 2.0.1.
#   audiosr 0.0.7 pins old numpy/librosa/transformers                         -> torch 2.0.1, py3.10.
#   sam_audio needs python >= 3.11 and a recent torch/torchaudio/torchcodec.
set -e
source "$(dirname "$0")/../local_env.sh"
ENVS="${DATA_ROOT}/conda_envs"
CONDA=~/anaconda3/bin/conda
TORCH20=(torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118)
COMMON=(soundfile tqdm scipy)

make_env() {  # make_env <name> <python>
    [ -x "${ENVS}/$1/bin/python" ] || "${CONDA}" create -y -q -p "${ENVS}/$1" "python=$2"
}

build_deepfilternet() {
    make_env deepfilternet 3.10
    local P="${ENVS}/deepfilternet/bin/python"
    $P -m pip install "${TORCH20[@]}"
    $P -m pip install deepfilternet==0.5.6 "numpy<2" librosa "${COMMON[@]}"
}

build_audiosr() {
    make_env audiosr 3.10
    local P="${ENVS}/audiosr/bin/python"
    $P -m pip install "${TORCH20[@]}"
    $P -m pip install audiosr==0.0.7 "${COMMON[@]}"
    # audiosr pulls a PyPI torchvision built for a different CUDA than torch 2.0.1+cu118, and
    # librosa 0.9.2 (pinned by audiosr) imports pkg_resources, gone from new setuptools.
    $P -m pip install --force-reinstall --no-deps torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
    $P -m pip install "setuptools<70" matplotlib   # audiosr imports matplotlib but does not declare it
}

build_sam_audio() {
    make_env sam_audio 3.11
    local P="${ENVS}/sam_audio/bin/python"
    $P -m pip install torch torchaudio torchvision torchcodec
    $P -m pip install "git+https://github.com/facebookresearch/sam-audio.git" librosa "${COMMON[@]}"
    # sam_audio's ModelHubMixin._from_pretrained still takes proxies/resume_download, which
    # huggingface_hub >= 1.0 no longer passes; transformers 5 requires hub >= 1.0.
    $P -m pip install "huggingface_hub<1.0" "transformers>=4.54,<5"
}

targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(deepfilternet audiosr sam_audio)
for t in "${targets[@]}"; do
    echo "=== building ${t}"
    "build_${t}"
    echo "=== ${t} done"
done
