# Machine-specific paths for running on the local PC (RTX 4070, 12 GB).
# Everything large (conda env, caches, datasets, model weights, outputs) lives on the
# data disk, not the nearly-full system disk. Source this before running anything:
#     source local_env.sh
# The data disk (/dev/sda4, label 新增磁碟區) is auto-mounted by udisks, and its mount
# name (新增磁碟區, 新增磁碟區3, ...) depends on mount order, so find it instead of hard-coding.
if [ -z "${DATA_ROOT:-}" ]; then
    for d in /media/jf099005-u/*/LALM_UQ; do
        [ -d "$d/conda_envs" ] && DATA_ROOT="$d" && break
    done
fi
if [ -z "${DATA_ROOT:-}" ]; then
    echo "local_env.sh: data disk not mounted (run: udisksctl mount -b /dev/sda4)" >&2
    return 1 2>/dev/null || exit 1
fi
export DATA_ROOT

export PY="${DATA_ROOT}/conda_envs/lalm_uq/bin/python"
export HF_HOME="${DATA_ROOT}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
# Keep the HF login token in the default place (`huggingface-cli login` writes it there), not
# under HF_HOME on the data disk. Needed for gated models (facebook/sam-audio-*).
export HF_TOKEN_PATH="${HOME}/.cache/huggingface/token"
export PIP_CACHE_DIR="${DATA_ROOT}/pip_cache"
export CONDA_PKGS_DIRS="${DATA_ROOT}/conda_pkgs"
export TMPDIR="${DATA_ROOT}/tmp"
export VLLM_CACHE_ROOT="${DATA_ROOT}/vllm_cache"
export TORCHINDUCTOR_CACHE_DIR="${DATA_ROOT}/torchinductor_cache"
export TRITON_CACHE_DIR="${DATA_ROOT}/triton_cache"

# Qwen2.5-Omni-7B (bf16, ~18 GB for the thinker) does not fit in 12 GB, so the local
# runs use Qwen2.5-Omni-3B. Fallback: Qwen2-Audio-7B-Instruct.
export UQ_MODEL="${DATA_ROOT}/models/Qwen2.5-Omni-3B"

export DCASE_QA_JSON="${DATA_ROOT}/datasets/dcase_7481_local.json"
export STAGE2_2_DIR="${DATA_ROOT}/datasets/gen_2nd_2_data_exp"
export UQ_RESULT_DIR="${DATA_ROOT}/exp/eval_result"

mkdir -p "${TMPDIR}" "${HF_HOME}"
