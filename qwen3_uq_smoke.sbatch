#!/bin/bash
#SBATCH --job-name=qwen3_uq_smoke
#SBATCH --partition=normal
#SBATCH --account=MST113234
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=00:40:00
#SBATCH -o ./logs/qwen3_uq_smoke_%j.log
#SBATCH -e ./logs/qwen3_uq_smoke_%j.err

# Qwen3-Omni UQ benchmark smoke test (QWEN3_UQ_BENCHMARK.md §4a): 8 items x 2 sides,
# separate output dir, always from scratch (--no-cont: old smoke results would otherwise be reused
# and nothing would run). Submit from the repo root:  sbatch qwen3_uq_smoke.sbatch
# Check the results as in QWEN3_UQ_BENCHMARK.md §5, then submit qwen3_uq_full.sbatch.
# Overrides: SET=<exported set> MODEL=<model dir> sbatch qwen3_uq_smoke.sbatch

ml load miniconda3/24.11.1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate verl-omni
unset ROCR_VISIBLE_DEVICES          # Slurm exports it; some GPU stacks reject it
export PYTHONUNBUFFERED=1
# Thread caps: stage-3 jobs on these nodes died with "pthread_create failed" without them.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
# FlashInfer's top-k/top-p sampler JIT-compiles with nvcc, which the GPU nodes don't have
# (job 427150: "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist").
# 0 = vLLM's PyTorch sampler; same sampling distribution, no compile.
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p logs
source ./cluster_paths.sh
echo "Start: $(date)  Node: $(hostname)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv

SET=${SET:-p_true_delta_ge0.1}
MODEL=${MODEL:-/work/u1501463/Qwen3-Omni-30B-A3B-Instruct}

echo "===== step 0: preflight (QWEN3_UQ_BENCHMARK.md §3, no model load) ====="
python -m py_compile qwen3_uq_benchmark.py || exit 1
python -c "import vllm, librosa, tqdm, flashinfer.comm; from vllm import LLM, SamplingParams; print('vllm', vllm.__version__)" || exit 1
python -c "from vllm import ModelRegistry; print([a for a in ModelRegistry.get_supported_archs() if 'Qwen3Omni' in a])" || exit 1
python -c "import qwen3_uq_benchmark; print('qwen3_uq_benchmark imports OK')" || exit 1

echo "===== step 1: 8-item run ====="
python qwen3_uq_benchmark.py \
    --subset_dir "${REMOTE_EXPORT_ROOT}/${SET}" \
    --output_dir "${REMOTE_QWEN3_OUTPUT_ROOT}/${SET}_smoke" \
    --model "${MODEL}" --tensor_parallel_size 2 --dtype bfloat16 \
    --max_model_len 8192 \
    --download_dir "${REMOTE_HF_CACHE}" \
    --batch_size 8 --limit 8 --no-cont \
    --uq_num_samples 10 --uq_sample_temperature 1.0 --uq_p_true --no-uq_semantic_entropy
STATUS=$?
echo "End: $(date)  exit=${STATUS}"
exit ${STATUS}
