#!/bin/bash
#SBATCH --job-name=uq_subset_stats
#SBATCH --partition=dev                         # short CPU-only job; every partition is GPU nodes, dev has the shortest queue
#SBATCH --account=MST113234
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1                       # not used -- only re-reads result JSONs; try 0 if your QOS allows GPU-less jobs
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH -o ./logs/uq_subset_stats_%j.log
#SBATCH -e ./logs/uq_subset_stats_%j.err

# Re-summarise an existing Qwen3 UQ run on the items whose ΔP(True) > MIN_DELTA, by the selection
# model (Qwen2.5, from filter.json) and by Qwen3 itself. No model is loaded.
# Submit from the repo root:   sbatch uq_subset_stats.sbatch
# Overrides: SET=<export name> RESULTS=<qwen3_uq output dir name> MIN_DELTA=0.4 sbatch uq_subset_stats.sbatch

ml load miniconda3/24.11.1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate verl-omni
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
mkdir -p logs
source ./cluster_paths.sh

SET=${SET:-audiomcq500k_p_true_delta_ge0.2}
RESULTS=${RESULTS:-audiomcq500k_p_true_delta_ge0.2_n754}
MIN_DELTA=${MIN_DELTA:-0.4}
echo "Start: $(date)  SET=$SET  RESULTS=$RESULTS  MIN_DELTA=$MIN_DELTA"

python uq_subset_stats.py \
    --subset_dir "${REMOTE_EXPORT_ROOT}/${SET}" \
    --results_dir "${REMOTE_QWEN3_OUTPUT_ROOT}/${RESULTS}" \
    --min_delta "${MIN_DELTA}"
STATUS=$?
echo "End: $(date)  exit=${STATUS}"
exit ${STATUS}
