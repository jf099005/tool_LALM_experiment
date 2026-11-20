#!/bin/bash
#SBATCH --job-name=uq_pair_stats
#SBATCH --partition=dev                         # short CPU-only job; every partition is GPU nodes, dev has the shortest queue
#SBATCH --account=MST113234
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1                       # not used -- only re-reads result JSONs; try 0 if your QOS allows GPU-less jobs
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH -o ./logs/uq_pair_stats_%j.log
#SBATCH -e ./logs/uq_pair_stats_%j.err

# Old Qwen2.5-Omni UQ runs on DCASE 2025 AudioQA (stage-2 disturb -> recover experiment,
# tool_use_training/gen_2nd_stage_data/): accuracy, confusion, P(True) change, and how many items'
# P(True) rose by > 0 / 0.1 / 0.2 / 0.4. Read-only on the result files. No model is loaded.
# Submit from the repo root:   sbatch uq_pair_stats.sbatch

ml load miniconda3/24.11.1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate verl-omni
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
mkdir -p logs
echo "Start: $(date)"

D=tool_use_training/gen_2nd_stage_data
OUT=/work/u1501463/LALM_UQ/dcase_stage2_uq_stats
mkdir -p "$OUT"

python uq_pair_stats.py --name "DCASE disturbed -> recovered (results/)" \
    --before "$D/results/disturbed_uq_results_no_tool_chain_results.json" \
    --after  "$D/results/recovered_uq_results_no_tool_chain_results.json" \
    --save "$OUT/results_disturbed_vs_recovered.json" || exit 1

python uq_pair_stats.py --name "DCASE disturbed -> tool_applied (exp/eval_result/)" \
    --before "$D/exp/eval_result/disturbed_uq_results_no_tool_chain_results.json" \
    --after  "$D/exp/eval_result/tool_applied_uq_results_no_tool_chain_results.json" \
    --save "$OUT/eval_result_disturbed_vs_tool_applied.json" || exit 1

echo "End: $(date)  exit=0"
