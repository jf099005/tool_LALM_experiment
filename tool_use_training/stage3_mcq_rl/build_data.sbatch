#!/bin/bash
#SBATCH --job-name=s3_build_data
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --account=MST113234
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1                       # not used; set to 0 if your partition allows CPU-only jobs
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH -o ./logs/build_data_%j.log
#SBATCH -e ./logs/build_data_%j.err

# Step 2 of stage-3: build train/val/eval parquet (reads audio headers for duration filtering,
# extracts MMAU test-mini wavs from its parquet once). Extra args go to build_mcq_dataset.py,
# e.g.  sbatch build_data.sbatch --compa-r-n 12000 --audioset-n 3000

ml load miniconda3/24.11.1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate verl-omni
mkdir -p logs
echo "Start: $(date)  Node: $(hostname)"
python build_mcq_dataset.py --output-dir /work/u1501463/stage3_mcq_rl/data "$@"
echo "End: $(date)"
