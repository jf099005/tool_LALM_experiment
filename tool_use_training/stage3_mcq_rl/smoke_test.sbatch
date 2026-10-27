#!/bin/bash
#SBATCH --job-name=s3_smoke
#SBATCH --partition=normal
#SBATCH --time=00:40:00                         # hard cap on cost; step 2 itself is bounded to 30 min below
#SBATCH --account=MST113234
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2                       # minimum that keeps TP=2 rollout + FSDP actor; `sbatch --gpus-per-node=4 ...` mirrors train.sbatch
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G                              # explicit (cgroup ConstrainRAMSpace=yes): offloaded actor+ref bf16 params ~2x62 GB + Ray/vLLM/tool servers
#SBATCH --ntasks-per-node=1
#SBATCH -o ./logs/smoke_%j.log
#SBATCH -e ./logs/smoke_%j.err
#SBATCH --mail-type=END,FAIL                    # Slurm mails on finish (any exit, incl. timeout); cluster MailProg=/bin/mailx
#SBATCH --mail-user=jf099005@gmail.com

# Stage-3 end-to-end smoke test, run BEFORE the real train.sbatch. Budget version (~2 GPU x <=40 min):
#   0. Env preflight: flashinfer.comm / vllm import (needs env_patches/apply_env_patches.sh on py3.10),
#      plus the process/thread/memory limits of this job (for diagnosing resource errors).
#   1. CPU agent-loop test (tests/smoke_agent_loop.py): token parity with stage-1 format.
#   2. A tiny real GSPO run: val on 2 MMAU items, then ONE training step (batch 2 x 2 rollouts).
#      Checks vllm (+cu129) starts on this node, heavy-tool servers come up, multi-turn rollouts
#      with real tool calls run (greedy val + sampled train), and one actor update completes.
# Proves the pipeline works, not train.sbatch's memory/throughput: vLLM runs eager (no CUDA-graph
# capture / torch.compile -- the slowest part of startup), max_num_seqs=16, 2048-token responses.
# Submit from this directory:  cd tool_use_training/stage3_mcq_rl && sbatch smoke_test.sbatch
# Uses the merged stage-1 model if merge_stage1.sbatch already ran, else the raw base model.
# Look for "SMOKE TEST PASSED" (step 1) and "SMOKE STEP 2 OK" at the end of the log.
#
# History: 410458 hung silently after Ray start (-> unbuffered output, num_cpus, timeout, Ray log copy);
# 412571 died with "pthread_create failed: Resource temporarily unavailable" in raylet, dashboard and
# the vLLM worker's Gloo group (-> thread-pool caps, fewer workers, no dashboard, explicit --mem,
# soft `ulimit -u` raised to the hard limit). See CLAUDE.md "Current status".

set -o pipefail
ml load miniconda3/24.11.1
module load cuda/12.4
export HEAVY_TOOL_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
module unload cuda/12.4
source $(conda info --base)/etc/profile.d/conda.sh
conda activate verl-omni
mkdir -p logs
echo "Start: $(date)  Node: $(hostname)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv

# --- Thread budget -------------------------------------------------------------------------
# On these 112-core nodes every library sizes its pools to the core count (OpenMP/MKL, torch,
# tokenizers/rayon, Ray gRPC), times ~20 processes (Ray workers, agent-loop + reward workers,
# vLLM TP workers, 3 heavy-tool servers) -> thousands of threads -> EAGAIN (job 412571).
# All of these are inherited by Ray workers, vLLM subprocesses and the tool servers.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_MAX_THREADS=2
export RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false TORCHINDUCTOR_COMPILE_THREADS=1
export RAY_num_server_call_thread=2
ulimit -S -u "$(ulimit -H -u)" 2>/dev/null   # soft -> hard user process/thread limit (no privileges needed)
echo "limits: nproc(soft/hard)=$(ulimit -S -u)/$(ulimit -H -u) stack=$(ulimit -s)KB open_files=$(ulimit -n)"
echo "threads already owned by $USER on this node: $(ps -u "$USER" -L --no-headers 2>/dev/null | wc -l)"
cat /sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_${SLURM_JOB_ID}/memory.limit_in_bytes 2>/dev/null \
    | awk '{printf "job memory limit: %.0f GB\n", $1/1e9}'

SMOKE_DATA=${SMOKE_DATA:-/work/u1501463/stage3_mcq_rl/data_smoke}
if [ ! -f "$SMOKE_DATA/train.parquet" ]; then
    python build_mcq_dataset.py --output-dir "$SMOKE_DATA" --compa-r-n 20 --dcase-n 20 --audioset-n 20 --val-n 8 || exit 1
fi

MERGED=/work/u1501463/stage3_mcq_rl/Qwen3-Omni-30B-A3B-stage1-merged
export MODEL_PATH=${MODEL_PATH:-$( [ -f "$MERGED/config.json" ] && echo "$MERGED" || echo /work/u1501463/Qwen3-Omni-30B-A3B-Instruct )}
echo "MODEL_PATH=$MODEL_PATH"

echo "===== step 0: env preflight (local patches, see ENV_NOTES.md) ====="
# vLLM workers import flashinfer.comm while building the TP communicator; on Python 3.10 it
# needs env_patches/apply_env_patches.sh (smoke job 390038). Fail here in seconds instead of
# after Ray + vLLM startup.
python -c "import sys, flashinfer.comm, vllm; print('python', sys.version.split()[0], '| flashinfer.comm OK | vllm', vllm.__version__)" \
    || { echo "PREFLIGHT FAILED: run 'bash env_patches/apply_env_patches.sh' in the verl-omni env (see ENV_NOTES.md)" >&2; exit 1; }

echo "===== step 1: CPU agent-loop smoke test ====="
python tests/smoke_agent_loop.py --parquet "$SMOKE_DATA/train.parquet" || exit 1

echo "===== step 2: tiny GSPO run with real tool calls ====="
# Layout follows the allocation: TP=2 rollout over all GPUs, heavy-tool servers on the last one.
# With 2 GPUs: vLLM at 0.5 util = 40 GB/GPU, ~31 GB of it weights, ~9 GB KV (>100k tokens) --
# ample for 4 trajectories; the tool models (~15 GB) share GPU 1's remaining half.
export NGPUS=${SLURM_GPUS_ON_NODE:-2}
export ROLLOUT_TP=2
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.5}
export HEAVY_TOOL_GPU=${HEAVY_TOOL_GPU:-$((NGPUS - 1))}
export STAGE3_TOOL_WORKDIR=/work/u1501463/stage3_mcq_rl/tool_work/${SLURM_JOB_ID}
export STAGE3_KEEP_TOOL_OUTPUTS=0
export STAGE3_TOOL_TIMEOUT=${STAGE3_TOOL_TIMEOUT:-180}   # slow call (e.g. AudioSR) -> tool error, not a stall
export TRAIN_FILE="$SMOKE_DATA/train.parquet"
export VAL_FILES="['$SMOKE_DATA/val.parquet']"
export OUT_ROOT=/work/u1501463/stage3_mcq_rl/smoke
export EXP_NAME=smoke_${SLURM_JOB_ID}
export TOTAL_STEPS=1
export PYTHONUNBUFFERED=1
STEP2_TIMEOUT=${STEP2_TIMEOUT:-30m}
RAY_LOG_COPY=/work/u1501463/stage3_mcq_rl/smoke/ray_logs/${SLURM_JOB_ID}
echo "NGPUS=$NGPUS HEAVY_TOOL_GPU=$HEAVY_TOOL_GPU STEP2_TIMEOUT=$STEP2_TIMEOUT"

# Thread-count samples for diagnosis: one at +4 min (412571 died during vLLM init at ~+4 min),
# then every 5 min (NCHC: no polling below 300 s).
(
    sleep 240
    while true; do
        echo "[threads $(date +%T)] $USER owns $(ps -u "$USER" -L --no-headers 2>/dev/null | wc -l) threads"
        sleep 300
    done
) &
THREAD_MONITOR_PID=$!

timeout --signal=TERM --kill-after=60 "$STEP2_TIMEOUT" \
python launch_with_tool_pool.py -- bash run_grpo.sh \
    ray_kwargs.ray_init.num_cpus=${SLURM_CPUS_PER_TASK:-16} \
    +ray_kwargs.ray_init.include_dashboard=False \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    reward.num_workers=2 \
    data.dataloader_num_workers=0 \
    data.train_batch_size=2 \
    data.val_max_samples=2 \
    data.max_response_length=2048 \
    actor_rollout_ref.rollout.response_length=2048 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.rollout.enforce_eager=True \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=16 \
    actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16] \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=true
STATUS=$?
kill ${THREAD_MONITOR_PID} 2>/dev/null
[ $STATUS -eq 124 ] && echo "STEP 2 TIMED OUT after $STEP2_TIMEOUT (hang) -- Ray logs: $RAY_LOG_COPY" >&2
if [ -d /tmp/ray/session_latest/logs ]; then
    mkdir -p "$RAY_LOG_COPY" && cp -rL /tmp/ray/session_latest/logs/. "$RAY_LOG_COPY/" 2>/dev/null
    echo "Ray session logs copied to $RAY_LOG_COPY"
fi
rm -rf "$STAGE3_TOOL_WORKDIR"
[ $STATUS -eq 0 ] && echo "SMOKE STEP 2 OK (val generations: $OUT_ROOT/val_generations/$EXP_NAME)"
echo "End: $(date)  exit=${STATUS}"
exit ${STATUS}
