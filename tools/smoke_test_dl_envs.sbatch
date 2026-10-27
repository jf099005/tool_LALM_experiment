#!/bin/bash
#SBATCH --job-name=dl_env_smoke
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --account=MST113234
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH -o ./logs/dl_env_smoke_%j.log
#SBATCH -e ./logs/dl_env_smoke_%j.err

# Smoke-tests each heavy DL tool through the same out-of-process path that
# tools/tools_registry.py::_run_tool_subprocess uses (tool_batch_execute.py under the
# tool's own conda env). Submit from the repo root:  sbatch tools/smoke_test_dl_envs.sbatch
# Model weights are downloaded on first use (DeepFilterNet: GitHub release; AudioSR: HF
# haoheliu/audiosr_*; SAM-Audio: gated HF facebook/sam-audio-large -> needs `hf auth login`
# or HF_TOKEN with granted access), so the compute node needs outbound internet unless the
# weights are already cached under ~/.cache.

mkdir -p ./logs
ml load miniconda3/24.11.1
module load cuda/12.4

ENVS=/home/u1501463/.conda/envs
REPO=/home/u1501463/tool_use_LALM
OUT=/work/u1501463/dl_env_smoke_test
SRC=$REPO/test.wav
mkdir -p "$OUT"

# SAM-Audio writes its output next to the input, so give it its own copy.
cp "$SRC" "$OUT/sam_input.wav"

cat > "$OUT/hve.json"  <<EOF
[{"audio_path": "$SRC", "output_path": "$OUT/hve_out.wav"}]
EOF
cat > "$OUT/sr.json"   <<EOF
[{"audio_path": "$SRC", "model_name": "basic", "output_path": "$OUT/sr_out.wav"}]
EOF
cat > "$OUT/ext.json"  <<EOF
[{"audio_path": "$OUT/sam_input.wav", "target_description": "vocals"}]
EOF

run() {  # run <env> <tool> <input.json>
    echo "=== $2 (env: $1) ==="
    "$ENVS/$1/bin/python" "$REPO/tools/tool_batch_execute.py" \
        --tool-name "$2" --input-file "$3" --output-file "$OUT/$2_result.json" \
        && cat "$OUT/$2_result.json"
    echo
}

run deepfilternet human_voice_enhance "$OUT/hve.json"
run audiosr       super_resolution    "$OUT/sr.json"
run sam_audio     extract_target      "$OUT/ext.json"
