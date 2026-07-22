#!/bin/bash
# Stage-2 (2nd generation strategy) tool-use training data: sample real audio, apply a
# randomly drawn tools/ call straight to it -- no disturb/recover pairing (contrast with
# gen_2nd_stage_data/run.sh). tool_config.json controls which tools are in play and their
# parameter ranges.
#
# Needs librosa/soundfile/scipy for the light tools this script calls in-process, so this
# whole script runs under the Whisper env; any enabled heavy ML tool (remove_target/
# extract_target/human_voice_enhance/super_resolution) shells out to its own conda env on
# its own (see tool_config.json's per-tool "env").
PY=/home/u1501463/miniconda3/envs/Whisper/bin/python

exp=mmau_stage2_2
work_dir=./exp/

mkdir -p ${work_dir}

# Sample real audio (here: the DCASE 2025 QA subset -- swap --input-json for whatever
# dcase_subset.json/MMAU-shaped file(s) you're drawing from, or use --sources audioset vctk
# / --audio-glob for QA-less raw audio pools instead), apply a random tool chain per
# sample, and write original_subset.json / tool_applied_subset.json.
$PY build_dataset.py \
    --tool-config ./tool_config.json \
    --input-json ./dcase_subset.json \
    --num-samples 50 \
    --min-tools 1 \
    --max-tools 1 \
    --output-dir ${work_dir} \
    --seed 0

# Both outputs are ready to evaluate directly:
#    python qwen25_with_tool_chain_evaluation.py --subset_path ${work_dir}/original_subset.json ...
#    python qwen25_with_tool_chain_evaluation.py --subset_path ${work_dir}/tool_applied_subset.json ...
# (i.e. repoint run_UQ_for_stage2.sh's disturbed_path/recovered_path -- or a copy of it --
# at these two files.)
