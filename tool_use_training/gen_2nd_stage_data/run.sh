#!/bin/bash
# Stage-2 tool-use training data: disturb -> recover.
# See DISTURB_RECOVER_MAPPING.md for the disturb-op -> recovery-tool table.
#
# Needs librosa/soundfile for the disturbance step (audio_edit ops), so this
# whole script runs under the Whisper env; recovery execution shells out to
# each tool's own conda env on its own (see disturb_recover.py's TOOL_ENV).
PY=/home/u1501463/miniconda3/envs/Whisper/bin/python

exp=mmau_stage2
work_dir=/work/u1501463/gen_2nd_stage/${exp}

mkdir ${work_dir}
# 1. Sample QA pairs from a larger benchmark file (independent of disturb/recover --
#    swap --input for whatever dcase_subset.json/MMAU-shaped file(s) you're drawing from).
$PY sample_qa_pairs.py \
    --input /home/u1501463/tool_use_LALM/dcase_subset.json \
    --num-samples 200 \
    --seed 0 \
    --output ${work_dir}/sampled_qa_pairs.json

# 2/3/4. Disturb each sampled QA pair's audio, build+run the tool-call recovery
#    chain, and write disturbed_subset.json / recovered_subset.json + the
#    apply_tools.py-compatible per-turn schedules.
$PY build_by_disturb.py \
    --input-json ${work_dir}/sampled_qa_pairs.json \
    --output-dir ${work_dir}/disturbed \
    --tool-results-dir ${work_dir}/recovered \
    --min-ops 1 \
    --max-ops 1 \
    --seed 0

# 5. Both outputs are ready to evaluate directly:
#    python qwen25_with_tool_chain_evaluation.py --subset_path ${work_dir}/disturbed/disturbed_subset.json ...
#    python qwen25_with_tool_chain_evaluation.py --subset_path ${work_dir}/recovered/recovered_subset.json ...
