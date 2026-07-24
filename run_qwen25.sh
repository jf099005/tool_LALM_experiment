exp_name=speed_12

# Use the vllm conda env's python explicitly by absolute path.
# NOTE: on this machine, pyenv's shims take priority over conda in PATH even
# after `conda activate`, so bare `python`/`conda activate vllm && python`
# can silently run under pyenv's Python 3.10 instead of this env's 3.11.
# PYTHONNOUSERSITE=1 also guards against the system pip.conf (`user = true`)
# pulling in packages from ~/.local instead of this env's site-packages.
export PYTHONNOUSERSITE=1
PYTHON=~/miniconda3/envs/qwen3_omni/bin/python

$PYTHON qwen25_batch_evaluation.py\
    --subset_path ./dcase_subset.json \
    --output_path predictions/qwen25/Dcase_${exp_name}_overwrite.json \
    --batch_size 4
    # --tool_results_path ./apply_tool_results_MMAU_${exp_name}/tool_outputs \
    # --tensor_parallel_size 4 \