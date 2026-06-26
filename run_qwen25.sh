exp_name=speed_12
python qwen25_with_tool_chain_evaluation.py \
    --load_tool_chain_results \
    --overwrite_original_audio \
    --output_path predictions/qwen25/MMAU_${exp_name}_overwrite.json \
    --tool_results_path ./apply_tool_results_MMAU_${exp_name}/tool_outputs \
    # --tensor_parallel_size 4 \
