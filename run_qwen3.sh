exp_name=no_vocal
python qwen3_with_tool_chain_evaluation.py \
    --tensor_parallel_size 4 \
    --output_path predictions/qwen3/MMAU_${exp_name}_overwrite.json \
    --tool_results_path ./apply_tool_results_MMAU_${exp_name}/ \
    --load_tool_chain_results \
    --overwrite_original_audio