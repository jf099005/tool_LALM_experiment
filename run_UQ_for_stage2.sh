exp_name=human_voice_amplify

# --- Uncertainty quantification controls -----------------------------------
# Number of stochastic samples drawn per item (temperature ~1.0) used for the
# sample-based entropy metrics below. Set to 0 to skip them entirely (e.g. if
# you only want P(True)).
uq_num_samples=10
uq_sample_temperature=1.0

# Per-method on/off switches. Each corresponds to --uq_xxx / --no-uq_xxx.
uq_predictive_entropy=true             # H_pred, requires uq_num_samples > 0
uq_length_normalized_entropy=true      # H_norm_tok, requires uq_num_samples > 0
uq_discrete_semantic_entropy=true      # H_disc + answer distribution, requires uq_num_samples > 0
uq_semantic_entropy=false              # H_sem via NLI entailment clustering (loads a cross-encoder model)
uq_p_true=true                         # P(True) self-verification (one extra generation per item)
# -----------------------------------------------------------------------------

result_path=/home/u1501463/tool_use_LALM/tool_use_training/gen_2nd_stage_data/exp/eval_result
mkdir -p ${result_path}

to_flag() {
    # $1 = flag base name (e.g. uq_p_true), $2 = true/false
    if [ "$2" = true ]; then
        echo "--$1"
    else
        echo "--no-$1"
    fi
}

original_path=tool_use_training/gen_2nd_2_data/exp/original_subset.json
tool_applied_path=tool_use_training/gen_2nd_2_data/exp/tool_applied_subset.json

echo "Running uncertainty quantification on original subset..."

python qwen25_with_tool_chain_evaluation.py \
    --subset_path ${original_path} \
    --compute_uncertainty \
    --output_path ${result_path}/disturbed_uq_results.json \
    --uq_num_samples ${uq_num_samples} \
    --uq_sample_temperature ${uq_sample_temperature} \
#     # --load_tool_chain_results \
#     # --overwrite_original_audio \
#     # --output_path predictions/qwen25/Dcase_${exp_name}_overwrite_uq.json \
#     # --tool_results_path ./tool_outputs/dcase_small_${exp_name}/tool_outputs/ \

#     # $(to_flag uq_predictive_entropy ${uq_predictive_entropy}) \
#     # $(to_flag uq_length_normalized_entropy ${uq_length_normalized_entropy}) \
#     # $(to_flag uq_discrete_semantic_entropy ${uq_discrete_semantic_entropy}) \
#     # $(to_flag uq_semantic_entropy ${uq_semantic_entropy}) \
#     # $(to_flag uq_p_true ${uq_p_true}) \

echo "Running uncertainty quantification on recovered subset..."

python qwen25_with_tool_chain_evaluation.py \
    --subset_path ${tool_applied_path} \
    --compute_uncertainty \
    --output_path ${result_path}/tool_applied_uq_results.json \
    --uq_num_samples ${uq_num_samples} \
    --uq_sample_temperature ${uq_sample_temperature} \
    # --load_tool_chain_results \
    # --overwrite_original_audio \
    # --output_path predictions/qwen25/Dcase_${exp_name}_overwrite_uq.json \
    # --tool_results_path ./tool_outputs/dcase_small_${exp_name}/tool_outputs/ \

    # $(to_flag uq_predictive_entropy ${uq_predictive_entropy}) \
    # $(to_flag uq_length_normalized_entropy ${uq_length_normalized_entropy}) \
    # $(to_flag uq_discrete_semantic_entropy ${uq_discrete_semantic_entropy}) \
    # $(to_flag uq_semantic_entropy ${uq_semantic_entropy}) \
    # $(to_flag uq_p_true ${uq_p_true}) \

