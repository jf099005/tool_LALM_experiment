exp_name=no_vocal

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

output_path=predictions/qwen25/MMAU_${exp_name}_overwrite_uq.json

to_flag() {
    # $1 = flag base name (e.g. uq_p_true), $2 = true/false
    if [ "$2" = true ]; then
        echo "--$1"
    else
        echo "--no-$1"
    fi
}

python qwen25_with_tool_chain_evaluation.py \
    --subset_path ./dcase_subset.json\
    --compute_uncertainty \
    --uq_num_samples ${uq_num_samples} \
    --uq_sample_temperature ${uq_sample_temperature} \
    # $(to_flag uq_predictive_entropy ${uq_predictive_entropy}) \
    # $(to_flag uq_length_normalized_entropy ${uq_length_normalized_entropy}) \
    # $(to_flag uq_discrete_semantic_entropy ${uq_discrete_semantic_entropy}) \
    # $(to_flag uq_semantic_entropy ${uq_semantic_entropy}) \
    # $(to_flag uq_p_true ${uq_p_true}) \
    --load_tool_chain_results \
    --overwrite_original_audio \
    --output_path predictions/qwen25/Dcase_${exp_name}_overwrite_uq.json \
    --tool_results_path ./tool_outputs/dcase_small_${exp_name}/tool_outputs/tool_outputs \
