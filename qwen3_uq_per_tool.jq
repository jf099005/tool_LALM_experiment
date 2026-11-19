# Per-tool accuracy, "ref" (the run the set was filtered on, from filter.json: Qwen2.5 for the
# AudioMCQ 50k/500k sets, quantized Qwen3 IQ2_M for audiomcq50k_lowp_*) vs bf16 Qwen3.
# CPU only (jq), safe on the login node. Usage, with S=<exported set>, R=<Qwen3 output dir>:
#   S=/work/u1501463/LALM_UQ/exports/audiomcq500k_p_true_delta_ge0.2
#   R=/work/u1501463/LALM_UQ/qwen3_uq/audiomcq500k_p_true_delta_ge0.2_n754
#   jq -n -r --slurpfile man $S/manifest.json --slurpfile flt $S/filter.json \
#     --slurpfile q3o $R/Qwen3-Omni-30B-A3B-Instruct_original_uq_results_no_tool_chain_results.json \
#     --slurpfile q3t $R/Qwen3-Omni-30B-A3B-Instruct_tool_applied_uq_results_no_tool_chain_results.json \
#     -f qwen3_uq_per_tool.jq
# Output: one line per tool chain: n, accuracy % original -> tool-applied, and the T->F / F->T counts.
($man[0] | map({(.id): ([.tool_chain[].tool] | join("+"))}) | add) as $tool
| ($flt[0].items | map({(.id): .}) | add) as $q25
| ($q3o[0] | map({(.id): .correct}) | add) as $o
| ($q3t[0] | map({(.id): .correct}) | add) as $t
| [$q3o[0][].id]
| group_by($tool[.])
| map({
    tool: $tool[.[0]],
    n: length,
    q25_tf: (map(select($q25[.].correct_original and ($q25[.].correct_tool_applied | not))) | length),
    q25_ft: (map(select(($q25[.].correct_original | not) and $q25[.].correct_tool_applied)) | length),
    q25_acc_o: (map(if $q25[.].correct_original then 1 else 0 end) | add / length * 100 | round),
    q25_acc_t: (map(if $q25[.].correct_tool_applied then 1 else 0 end) | add / length * 100 | round),
    q3_tf: (map(select($o[.] and ($t[.] | not))) | length),
    q3_ft: (map(select(($o[.] | not) and $t[.])) | length),
    q3_acc_o: (map(if $o[.] then 1 else 0 end) | add / length * 100 | round),
    q3_acc_t: (map(if $t[.] then 1 else 0 end) | add / length * 100 | round)
  })
| sort_by(-.n)[]
| "\(.tool)\tn=\(.n)\tref \(.q25_acc_o)->\(.q25_acc_t) (T>F \(.q25_tf), F>T \(.q25_ft))\tQ3 \(.q3_acc_o)->\(.q3_acc_t) (T>F \(.q3_tf), F>T \(.q3_ft))"
