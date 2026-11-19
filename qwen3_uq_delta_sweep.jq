# ΔP(True) threshold sweep (0.2 .. 0.8) for a set selected on a reference run (filter.json items with
# p_true_*/correct_*: quantized Qwen3 IQ2_M for audiomcq50k_lowp_*), vs the bf16 Qwen3 results.
# CPU only (jq). $by = "ref" thresholds on the reference run's ΔP(True), "bf16" on bf16's own.
#   S=/work/u1501463/LALM_UQ/exports/audiomcq50k_lowp_ptrue_lt0.8_dp_ge0.2
#   R=/work/u1501463/LALM_UQ/qwen3_uq/audiomcq50k_lowp_ptrue_lt0.8_dp_ge0.2_n4533
#   jq -n -r --arg by ref --slurpfile flt $S/filter.json \
#     --slurpfile o $R/Qwen3-Omni-30B-A3B-Instruct_original_uq_results_no_tool_chain_results.json \
#     --slurpfile t $R/Qwen3-Omni-30B-A3B-Instruct_tool_applied_uq_results_no_tool_chain_results.json \
#     -f qwen3_uq_delta_sweep.jq
def pct: . * 1000 | round / 10;
def r3: . * 1000 | round / 1000;
($o[0] | map({(.id): .}) | add) as $bo
| ($t[0] | map({(.id): .}) | add) as $bt
| [$flt[0].items[] | select($bo[.id] and $bt[.id]) | {
    base: .base_id,
    rd: (.p_true_tool_applied - .p_true_original),
    rco: .correct_original, rct: .correct_tool_applied,
    bpo: $bo[.id].uncertainty.p_true, bpt: $bt[.id].uncertainty.p_true,
    bco: $bo[.id].correct, bct: $bt[.id].correct
  } | .bd = (.bpt - .bpo)] as $rows
| (if $by == "bf16" then "bf16" else "ref" end) as $key
| "threshold on \($key) ΔP(True)   (ref = \($flt[0].model // "reference run"))",
  "thr\tvariants\tquestions\tref acc o->t\tref T>F/F>T (net)\tbf16 acc o->t\tbf16 T>F/F>T (net)\tref ΔP\tbf16 ΔP\tbf16 share ΔP>=0.2",
  ([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8][] as $thr
   | [$rows[] | select((if $key == "bf16" then .bd else .rd end) >= $thr)] as $s
   | ($s | length) as $n
   | if $n == 0 then "\($thr)\t0" else
     ([$s[] | select(.rco and (.rct | not))] | length) as $rtf
     | ([$s[] | select((.rco | not) and .rct)] | length) as $rft
     | ([$s[] | select(.bco and (.bct | not))] | length) as $btf
     | ([$s[] | select((.bco | not) and .bct)] | length) as $bft
     | "\($thr)\t\($n)\t\([$s[].base] | unique | length)"
       + "\t\([$s[] | if .rco then 1 else 0 end] | add / $n | pct)->\([$s[] | if .rct then 1 else 0 end] | add / $n | pct)"
       + "\t\($rtf)/\($rft) (\($rft - $rtf))"
       + "\t\([$s[] | if .bco then 1 else 0 end] | add / $n | pct)->\([$s[] | if .bct then 1 else 0 end] | add / $n | pct)"
       + "\t\($btf)/\($bft) (\($bft - $btf))"
       + "\t\([$s[].rd] | add / $n | r3)\t\([$s[].bd] | add / $n | r3)"
       + "\t\([$s[] | select(.bd >= 0.2)] | length / $n | pct)%"
     end)
