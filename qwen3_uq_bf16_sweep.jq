# bf16-only ΔP(True) threshold sweep (0.2 .. 0.8): selection and accuracy both from the bf16 Qwen3
# results; the reference run's values are not used. Optional --argjson pmax X adds the condition
# "bf16 original P(True) < X" (pmax 0.8 mirrors the quantized export rule). Overall + per tool.
#   R=/work/u1501463/LALM_UQ/qwen3_uq/audiomcq50k_lowp_ptrue_lt0.8_dp_ge0.2_n4533
#   jq -n -r --argjson pmax 2 --slurpfile o $R/<tag>_original_uq_results_no_tool_chain_results.json \
#     --slurpfile t $R/<tag>_tool_applied_uq_results_no_tool_chain_results.json \
#     --slurpfile man <export>/manifest.json -f qwen3_uq_bf16_sweep.jq
# Columns: group, thr, n, acc o->t, T>F/F>T (net), sign-test z, mean P(True) o->t.
def pct: . * 1000 | round / 10;
def r3: . * 1000 | round / 1000;
($man[0] | map({(.id): ([.tool_chain[].tool] | join("+"))}) | add) as $tool
| ($t[0] | map({(.id): .}) | add) as $bt
| [$o[0][] | select($bt[.id]) | {
    tool: ($tool[.id] // "(unknown)"),
    po: .uncertainty.p_true, pt: $bt[.id].uncertainty.p_true,
    co: .correct, ct: $bt[.id].correct
  } | select(.po != null and .pt != null and .po < $pmax) | .d = (.pt - .po)] as $rows
| def line($name; $g; $thr):
    [$g[] | select(.d >= $thr)] as $s | ($s | length) as $n
    | if $n == 0 then "\($name)\t\($thr)\t0" else
      ([$s[] | select(.co and (.ct | not))] | length) as $tf
      | ([$s[] | select((.co | not) and .ct)] | length) as $ft
      | "\($name)\t\($thr)\t\($n)"
        + "\t\([$s[] | if .co then 1 else 0 end] | add / $n | pct)->\([$s[] | if .ct then 1 else 0 end] | add / $n | pct)"
        + "\t\($tf)/\($ft) (\($ft - $tf))"
        + "\t\(if ($tf + $ft) > 0 then (($ft - $tf) / (($tf + $ft) | sqrt) * 10 | round / 10) else "n/a" end)"
        + "\t\([$s[].po] | add / $n | r3)->\([$s[].pt] | add / $n | r3)"
      end;
  "bf16 only; original P(True) < \($pmax); \($rows | length) variants before the ΔP threshold",
  "group\tthr\tn\tacc o->t\tT>F/F>T (net)\tz\tP(True) o->t",
  ([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8][] as $thr | line("ALL"; $rows; $thr)),
  ($rows | group_by(.tool) | sort_by(-length)[] as $g | [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8][] as $thr
   | line($g[0].tool; $g; $thr))
