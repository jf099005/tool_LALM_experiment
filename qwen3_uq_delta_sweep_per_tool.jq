# Per-tool ΔP(True) threshold sweep (0.2 .. 0.8); same inputs and --arg by ref|bf16 as
# qwen3_uq_delta_sweep.jq, plus the manifest is not needed (filter.json items carry .tool).
# Output (TSV): tool, threshold, n, ref acc o->t, ref T>F/F>T (net), bf16 acc o->t, bf16 T>F/F>T (net),
# bf16 sign-test z = (F>T - T>F) / sqrt(F>T + T>F).
#   jq -n -r --arg by ref --slurpfile flt $S/filter.json --slurpfile o <bf16 original results> \
#     --slurpfile t <bf16 tool_applied results> -f qwen3_uq_delta_sweep_per_tool.jq
def pct: . * 1000 | round / 10;
($o[0] | map({(.id): .}) | add) as $bo
| ($t[0] | map({(.id): .}) | add) as $bt
| (if $by == "bf16" then "bf16" else "ref" end) as $key
| [$flt[0].items[] | select($bo[.id] and $bt[.id]) | {
    tool: (.tool // "(unknown)"),
    rd: (.p_true_tool_applied - .p_true_original),
    rco: .correct_original, rct: .correct_tool_applied,
    bd: ($bt[.id].uncertainty.p_true - $bo[.id].uncertainty.p_true),
    bco: $bo[.id].correct, bct: $bt[.id].correct
  }] as $rows
| "tool\tthr(\($key) ΔP)\tn\tref acc o->t\tref T>F/F>T (net)\tbf16 acc o->t\tbf16 T>F/F>T (net)\tbf16 z",
  ($rows | group_by(.tool) | sort_by(-length)[] as $g
   | [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8][] as $thr
   | [$g[] | select((if $key == "bf16" then .bd else .rd end) >= $thr)] as $s
   | ($s | length) as $n
   | if $n == 0 then "\($g[0].tool)\t\($thr)\t0" else
     ([$s[] | select(.rco and (.rct | not))] | length) as $rtf
     | ([$s[] | select((.rco | not) and .rct)] | length) as $rft
     | ([$s[] | select(.bco and (.bct | not))] | length) as $btf
     | ([$s[] | select((.bco | not) and .bct)] | length) as $bft
     | "\($g[0].tool)\t\($thr)\t\($n)"
       + "\t\([$s[] | if .rco then 1 else 0 end] | add / $n | pct)->\([$s[] | if .rct then 1 else 0 end] | add / $n | pct)"
       + "\t\($rtf)/\($rft) (\($rft - $rtf))"
       + "\t\([$s[] | if .bco then 1 else 0 end] | add / $n | pct)->\([$s[] | if .bct then 1 else 0 end] | add / $n | pct)"
       + "\t\($btf)/\($bft) (\($bft - $btf))"
       + "\t\(if ($btf + $bft) > 0 then (($bft - $btf) / (($btf + $bft) | sqrt) * 10 | round / 10) else "n/a" end)"
     end)
