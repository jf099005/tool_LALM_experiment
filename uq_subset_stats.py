"""Re-summarise an existing Qwen3 UQ run on a sub-selection of its items (no model, CPU only).

Uses qwen3_uq_benchmark.summarize_group, so the numbers are computed exactly like the
<tag>_summary.json of the full run, just over fewer items. Two selection rules, both strict
(delta > threshold) and both on P(True) = tool-applied minus original:
  - reference: the run the set was filtered on (filter.json, e.g. Qwen2.5-Omni-3B)
  - qwen3:     the Qwen3 results themselves

    python uq_subset_stats.py \
        --subset_dir /work/u1501463/LALM_UQ/exports/audiomcq500k_p_true_delta_ge0.2 \
        --results_dir /work/u1501463/LALM_UQ/qwen3_uq/audiomcq500k_p_true_delta_ge0.2_n754 \
        --min_delta 0.4
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from qwen3_uq_benchmark import RESULT_SUFFIX, SIDES, load_previous_results, summarize_group


def pct(v):
    return "n/a" if v is None else f"{v:.2%}"


def report(title: str, ids, pairs, tools, ref) -> dict:
    group = [pairs[i] for i in ids]
    o = summarize_group(group)
    print(f"\n=== {title}: {o['n']} paired item(s), {len({i.split('__')[0] for i in ids})} distinct question(s) ===")
    if not o["n"]:
        return {"overall": o}
    c = o["confusion"]
    print(f"accuracy        : {pct(o['accuracy_original'])} -> {pct(o['accuracy_tool_applied'])}"
          f"   gain {o['accuracy_gain']:+.2%}")
    print(f"confusion       : T->T {c['T->T']}  T->F {c['T->F']}  F->T {c['F->T']}  F->F {c['F->F']}"
          f"   net {o['net_correct_gain']:+d}")
    for m, v in o["uq"].items():
        fmt = lambda x: "n/a" if x is None else f"{x:.4f}"
        print(f"{m:26}: {fmt(v['mean_original'])} -> {fmt(v['mean_tool_applied'])}   mean Δ {fmt(v['mean_delta'])}")
    out = {"overall": o}
    common = [i for i in ids if i in ref]
    if common:
        r_o = sum(ref[i]["correct_original"] for i in common) / len(common)
        r_t = sum(ref[i]["correct_tool_applied"] for i in common) / len(common)
        print(f"reference run   : {pct(r_o)} -> {pct(r_t)} on the same {len(common)} items")
        out["reference_run"] = {"n": len(common), "accuracy_original": r_o, "accuracy_tool_applied": r_t}
    by_tool = collections.defaultdict(list)
    for i in ids:
        by_tool[tools.get(i, "(unknown)")].append(pairs[i])
    out["per_tool"] = {t: summarize_group(p) for t, p in sorted(by_tool.items(), key=lambda kv: -len(kv[1]))}
    print(f"\n{'tool':24}{'n':>6}{'acc orig':>10}{'acc tool':>10}{'gain':>9}{'net':>6}")
    for t, g in out["per_tool"].items():
        print(f"{t:24}{g['n']:>6}{pct(g['accuracy_original']):>10}{pct(g['accuracy_tool_applied']):>10}"
              f"{g['accuracy_gain']:>+9.1%}{g['net_correct_gain']:>+6d}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset_dir", type=Path, required=True, help="the exported set (filter.json, manifest.json)")
    ap.add_argument("--results_dir", type=Path, required=True, help="output dir of the Qwen3 run")
    ap.add_argument("--tag", default="Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--min_delta", type=float, default=0.4, help="keep items with ΔP(True) > this")
    args = ap.parse_args()

    res = {side: {r["id"]: r for r in load_previous_results(str(args.results_dir / f"{args.tag}_{side}{RESULT_SUFFIX}"))}
           for side in SIDES}
    ids_all = [i for i in res["original"] if i in res["tool_applied"]]
    pairs = {i: (res["original"][i], res["tool_applied"][i]) for i in ids_all}
    tools = {r["id"]: "+".join(s["tool"] for s in r["tool_chain"]) or "(none)"
             for r in json.loads((args.subset_dir / "manifest.json").read_text(encoding="utf-8"))}
    flt = json.loads((args.subset_dir / "filter.json").read_text(encoding="utf-8"))
    ref = {it["id"]: it for it in flt.get("items", [])}
    ref_name = Path(flt.get("uq_results", {}).get("original", "reference")).name.split("_pool")[0].split("_original")[0]

    def p_true(i, side):
        return (res[side][i].get("uncertainty") or {}).get("p_true")

    sel_ref = [i for i in ids_all if i in ref
               and ref[i]["p_true_tool_applied"] - ref[i]["p_true_original"] > args.min_delta]
    sel_q3 = [i for i in ids_all if p_true(i, "original") is not None and p_true(i, "tool_applied") is not None
              and p_true(i, "tool_applied") - p_true(i, "original") > args.min_delta]

    summary = {
        "results_dir": str(args.results_dir), "subset_dir": str(args.subset_dir), "min_delta": args.min_delta,
        "n_total": len(ids_all),
        f"reference_delta_gt_{args.min_delta}": report(
            f"{ref_name} ΔP(True) > {args.min_delta} (selection model)", sel_ref, pairs, tools, ref),
        f"qwen3_delta_gt_{args.min_delta}": report(
            f"Qwen3 ΔP(True) > {args.min_delta}", sel_q3, pairs, tools, ref),
    }
    out = args.results_dir / f"{args.tag}_subset_p_true_delta_gt{args.min_delta}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
