#!/usr/bin/env python3
"""Per-tool effect of tool-applied audio on correctness and UQ, against each question's
original-audio UQ (paired on base_id, the 500k / extra-tool layout).

    python analyze_tool_uq.py --tool-uq RESULTS.json [...] --manifest manifest.json [...] \
        --original-uq POOL_ORIGINAL_RESULTS.json [--baseline-slices DIR] [--out summary.json]

For every tool: items, accuracy original -> tool-applied (gain), T->T/T->F/F->T/F->F,
mean ΔP(True) and the share/counts with ΔP(True) > 0 / > 0.1 / > 0.2, mean Δ discrete
semantic entropy. --baseline-slices adds the same table for the 500k tools restricted to the
same questions (needs the 500k slice manifests + UQ results), for a like-for-like comparison.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path


def load_results(path):
    p = Path(path)
    rows = {}
    for f in (p, Path(f"{p}.partial.jsonl")):
        if not f.exists():
            continue
        it = (json.loads(l) for l in f.open(encoding="utf-8") if l.strip()) if f.name.endswith(".jsonl") \
            else json.loads(f.read_text(encoding="utf-8"))
        for r in it:
            rows[r["id"]] = r
    return rows


def tool_of(manifests):
    out = {}
    for m in manifests:
        for rec in json.loads(Path(m).read_text(encoding="utf-8")):
            out[rec["id"]] = (rec.get("base_id"), rec["tool_chain"][0]["tool"] if rec.get("tool_chain") else "(none)",
                              (rec["tool_chain"][0].get("parameters") if rec.get("tool_chain") else {}))
    return out


def summarize(pairs):
    n = len(pairs)
    conf = collections.Counter(("T" if o["correct"] else "F") + "->" + ("T" if t["correct"] else "F") for o, t in pairs)
    def m(e, k):
        return (e.get("uncertainty") or {}).get(k)
    dp = [m(t, "p_true") - m(o, "p_true") for o, t in pairs if m(t, "p_true") is not None and m(o, "p_true") is not None]
    dh = [m(t, "discrete_semantic_entropy") - m(o, "discrete_semantic_entropy") for o, t in pairs
          if m(t, "discrete_semantic_entropy") is not None and m(o, "discrete_semantic_entropy") is not None]
    acc_o = sum(o["correct"] for o, _ in pairs) / n
    acc_t = sum(t["correct"] for _, t in pairs) / n
    return {
        "n": n, "acc_original": acc_o, "acc_tool": acc_t, "acc_gain": acc_t - acc_o,
        "confusion": {k: conf[k] for k in ("T->T", "T->F", "F->T", "F->F")},
        "mean_dp_true": sum(dp) / len(dp) if dp else None,
        "share_dp_gt0": sum(d > 0 for d in dp) / len(dp) if dp else None,
        "n_dp_gt0.1": sum(d > 0.1 for d in dp), "n_dp_gt0.2": sum(d > 0.2 for d in dp),
        "mean_d_disc_entropy": sum(dh) / len(dh) if dh else None,
    }


def table(title, by_tool):
    print(f"\n{title}")
    print(f"  {'tool':26}{'n':>5}{'acc orig':>9}{'acc tool':>9}{'gain':>8}{'T->F':>6}{'F->T':>6}"
          f"{'meanΔP':>8}{'P↑%':>6}{'ΔP>.1':>6}{'ΔP>.2':>6}{'ΔH_disc':>8}")
    for tool, s in sorted(by_tool.items(), key=lambda kv: kv[1]["acc_gain"]):
        c = s["confusion"]
        print(f"  {tool:26}{s['n']:>5}{s['acc_original']:>9.1%}{s['acc_tool']:>9.1%}{s['acc_gain']:>+8.1%}"
              f"{c['T->F']:>6}{c['F->T']:>6}{s['mean_dp_true']:>+8.3f}{s['share_dp_gt0']:>6.0%}"
              f"{s['n_dp_gt0.1']:>6}{s['n_dp_gt0.2']:>6}{s['mean_d_disc_entropy']:>+8.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tool-uq", nargs="+", required=True)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--original-uq", required=True)
    ap.add_argument("--baseline-slices", default=None, help="500k dir with slice_*/manifest.json")
    ap.add_argument("--baseline-uq-glob", default=None, help="glob of the 500k slice tool-applied UQ results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    orig = load_results(args.original_uq)
    tools = tool_of(args.manifest)
    res = {}
    for f in args.tool_uq:
        res.update(load_results(f))
    by = collections.defaultdict(list)
    for i, r in res.items():
        if i in tools and tools[i][0] in orig:
            by[tools[i][1]].append((orig[tools[i][0]], r))
    summary = {"new_tools": {t: summarize(p) for t, p in by.items()}}
    applied = collections.Counter(t[1] for t in tools.values())
    table("New tools (applied items; failed/not-applicable items are not in the manifest)", summary["new_tools"])
    questions = {tools[i][0] for i in res if i in tools}

    if args.baseline_slices and args.baseline_uq_glob:
        base_tools = tool_of(sorted(glob.glob(f"{args.baseline_slices}/slice_*/manifest.json")))
        keep = {i for i, (b, _, _) in base_tools.items() if b in questions}
        bres = {}
        for f in sorted(glob.glob(args.baseline_uq_glob)):
            for e in json.loads(Path(f).read_text(encoding="utf-8")):
                if e["id"] in keep:
                    bres[e["id"]] = e
        bby = collections.defaultdict(list)
        for i, r in bres.items():
            bby[base_tools[i][1]].append((orig[base_tools[i][0]], r))
        summary["baseline_tools_same_questions"] = {t: summarize(p) for t, p in bby.items()}
        table(f"500k tools on the same {len(questions)} questions (for comparison)", summary["baseline_tools_same_questions"])

    summary["applied_items_per_tool"] = dict(applied)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nsummary: {args.out}")


if __name__ == "__main__":
    main()
