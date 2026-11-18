"""Qwen3-Omni (local llama.cpp run) vs Qwen2.5-Omni-3B on the same paired items: accuracy,
confusion, P(True) and the tool effect dP(True) = P(True | tool audio) - P(True | original),
overall and per tool. Works on a run that is still going (reads the *.partial.jsonl progress).

    python compare_qwen3_vs_qwen25.py --qwen3_dir RUN_DIR --qwen25_prefix PREFIX --manifest manifest.json
PREFIX + "_{original,tool_applied}_uq_results_no_tool_chain_results.json" are the Qwen2.5 files.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr


def load_results(path):
    rows = []
    if os.path.exists(path):
        rows = json.load(open(path, encoding="utf-8"))
    if os.path.exists(path + ".partial.jsonl"):
        for line in open(path + ".partial.jsonl", encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # line cut off by a crash
    return {r["id"]: r for r in rows}


def qwen3_side(run_dir, side):
    hits = [p for p in glob.glob(f"{run_dir}/*_{side}_uq_results_no_tool_chain_results.json*")]
    base = sorted({p.removesuffix(".partial.jsonl") for p in hits})
    assert len(base) == 1, f"expected one {side} result file in {run_dir}, got {base}"
    return load_results(base[0])


def pt(r):
    v = (r.get("uncertainty") or {}).get("p_true")
    return np.nan if v is None else float(v)


def tool_of(manifest, i):
    chain = (manifest.get(i) or {}).get("tool_chain") or []
    return chain[0]["tool"] if chain else "?"


def confusion(o, t):
    c = {"T->T": 0, "T->F": 0, "F->T": 0, "F->F": 0}
    for a, b in zip(o, t):
        c[("T" if a else "F") + "->" + ("T" if b else "F")] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen3_dir", required=True)
    ap.add_argument("--qwen25_prefix", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--json_out")
    a = ap.parse_args()

    q3 = {s: qwen3_side(a.qwen3_dir, s) for s in ("original", "tool_applied")}
    q25 = {s: {r["id"]: r for r in json.load(open(f"{a.qwen25_prefix}_{s}_uq_results_no_tool_chain_results.json"))}
           for s in ("original", "tool_applied")}
    manifest = json.load(open(a.manifest))
    if isinstance(manifest, list):  # build_dataset writes a list of {id, tool_chain, ...}
        manifest = {r["id"]: r for r in manifest}
    ids = [i for i in q3["original"] if i in q3["tool_applied"] and i in q25["original"] and i in q25["tool_applied"]]
    n = len(ids)
    out = {"n_paired": n}
    models = {"qwen3": q3, "qwen25": q25}
    d = {}
    print(f"paired items: {n}\n")
    print(f"{'':30s}{'Qwen3 (local quant)':>22s}{'Qwen2.5-Omni-3B':>18s}")
    for m, res in models.items():
        po = np.array([pt(res["original"][i]) for i in ids]); ptl = np.array([pt(res["tool_applied"][i]) for i in ids])
        co = [res["original"][i]["correct"] for i in ids]; ct = [res["tool_applied"][i]["correct"] for i in ids]
        d[m] = ptl - po
        out[m] = {
            "acc_original": float(np.mean(co)), "acc_tool": float(np.mean(ct)),
            "confusion": confusion(co, ct),
            "p_true_original": float(np.nanmean(po)), "p_true_tool": float(np.nanmean(ptl)),
            "delta_mean": float(np.nanmean(d[m])),
            "frac_delta_ge_0.1": float(np.nanmean(d[m] >= 0.1)), "frac_delta_le_-0.1": float(np.nanmean(d[m] <= -0.1)),
            "frac_delta_gt_0": float(np.nanmean(d[m] > 0)),
            "null_p_true": int(np.isnan(po).sum() + np.isnan(ptl).sum()),
        }
    for k in ["acc_original", "acc_tool", "p_true_original", "p_true_tool", "delta_mean",
              "frac_delta_ge_0.1", "frac_delta_le_-0.1", "frac_delta_gt_0", "null_p_true"]:
        print(f"{k:30s}{out['qwen3'][k]:22.4f}{out['qwen25'][k]:18.4f}")
    for m in models:
        print(f"{'confusion ' + m:30s}{json.dumps(out[m]['confusion'])}")

    ok = ~np.isnan(d["qwen3"]) & ~np.isnan(d["qwen25"])
    x, y = d["qwen3"][ok], d["qwen25"][ok]
    agree_pos = float(np.mean((x > 0) == (y > 0)))
    out["item_level"] = {"delta_pearson": float(pearsonr(x, y)[0]), "delta_spearman": float(spearmanr(x, y)[0]),
                         "agree_on_delta_gt_0": agree_pos}
    print(f"\nitem-level dP(True), Qwen3 vs Qwen2.5: Pearson {out['item_level']['delta_pearson']:.3f}  "
          f"Spearman {out['item_level']['delta_spearman']:.3f}  agree on (d>0) {agree_pos:.3f}")

    per = defaultdict(list)
    for k, i in enumerate(ids):
        if ok[k]:
            per[tool_of(manifest, i)].append(k)
    rows = []
    print(f"\n{'tool':24s}{'n':>7s}{'dP q3':>9s}{'±se':>7s}{'dP q25':>9s}{'±se':>7s}{'dAcc q3':>9s}{'dAcc q25':>10s}")
    idx = {i: k for k, i in enumerate(ids)}
    for tool, ks in sorted(per.items(), key=lambda kv: -len(kv[1])):
        ks = np.array(ks)
        r = {"tool": tool, "n": len(ks)}
        for m in models:
            dv = d[m][ks]
            r[f"dP_{m}"] = float(dv.mean()); r[f"se_{m}"] = float(dv.std(ddof=1) / np.sqrt(len(ks))) if len(ks) > 1 else np.nan
            sel = [ids[k] for k in ks]
            r[f"dAcc_{m}"] = float(np.mean([models[m]["tool_applied"][i]["correct"] for i in sel])
                                   - np.mean([models[m]["original"][i]["correct"] for i in sel]))
        rows.append(r)
        print(f"{tool:24s}{r['n']:7d}{r['dP_qwen3']:9.3f}{r['se_qwen3']:7.3f}{r['dP_qwen25']:9.3f}{r['se_qwen25']:7.3f}"
              f"{r['dAcc_qwen3']:9.3f}{r['dAcc_qwen25']:10.3f}")
    big = [r for r in rows if r["n"] >= 30]
    if len(big) >= 3:
        rs = spearmanr([r["dP_qwen3"] for r in big], [r["dP_qwen25"] for r in big])[0]
        rp = pearsonr([r["dP_qwen3"] for r in big], [r["dP_qwen25"] for r in big])[0]
        out["tool_level"] = {"n_tools": len(big), "dP_spearman": float(rs), "dP_pearson": float(rp)}
        print(f"\ntool-level mean dP(True) (tools with n>=30, {len(big)} tools): Spearman {rs:.3f}  Pearson {rp:.3f}")
    out["per_tool"] = rows
    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
