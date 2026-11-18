"""Compare a quantised Qwen3-Omni UQ run against a reference run on the same items.

Each run dir is a qwen3_uq_benchmark.py --output_dir (original + tool_applied result files).
Reports, per side, the agreement of the chosen option and the correlation of P(True), and for
the tool effect (tool_applied - original) the correlation and sign agreement of dP(True) -- the
quantity the tool-selection experiments filter on.

    python compare_quant_uq.py REF_DIR CAND_DIR [CAND_DIR ...]
"""
import glob
import json
import re
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr


def load(run_dir):
    out = {}
    for side in ("original", "tool_applied"):
        (path,) = glob.glob(f"{run_dir}/*_{side}_uq_results_no_tool_chain_results.json")
        out[side] = {r["id"]: r for r in json.load(open(path))}
    return out


def option(pred):
    m = re.match(r"\s*\(?([A-Da-d])[\.\):]", pred or "")
    return m.group(1).upper() if m else (pred or "").strip()[:40]


def p_true(r):
    v = (r.get("uncertainty") or {}).get("p_true")
    return np.nan if v is None else v


def compare(ref, cand):
    ids = [i for i in ref["original"] if all(i in cand[s] and i in ref[s] for s in ref)]
    row = {"n": len(ids)}
    for s in ("original", "tool_applied"):
        a = np.array([p_true(ref[s][i]) for i in ids]); b = np.array([p_true(cand[s][i]) for i in ids])
        ok = ~np.isnan(a) & ~np.isnan(b)
        row[f"{s}_same_option"] = np.mean([option(ref[s][i]["prediction"]) == option(cand[s][i]["prediction"]) for i in ids])
        row[f"{s}_acc"] = np.mean([cand[s][i]["correct"] for i in ids])
        row[f"{s}_ref_acc"] = np.mean([ref[s][i]["correct"] for i in ids])
        row[f"{s}_ptrue_spearman"] = spearmanr(a[ok], b[ok])[0]
        row[f"{s}_ptrue_mean"] = np.nanmean(b); row[f"{s}_ref_ptrue_mean"] = np.nanmean(a)
        row[f"{s}_null_ptrue"] = int(np.isnan(b).sum())
    da = np.array([p_true(ref["tool_applied"][i]) - p_true(ref["original"][i]) for i in ids])
    db = np.array([p_true(cand["tool_applied"][i]) - p_true(cand["original"][i]) for i in ids])
    ok = ~np.isnan(da) & ~np.isnan(db)
    da, db = da[ok], db[ok]
    row["delta_pearson"] = pearsonr(da, db)[0]
    row["delta_spearman"] = spearmanr(da, db)[0]
    big = np.abs(da) >= 0.1  # items where the reference sees a real tool effect
    row["delta_sign_agree(|ref|>=0.1)"] = np.mean(np.sign(da[big]) == np.sign(db[big])) if big.any() else np.nan
    row["n_ref_big_delta"] = int(big.sum())
    row["delta_mean"] = db.mean(); row["ref_delta_mean"] = da.mean()
    return row


def main():
    ref = load(sys.argv[1])
    rows = {d.rstrip("/").split("/")[-1]: compare(ref, load(d)) for d in sys.argv[2:]}
    keys = list(next(iter(rows.values())))
    print(f"reference: {sys.argv[1]}")
    print(f"{'metric':34s}" + "".join(f"{k[:14]:>15s}" for k in rows))
    for k in keys:
        print(f"{k:34s}" + "".join(f"{rows[c][k]:15.3f}" if isinstance(rows[c][k], float) else f"{rows[c][k]:>15}" for c in rows))


if __name__ == "__main__":
    main()
