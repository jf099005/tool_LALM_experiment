"""Compare two UQ result files of the same items (before vs after a tool), no model, CPU only.

Works on any pair of `*_uq_results_no_tool_chain_results.json` files (Qwen2.5 or Qwen3 runs):
entries are paired by `id`. Prints the same overall block as qwen3_uq_benchmark's summary
(via its summarize_group), plus how many items' P(True) rose and by how much.

    python uq_pair_stats.py --before disturbed.json --after recovered.json --name "DCASE disturbed->recovered"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen3_uq_benchmark import summarize_group

THRESHOLDS = (0.0, 0.1, 0.2, 0.4)


def pct(v):
    return "n/a" if v is None else f"{v:.2%}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", type=Path, required=True, help="results without the tool (original / disturbed)")
    ap.add_argument("--after", type=Path, required=True, help="results with the tool (tool-applied / recovered)")
    ap.add_argument("--name", default=None, help="label for the printout")
    ap.add_argument("--save", type=Path, default=None, help="optional JSON output path")
    args = ap.parse_args()

    # Plain read-only json.load: load_previous_results would rename an unreadable file to
    # *.corrupt, and these inputs may be the only copy of an old run.
    before = {r["id"]: r for r in json.loads(args.before.read_text(encoding="utf-8"))}
    after = {r["id"]: r for r in json.loads(args.after.read_text(encoding="utf-8"))}
    ids = [i for i in before if i in after]
    pairs = [(before[i], after[i]) for i in ids]
    o = summarize_group(pairs)

    print(f"\n=== {args.name or args.after.stem}: {o['n']} paired item(s) "
          f"({len(before)} before / {len(after)} after) ===")
    c = o["confusion"]
    print(f"accuracy        : {pct(o['accuracy_original'])} -> {pct(o['accuracy_tool_applied'])}"
          f"   gain {o['accuracy_gain']:+.2%}")
    print(f"confusion       : T->T {c['T->T']}  T->F {c['T->F']}  F->T {c['F->T']}  F->F {c['F->F']}"
          f"   net {o['net_correct_gain']:+d}")
    for m, v in o["uq"].items():
        fmt = lambda x: "n/a" if x is None else f"{x:.4f}"
        print(f"{m:26}: {fmt(v['mean_original'])} -> {fmt(v['mean_tool_applied'])}   mean Δ {fmt(v['mean_delta'])}")

    deltas = []
    for b, a in pairs:
        pb, pa = (b.get("uncertainty") or {}).get("p_true"), (a.get("uncertainty") or {}).get("p_true")
        if pb is not None and pa is not None:
            deltas.append((pa - pb, b, a))
    by_threshold = {}
    if deltas:
        print(f"\nΔP(True) (after − before) over {len(deltas)} item(s) with P(True) on both sides:")
        print(f"{'rule':16}{'n':>6}{'share':>9}{'acc before':>12}{'acc after':>11}{'gain':>9}{'net':>6}")
        for t in THRESHOLDS:
            sel = [(b, a) for d, b, a in deltas if d > t]
            g = summarize_group(sel)
            by_threshold[f"delta_gt_{t}"] = g
            if not sel:
                print(f"{'Δ > ' + str(t):16}{0:>6}{'0.0%':>9}")
                continue
            print(f"{'Δ > ' + str(t):16}{g['n']:>6}{g['n'] / len(deltas):>9.1%}{pct(g['accuracy_original']):>12}"
                  f"{pct(g['accuracy_tool_applied']):>11}{g['accuracy_gain']:>+9.1%}{g['net_correct_gain']:>+6d}")
        down = [(b, a) for d, b, a in deltas if d < 0]
        g = summarize_group(down)
        by_threshold["delta_lt_0"] = g
        if down:
            print(f"{'Δ < 0':16}{g['n']:>6}{g['n'] / len(deltas):>9.1%}{pct(g['accuracy_original']):>12}"
                  f"{pct(g['accuracy_tool_applied']):>11}{g['accuracy_gain']:>+9.1%}{g['net_correct_gain']:>+6d}")

    if args.save:
        args.save.write_text(json.dumps({"name": args.name, "before": str(args.before), "after": str(args.after),
                                         "overall": o, "by_p_true_delta": by_threshold},
                                        indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nsaved: {args.save}")


if __name__ == "__main__":
    main()
