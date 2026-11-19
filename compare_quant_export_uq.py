"""Compare a quantized Qwen3-Omni run (done on a PC, e.g. llama.cpp IQ2_M) with the full-precision
(bf16 vLLM) run of qwen3_uq_benchmark.py on the same exported set.

The quantized run's per-item values come with the export:
    filter.json        items[]: id, base_id, tool, round, p_true_original, p_true_tool_applied,
                       correct_original, correct_tool_applied      (one row per tool variant)
    base_ptrue_50k.json  {base_id: {p_true, correct}}               (original audio, all questions)
The quantized run has only P(True) and correctness (no entropies, no answer text), so the
comparison covers accuracy, P(True), their changes with the tool, agreement between the two
models, and how well P(True) separates right from wrong answers (AUROC).

Selection caveat: the set was chosen on the *quantized* model's values (low original P(True),
large ΔP(True)), so the quantized numbers are biased by construction; the bf16 numbers are not.

CPU only, standard library. Usage (also run at the end of qwen3_uq_full.sbatch):
    python compare_quant_export_uq.py --subset_dir /work/u1501463/LALM_UQ/exports/<SET> \
        --results_dir /work/u1501463/LALM_UQ/qwen3_uq/<OUT_NAME>
Writes <results_dir>/<tag>_quant_comparison.json and prints the tables.
"""
from __future__ import annotations

import collections
import json
import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Sequence

SIDES = ("original", "tool_applied")
RESULT_SUFFIX = "_uq_results_no_tool_chain_results.json"


# ---------------------------------------------------------------------------
# Small statistics helpers
# ---------------------------------------------------------------------------

def mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    mx, my = mean([p[0] for p in pairs]), mean([p[1] for p in pairs])
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sxx = sum((x - mx) ** 2 for x, _ in pairs)
    syy = sum((y - my) ** 2 for _, y in pairs)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else None


def ranks(xs: Sequence[float]) -> List[float]:
    """Average ranks (1-based), ties share the mean rank."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(xs, ys) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    return pearson(ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs]))


def auroc(scores, labels) -> Optional[float]:
    """P(score of a correct item > score of a wrong item); ties count 1/2 (Mann-Whitney)."""
    pairs = [(s, bool(l)) for s, l in zip(scores, labels) if s is not None and l is not None]
    n_pos = sum(l for _, l in pairs)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    r = ranks([s for s, _ in pairs])
    rank_sum_pos = sum(ri for ri, (_, l) in zip(r, pairs) if l)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def agreement(a: Sequence[bool], b: Sequence[bool]) -> dict:
    """Fraction of items both models get the same (right/wrong), and Cohen's kappa."""
    n = len(a)
    if n == 0:
        return {"n": 0, "agree": None, "kappa": None}
    agree = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return {"n": n, "agree": agree, "kappa": (agree - pe) / (1 - pe) if pe < 1 else None}


def confusion(orig: Sequence[bool], tool: Sequence[bool]) -> dict:
    c = collections.Counter(("T" if o else "F") + "->" + ("T" if t else "F") for o, t in zip(orig, tool))
    out = {k: c[k] for k in ("T->T", "T->F", "F->T", "F->F")}
    out["net"] = c["F->T"] - c["T->F"]
    return out


def acc(xs: Sequence[bool]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def load_bf16(results_dir: Path, tag: Optional[str]) -> tuple:
    if tag is None:  # the single <tag>_original...json in the dir
        found = sorted(results_dir.glob(f"*_original{RESULT_SUFFIX}"))
        if len(found) != 1:
            raise SystemExit(f"Pass --tag: found {len(found)} *_original{RESULT_SUFFIX} files in {results_dir}")
        tag = found[0].name[: -len(f"_original{RESULT_SUFFIX}")]
    res = {}
    for side in SIDES:
        p = results_dir / f"{tag}_{side}{RESULT_SUFFIX}"
        if not p.exists():
            raise SystemExit(f"Missing {p} (has the bf16 run finished? *.partial.jsonl means it is still running)")
        res[side] = {r["id"]: r for r in load_json(p)}
    return tag, res


def bf16_p_true(entry) -> Optional[float]:
    return (entry.get("uncertainty") or {}).get("p_true")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_rows(rows: List[dict]) -> dict:
    """rows: one per tool variant, with q_*/b_* (quantized / bf16) correctness and P(True) per side."""
    out = {"n": len(rows)}
    for who, label in (("q", "quantized"), ("b", "bf16")):
        po = [r[f"{who}_p_original"] for r in rows]
        pt = [r[f"{who}_p_tool_applied"] for r in rows]
        co = [r[f"{who}_c_original"] for r in rows]
        ct = [r[f"{who}_c_tool_applied"] for r in rows]
        a_o, a_t = acc(co), acc(ct)
        out[label] = {
            "accuracy_original": a_o,
            "accuracy_tool_applied": a_t,
            "accuracy_gain": (a_t - a_o) if rows else None,
            "confusion": confusion(co, ct),
            "mean_p_true_original": mean(po),
            "mean_p_true_tool_applied": mean(pt),
            "mean_delta_p_true": mean([t - o for o, t in zip(po, pt) if o is not None and t is not None]),
            "frac_delta_p_true_ge_0.2": mean([(t - o) >= 0.2 for o, t in zip(po, pt) if o is not None and t is not None]),
            "frac_p_true_original_lt_0.8": mean([o < 0.8 for o in po if o is not None]),
            "auroc_p_true_original": auroc(po, co),
            "auroc_p_true_tool_applied": auroc(pt, ct),
        }
    out["agreement"] = {}
    for side in SIDES:
        qp, bp = [r[f"q_p_{side}"] for r in rows], [r[f"b_p_{side}"] for r in rows]
        out["agreement"][side] = {
            **agreement([r[f"q_c_{side}"] for r in rows], [r[f"b_c_{side}"] for r in rows]),
            "p_true_pearson": pearson(qp, bp),
            "p_true_spearman": spearman(qp, bp),
            "p_true_mean_abs_diff": mean([abs(q - b) for q, b in zip(qp, bp) if q is not None and b is not None]),
        }
    qd = [r["q_p_tool_applied"] - r["q_p_original"] for r in rows]
    bd = [(r["b_p_tool_applied"] - r["b_p_original"])
          if r["b_p_tool_applied"] is not None and r["b_p_original"] is not None else None for r in rows]
    out["agreement"]["delta_p_true"] = {"pearson": pearson(qd, bd), "spearman": spearman(qd, bd)}
    # Same direction of the tool's effect on correctness (gain / loss / no change)?
    sign = lambda o, t: (t > o) - (t < o)
    out["agreement"]["correctness_change_same"] = mean(
        [sign(r["q_c_original"], r["q_c_tool_applied"]) == sign(r["b_c_original"], r["b_c_tool_applied"]) for r in rows])
    return out


def main():
    ap = ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--subset_dir", type=Path, required=True, help="exported set (filter.json, base_ptrue_*.json)")
    ap.add_argument("--results_dir", type=Path, required=True, help="qwen3_uq_benchmark.py --output_dir of the bf16 run")
    ap.add_argument("--tag", default=None, help="bf16 result-file prefix (default: auto-detect)")
    ap.add_argument("--base_ptrue", type=Path, default=None, help="default: the subset's base_ptrue_*.json, if any")
    args = ap.parse_args()

    flt = load_json(args.subset_dir / "filter.json")
    items = flt.get("items") or []
    need = {"p_true_original", "p_true_tool_applied", "correct_original", "correct_tool_applied"}
    if not items or not need <= set(items[0]):
        raise SystemExit(f"filter.json items lack the quantized-run fields {sorted(need)}; nothing to compare")
    tag, bf = load_bf16(args.results_dir, args.tag)
    quant_model = flt.get("model", "quantized")

    rows, missing = [], 0
    for it in items:
        o, t = bf["original"].get(it["id"]), bf["tool_applied"].get(it["id"])
        if o is None or t is None:
            missing += 1
            continue
        rows.append({
            "id": it["id"], "base_id": it.get("base_id", it["id"]), "tool": it.get("tool", "(unknown)"),
            "round": it.get("round"),
            "q_p_original": it["p_true_original"], "q_p_tool_applied": it["p_true_tool_applied"],
            "q_c_original": bool(it["correct_original"]), "q_c_tool_applied": bool(it["correct_tool_applied"]),
            "b_p_original": bf16_p_true(o), "b_p_tool_applied": bf16_p_true(t),
            "b_c_original": bool(o["correct"]), "b_c_tool_applied": bool(t["correct"]),
        })

    report = {
        "quantized_model": quant_model, "bf16_tag": tag, "subset_dir": str(args.subset_dir),
        "results_dir": str(args.results_dir), "selection": flt.get("filter"),
        "n_items_in_filter": len(items), "n_compared": len(rows), "n_missing_in_bf16": missing,
        "variants": compare_rows(rows),
        "per_tool": {}, "per_round": {},
    }
    by_tool, by_round = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        by_tool[r["tool"]].append(r)
        by_round[r["round"]].append(r)
    report["per_tool"] = {k: compare_rows(v) for k, v in sorted(by_tool.items(), key=lambda kv: -len(kv[1]))}
    report["per_round"] = {str(k): compare_rows(v) for k, v in sorted(by_round.items(), key=lambda kv: str(kv[0]))}

    # Original audio, one row per question: bf16 (mean over the question's variants -- greedy, so the
    # same answer) vs the quantized base run.
    base_path = args.base_ptrue or next(iter(sorted(args.subset_dir.glob("base_ptrue_*.json"))), None)
    if base_path and base_path.exists():
        base = load_json(base_path)
        per_q = collections.defaultdict(list)
        for r in rows:
            per_q[r["base_id"]].append(r)
        qrows = []
        for bid, rs in per_q.items():
            if bid not in base:
                continue
            ps = [r["b_p_original"] for r in rs if r["b_p_original"] is not None]
            qrows.append({
                "q_p": base[bid].get("p_true"), "q_c": bool(base[bid].get("correct")),
                "b_p": mean(ps), "b_c": mean([r["b_c_original"] for r in rs]) >= 0.5,
                "b_c_consistent": len({r["b_c_original"] for r in rs}) == 1,
            })
        qp, bp = [r["q_p"] for r in qrows], [r["b_p"] for r in qrows]
        report["questions_original_audio"] = {
            "base_file": str(base_path), "n_questions": len(qrows),
            "quantized": {"accuracy": acc([r["q_c"] for r in qrows]), "mean_p_true": mean(qp),
                          "auroc_p_true": auroc(qp, [r["q_c"] for r in qrows])},
            "bf16": {"accuracy": acc([r["b_c"] for r in qrows]), "mean_p_true": mean(bp),
                     "auroc_p_true": auroc(bp, [r["b_c"] for r in qrows]),
                     "frac_p_true_lt_0.8": mean([p < 0.8 for p in bp if p is not None])},
            "agreement": {**agreement([r["q_c"] for r in qrows], [r["b_c"] for r in qrows]),
                          "p_true_pearson": pearson(qp, bp), "p_true_spearman": spearman(qp, bp),
                          "p_true_mean_abs_diff": mean([abs(q - b) for q, b in zip(qp, bp) if q is not None and b is not None])},
            "bf16_same_correctness_across_variants": mean([r["b_c_consistent"] for r in qrows]),
            # Whole quantized base run (all questions), for context.
            "quantized_all_questions": {"n": len(base), "accuracy": acc([bool(v.get("correct")) for v in base.values()]),
                                        "mean_p_true": mean([v.get("p_true") for v in base.values()])},
        }

    out = args.results_dir / f"{tag}_quant_comparison.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_report(report, out)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def f(x, pct=False, signed=False):
    if x is None:
        return "n/a"
    if pct:
        return f"{x:+.1%}" if signed else f"{x:.1%}"
    return f"{x:+.3f}" if signed else f"{x:.3f}"


def print_report(r: dict, out: Path):
    v, qn = r["variants"], r["quantized_model"]
    print(f"\n=== {qn} (quantized) vs {r['bf16_tag']} (bf16), {r['n_compared']} tool variants ===")
    print(f"selection: {r['selection']}")
    if r["n_missing_in_bf16"]:
        print(f"WARNING: {r['n_missing_in_bf16']} variant(s) missing from the bf16 results")
    print(f"\n{'':26}{'quantized':>22}{'bf16':>22}")
    for key, label, pct in (("accuracy_original", "accuracy original", True),
                            ("accuracy_tool_applied", "accuracy tool-applied", True),
                            ("accuracy_gain", "accuracy gain", True),
                            ("mean_p_true_original", "P(True) original", False),
                            ("mean_p_true_tool_applied", "P(True) tool-applied", False),
                            ("mean_delta_p_true", "mean ΔP(True)", False),
                            ("frac_delta_p_true_ge_0.2", "share ΔP(True) >= 0.2", True),
                            ("frac_p_true_original_lt_0.8", "share P(True) orig < 0.8", True),
                            ("auroc_p_true_original", "AUROC P(True) original", False),
                            ("auroc_p_true_tool_applied", "AUROC P(True) tool", False)):
        signed = key in ("accuracy_gain", "mean_delta_p_true")
        print(f"{label:26}{f(v['quantized'][key], pct, signed):>22}{f(v['bf16'][key], pct, signed):>22}")
    for who in ("quantized", "bf16"):
        c = v[who]["confusion"]
        print(f"confusion {who:16}T->T {c['T->T']}  T->F {c['T->F']}  F->T {c['F->T']}  F->F {c['F->F']}  net {c['net']:+d}")
    a = v["agreement"]
    print("\nagreement quantized vs bf16:")
    for side in SIDES:
        s = a[side]
        print(f"  {side:14} same right/wrong {f(s['agree'], True)} (kappa {f(s['kappa'])}), "
              f"P(True) pearson {f(s['p_true_pearson'])} spearman {f(s['p_true_spearman'])} "
              f"mean |diff| {f(s['p_true_mean_abs_diff'])}")
    print(f"  ΔP(True)       pearson {f(a['delta_p_true']['pearson'])} spearman {f(a['delta_p_true']['spearman'])}; "
          f"same correctness change (gain/loss/none) {f(a['correctness_change_same'], True)}")

    if "questions_original_audio" in r:
        qa = r["questions_original_audio"]
        g = qa["agreement"]
        print(f"\noriginal audio, per question ({qa['n_questions']} questions):")
        print(f"  accuracy  quantized {f(qa['quantized']['accuracy'], True)}  bf16 {f(qa['bf16']['accuracy'], True)};  "
              f"P(True) quantized {f(qa['quantized']['mean_p_true'])}  bf16 {f(qa['bf16']['mean_p_true'])}")
        print(f"  AUROC     quantized {f(qa['quantized']['auroc_p_true'])}  bf16 {f(qa['bf16']['auroc_p_true'])};  "
              f"bf16 share P(True) < 0.8: {f(qa['bf16']['frac_p_true_lt_0.8'], True)}")
        print(f"  agreement same right/wrong {f(g['agree'], True)} (kappa {f(g['kappa'])}), "
              f"P(True) pearson {f(g['p_true_pearson'])} spearman {f(g['p_true_spearman'])}")
        w = qa["quantized_all_questions"]
        print(f"  (whole quantized base run: {w['n']} questions, accuracy {f(w['accuracy'], True)}, "
              f"mean P(True) {f(w['mean_p_true'])})")

    print(f"\n{'tool':22}{'n':>6}{'quant acc o->t':>18}{'bf16 acc o->t':>18}{'quant ΔP':>10}{'bf16 ΔP':>10}{'q net':>7}{'b net':>7}")
    for t, g in r["per_tool"].items():
        q, b = g["quantized"], g["bf16"]
        print(f"{t:22}{g['n']:>6}"
              f"{f(q['accuracy_original'], True) + '->' + f(q['accuracy_tool_applied'], True):>18}"
              f"{f(b['accuracy_original'], True) + '->' + f(b['accuracy_tool_applied'], True):>18}"
              f"{f(q['mean_delta_p_true'], signed=True):>10}{f(b['mean_delta_p_true'], signed=True):>10}"
              f"{q['confusion']['net']:>+7d}{b['confusion']['net']:>+7d}")
    print(f"\nper round:")
    for k, g in r["per_round"].items():
        q, b = g["quantized"], g["bf16"]
        print(f"  round {k}: n={g['n']}  quant acc {f(q['accuracy_original'], True)}->{f(q['accuracy_tool_applied'], True)}"
              f"  bf16 acc {f(b['accuracy_original'], True)}->{f(b['accuracy_tool_applied'], True)}"
              f"  quant ΔP {f(q['mean_delta_p_true'], signed=True)}  bf16 ΔP {f(b['mean_delta_p_true'], signed=True)}")
    print(f"\ncomparison: {out}")


if __name__ == "__main__":
    main()
