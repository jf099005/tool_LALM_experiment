"""Per-tool analysis of tool-use evaluation results.

`run_eval.aggregate` only reports benchmark-wide averages plus a small
per-tool parameter block restricted to steps that already named the right
tool. This module breaks everything down *by ground-truth tool*:

  * tool-selection accuracy / precision / recall / F1 per tool, and a
    confusion matrix (GT tool -> predicted tool at the aligned position,
    with `<no_call>` / `<no_gt>` for missing / extra calls),
  * execution success, stop reasons and audio-fidelity numbers per tool
    (split by whether the right tool was chosen),
  * a direct parameter-by-parameter comparison against ground truth for
    every aligned (GT call, predicted call) pair with matching tool names:
    abs / signed / relative error, RMSE, exact-match and tolerance rates for
    numeric (and HH:MM:SS.mmm timestamp) parameters; accuracy and a
    value-level confusion table for categorical ones; missing / extra
    parameter counts; and the raw (gt, pred) value pairs.

Alignment is positional (gt_calls[i] vs predicted_steps[i]), same assumption
as `run_eval.compute_step_param_metrics` -- unambiguous for the current
single-step benchmark, see the note there before trusting it on multi-step
chains.

Pure-Python / JSON only (no model, no audio), so it can also be re-run
offline on an existing results file, see `analyze_results.py`.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

NO_CALL = "<no_call>"  # GT step exists but the model emitted no call at that position
NO_GT = "<no_gt>"      # model emitted a call beyond the end of the GT chain

# Parameters that are pure plumbing (fresh ids chosen by the model) -- never scored.
IGNORED_PARAMS = {"output_audio_id"}
# Schema-documented defaults: an omitted value on either side is treated as this.
PARAM_DEFAULTS: Dict[Tuple[str, str], Any] = {("pitch_shift", "bins_per_octave"): 12}

_TIMESTAMP_RE = re.compile(r"^\s*\d+:\d{1,2}:\d{1,2}(\.\d+)?\s*$")
EXACT_EPS = 1e-6
TIMESTAMP_TOLERANCES_SEC = (0.05, 0.1, 0.5)
RELATIVE_TOLERANCES = (0.05, 0.1)


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------

def _timestamp_to_sec(value: Any) -> Optional[float]:
    if not isinstance(value, str) or not _TIMESTAMP_RE.match(value):
        return None
    hh, mm, ss = value.strip().split(":")
    try:
        return int(hh) * 3600 + int(mm) * 60 + float(ss)
    except ValueError:
        return None


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        try:
            f = float(value.strip())
            return f if math.isfinite(f) else None
        except ValueError:
            return None
    return None


def _param_kind(gt_value: Any) -> str:
    """Decide how a parameter is compared, from its ground-truth value."""
    if _timestamp_to_sec(gt_value) is not None:
        return "timestamp"
    if isinstance(gt_value, bool):
        return "categorical"
    if isinstance(gt_value, (int, float)):
        return "numeric"
    return "categorical"


def _numeric_value(kind: str, value: Any) -> Optional[float]:
    return _timestamp_to_sec(value) if kind == "timestamp" else _to_float(value)


def _cat_key(value: Any) -> str:
    return value if isinstance(value, str) else repr(value)


def _safe_mean(vals: List[float]) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None


def _rate(num: int, den: int) -> Optional[float]:
    return num / den if den else None


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align_calls(
    gt_calls: List[Dict[str, Any]], predicted_steps: List[Dict[str, Any]]
) -> List[Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]:
    """Positional alignment, padded with None on whichever side is shorter."""
    n = max(len(gt_calls), len(predicted_steps))
    return [
        (gt_calls[i] if i < len(gt_calls) else None, predicted_steps[i] if i < len(predicted_steps) else None)
        for i in range(n)
    ]


def _gt_calls_for(result: Dict[str, Any], gt_calls_by_id: Optional[Dict[str, List[Dict[str, Any]]]]):
    """GT calls (with parameters) for one result; None if only names are known."""
    if result.get("gt_calls") is not None:
        return result["gt_calls"]
    if gt_calls_by_id and result.get("id") in gt_calls_by_id:
        return gt_calls_by_id[result["id"]]
    return None


# ---------------------------------------------------------------------------
# Parameter comparison
# ---------------------------------------------------------------------------

class _ParamAccumulator:
    def __init__(self, kind: str):
        self.kind = kind
        self.n_gt = 0            # GT pairs that specify this parameter
        self.n_missing = 0       # ... but the prediction omitted it
        self.n_unparseable = 0   # ... prediction gave something not comparable
        self.n_extra = 0         # prediction gave it, GT didn't
        self.abs_err: List[float] = []
        self.signed_err: List[float] = []
        self.rel_err: List[float] = []
        self.n_exact = 0
        self.cat_confusion: Dict[str, Counter] = defaultdict(Counter)
        self.pairs: List[Dict[str, Any]] = []

    def add(self, sample_id: str, gt_value: Any, pred_value: Any, pred_present: bool) -> Optional[bool]:
        """Returns whether this parameter matched exactly (None if not scored)."""
        self.n_gt += 1
        self.pairs.append({"id": sample_id, "gt": gt_value, "pred": pred_value if pred_present else None})
        if not pred_present:
            self.n_missing += 1
            if self.kind == "categorical":
                self.cat_confusion[_cat_key(gt_value)]["<missing>"] += 1
            return False
        if self.kind == "categorical":
            match = gt_value == pred_value
            self.cat_confusion[_cat_key(gt_value)][_cat_key(pred_value)] += 1
            self.n_exact += int(match)
            return match
        g, p = _numeric_value(self.kind, gt_value), _numeric_value(self.kind, pred_value)
        if g is None or p is None:
            self.n_unparseable += 1
            return False
        d = p - g
        self.signed_err.append(d)
        self.abs_err.append(abs(d))
        if g != 0:
            self.rel_err.append(abs(d) / abs(g))
        match = abs(d) <= EXACT_EPS
        self.n_exact += int(match)
        return match

    def summary(self, include_pairs: bool) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "kind": self.kind,
            "n_gt": self.n_gt,
            "n_missing_in_pred": self.n_missing,
            "n_extra_in_pred": self.n_extra,
            "exact_match_rate": _rate(self.n_exact, self.n_gt),
        }
        if self.kind == "categorical":
            out["accuracy"] = out["exact_match_rate"]
            out["value_confusion"] = {g: dict(c) for g, c in sorted(self.cat_confusion.items())}
        else:
            n = len(self.abs_err)
            out["n_unparseable_in_pred"] = self.n_unparseable
            out["n_compared"] = n
            out["mae"] = _safe_mean(self.abs_err)
            out["median_abs_error"] = statistics.median(self.abs_err) if n else None
            out["max_abs_error"] = max(self.abs_err) if n else None
            out["rmse"] = math.sqrt(sum(e * e for e in self.abs_err) / n) if n else None
            out["mean_signed_error"] = _safe_mean(self.signed_err)  # pred - gt: >0 means over-estimates
            out["std_signed_error"] = statistics.pstdev(self.signed_err) if n else None
            if self.kind == "timestamp":
                out["unit"] = "sec"
                for tol in TIMESTAMP_TOLERANCES_SEC:
                    out[f"within_{tol}s_rate"] = _rate(sum(e <= tol + EXACT_EPS for e in self.abs_err), n)
            else:
                out["mean_relative_error"] = _safe_mean(self.rel_err)
                for tol in RELATIVE_TOLERANCES:
                    out[f"within_{int(tol * 100)}pct_rate"] = (
                        _rate(sum(e <= tol + EXACT_EPS for e in self.rel_err), len(self.rel_err))
                    )
        if include_pairs:
            out["pairs"] = self.pairs
        return out


def _compare_params(
    tool: str, gt_params: Dict[str, Any], pred_params: Dict[str, Any],
    accs: Dict[str, _ParamAccumulator], sample_id: str,
) -> bool:
    """Feed one matched pair into `accs`; returns True if every scored GT param matched exactly."""
    all_exact = True
    for name, gt_value in gt_params.items():
        if name in IGNORED_PARAMS:
            continue
        pred_present = name in pred_params
        pred_value = pred_params.get(name)
        if not pred_present and (tool, name) in PARAM_DEFAULTS:
            pred_present, pred_value = True, PARAM_DEFAULTS[(tool, name)]
        acc = accs.get(name)
        if acc is None:
            acc = accs[name] = _ParamAccumulator(_param_kind(gt_value))
        if not acc.add(sample_id, gt_value, pred_value, pred_present):
            all_exact = False
    for name, pred_value in pred_params.items():
        if name in IGNORED_PARAMS or name in gt_params:
            continue
        if (tool, name) in PARAM_DEFAULTS and pred_value == PARAM_DEFAULTS[(tool, name)]:
            continue
        acc = accs.get(name)
        if acc is None:
            acc = accs[name] = _ParamAccumulator(_param_kind(pred_value))
        acc.n_extra += 1
    return all_exact


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_per_tool(
    results: List[Dict[str, Any]],
    gt_calls_by_id: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    include_pairs: bool = False,
) -> Dict[str, Any]:
    """Per-tool breakdown of `run_sample`-style result dicts.

    `gt_calls_by_id` backfills GT parameters for results produced before
    `gt_calls` was stored per sample (otherwise only tool names are known and
    the parameter comparison is skipped for those samples).
    `include_pairs` adds every raw (gt, pred) parameter value pair.
    """
    confusion: Dict[str, Counter] = defaultdict(Counter)
    exec_ok: Counter = Counter()        # correct-tool calls that executed successfully
    stop_reasons: Dict[str, Counter] = defaultdict(Counter)
    sample_stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    param_accs: Dict[str, Dict[str, _ParamAccumulator]] = defaultdict(dict)
    param_pairs_n: Counter = Counter()
    param_all_exact: Counter = Counter()
    n_without_gt_params = 0

    for r in results:
        gt_calls = _gt_calls_for(r, gt_calls_by_id)
        if gt_calls is None:
            n_without_gt_params += 1
            gt_calls_for_align = [{"tool_name": t} for t in r.get("gt_tool_sequence", [])]
        else:
            gt_calls_for_align = gt_calls

        for gt_call, pred_step in align_calls(gt_calls_for_align, r.get("predicted_steps", [])):
            gt_tool = gt_call["tool_name"] if gt_call else NO_GT
            pred_tool = pred_step["tool_name"] if pred_step else NO_CALL
            confusion[gt_tool][pred_tool] += 1
            if gt_call and pred_step and gt_tool == pred_tool:
                exec_ok[gt_tool] += int(bool(pred_step.get("success")))
                if gt_calls is not None:
                    param_pairs_n[gt_tool] += 1
                    all_exact = _compare_params(
                        gt_tool, gt_call.get("parameters") or {},
                        pred_step.get("parameters") if isinstance(pred_step.get("parameters"), dict) else {},
                        param_accs[gt_tool], r.get("id"),
                    )
                    param_all_exact[gt_tool] += int(all_exact)

        # Sample-level stats, attributed to every distinct tool in the GT chain.
        gt_seq = r.get("gt_tool_sequence", [])
        am, bm = r.get("audio_metrics") or {}, r.get("baseline_metrics") or {}
        pred_succ = set(r.get("predicted_successful_sequence", []))
        for tool in set(gt_seq):
            s = sample_stats[tool]
            stop_reasons[tool][r.get("stop_reason")] += 1
            s["n_samples"].append(1)
            s["exact_sequence_match"].append(float(bool(r.get("exact_sequence_match"))))
            s["tool_used_successfully"].append(float(tool in pred_succ))
            s["num_predicted_calls"].append(float(r.get("num_predicted_calls", 0)))
            s["num_extra_calls"].append(float(r.get("num_predicted_calls", 0) - len(gt_seq)))
            cs, bcs = am.get("closeness_score"), bm.get("closeness_score")
            if cs is not None:
                s["closeness"].append(cs)
                (s["closeness_correct"] if r.get("exact_sequence_match") else s["closeness_wrong"]).append(cs)
                if bcs is not None:
                    s["closeness_gain"].append(cs - bcs)
            for key in ("log_mel_cosine", "mfcc_cosine", "log_mel_l1"):
                if am.get(key) is not None:
                    s[key].append(am[key])
            if am.get("duration_ratio") and am["duration_ratio"] > 0:
                s["abs_log_duration_ratio"].append(abs(math.log(am["duration_ratio"])))

    gt_tools = sorted(t for t in confusion if t != NO_GT)
    pred_labels = sorted({p for row in confusion.values() for p in row if p != NO_CALL})
    col_totals: Counter = Counter()
    for row in confusion.values():
        for p, c in row.items():
            col_totals[p] += c

    per_tool: Dict[str, Any] = {}
    for tool in gt_tools:
        row = confusion[tool]
        support = sum(row.values())
        tp = row.get(tool, 0)
        predicted_as_tool = col_totals.get(tool, 0)
        precision = _rate(tp, predicted_as_tool)
        recall = _rate(tp, support)
        f1 = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0
        confused = [(p, c) for p, c in row.most_common() if p != tool]
        s = sample_stats.get(tool, {})
        entry: Dict[str, Any] = {
            "support_calls": support,
            "num_samples": len(s.get("n_samples", [])),
            "tool_selection_accuracy": recall,  # aligned call == this tool (row-normalized diagonal)
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "times_predicted": predicted_as_tool,
            "no_call_rate": _rate(row.get(NO_CALL, 0), support),
            "execution_success_rate_when_correct": _rate(exec_ok.get(tool, 0), tp),
            "most_confused_with": [{"pred_tool": p, "count": c, "rate": c / support} for p, c in confused[:3]],
            "exact_sequence_match_rate": _safe_mean(s.get("exact_sequence_match", [])),
            "tool_used_successfully_rate": _safe_mean(s.get("tool_used_successfully", [])),
            "mean_num_predicted_calls": _safe_mean(s.get("num_predicted_calls", [])),
            "mean_num_extra_calls": _safe_mean(s.get("num_extra_calls", [])),
            "stop_reason_counts": dict(stop_reasons.get(tool, {})),
            "audio": {
                "mean_closeness_score": _safe_mean(s.get("closeness", [])),
                "mean_closeness_gain_over_baseline": _safe_mean(s.get("closeness_gain", [])),
                "mean_closeness_when_exact_match": _safe_mean(s.get("closeness_correct", [])),
                "mean_closeness_when_wrong": _safe_mean(s.get("closeness_wrong", [])),
                "mean_log_mel_cosine": _safe_mean(s.get("log_mel_cosine", [])),
                "mean_mfcc_cosine": _safe_mean(s.get("mfcc_cosine", [])),
                "mean_log_mel_l1": _safe_mean(s.get("log_mel_l1", [])),
                "mean_abs_log_duration_ratio": _safe_mean(s.get("abs_log_duration_ratio", [])),
            },
        }
        if param_pairs_n.get(tool):
            entry["parameters"] = {
                "num_matched_calls_compared": param_pairs_n[tool],
                "all_params_exact_rate": param_all_exact[tool] / param_pairs_n[tool],
                "by_param": {
                    name: acc.summary(include_pairs) for name, acc in sorted(param_accs[tool].items())
                },
            }
        per_tool[tool] = entry

    # Confusion matrix in both count and row-normalized form.
    row_labels = gt_tools + ([NO_GT] if NO_GT in confusion else [])
    col_labels = sorted(set(pred_labels) | set(gt_tools)) + [NO_CALL]
    matrix = [[confusion[g].get(p, 0) for p in col_labels] for g in row_labels]
    matrix_norm = [[(c / sum(row)) if sum(row) else 0.0 for c in row] for row in matrix]

    total_calls = sum(sum(confusion[t].values()) for t in gt_tools)
    total_correct = sum(confusion[t].get(t, 0) for t in gt_tools)
    macro = {
        k: _safe_mean([v[k] for v in per_tool.values() if v[k] is not None])
        for k in ("precision", "recall", "f1", "tool_selection_accuracy")
    }

    return {
        "overall_call_level_tool_accuracy": _rate(total_correct, total_calls),
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "macro_f1": macro["f1"],
        "macro_tool_selection_accuracy": macro["tool_selection_accuracy"],
        "num_extra_calls_beyond_gt": sum(confusion.get(NO_GT, {}).values()),
        "num_samples_without_gt_params": n_without_gt_params,
        "confusion_matrix": {
            "rows_gt": row_labels,
            "cols_pred": col_labels,
            "counts": matrix,
            "row_normalized": matrix_norm,
        },
        "per_tool": per_tool,
    }


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------

def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _table(header: List[str], rows: List[List[str]]) -> str:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    line = lambda r: "  ".join(str(x).rjust(w) if i else str(x).ljust(w) for i, (x, w) in enumerate(zip(r, widths)))
    return "\n".join([line(header), line(["-" * w for w in widths])] + [line(r) for r in rows])


def format_tool_analysis(analysis: Dict[str, Any]) -> str:
    if not analysis:
        return "(no results)"
    out: List[str] = []
    out.append("=" * 80)
    out.append("PER-TOOL ANALYSIS")
    out.append("=" * 80)
    out.append(
        f"call-level tool accuracy: {_fmt(analysis['overall_call_level_tool_accuracy'])}   "
        f"macro P/R/F1: {_fmt(analysis['macro_precision'])} / {_fmt(analysis['macro_recall'])} / "
        f"{_fmt(analysis['macro_f1'])}   extra calls beyond GT: {analysis['num_extra_calls_beyond_gt']}"
    )
    if analysis.get("num_samples_without_gt_params"):
        out.append(f"(parameter comparison skipped for {analysis['num_samples_without_gt_params']} samples "
                   f"with no GT parameters available)")

    out.append("\n[1] Tool selection / execution / audio, per GT tool")
    per_tool = analysis["per_tool"]
    rows = []
    for tool, e in per_tool.items():
        rows.append([
            tool, str(e["support_calls"]), _fmt(e["tool_selection_accuracy"]), _fmt(e["precision"]),
            _fmt(e["f1"]), _fmt(e["no_call_rate"]), _fmt(e["execution_success_rate_when_correct"]),
            _fmt(e["exact_sequence_match_rate"]), _fmt(e["audio"]["mean_closeness_score"]),
            _fmt(e["audio"]["mean_closeness_gain_over_baseline"]),
            _fmt(e.get("parameters", {}).get("all_params_exact_rate")),
        ])
    out.append(_table(
        ["gt_tool", "n", "acc", "prec", "f1", "no_call", "exec_ok", "seq_EM", "closeness", "gain", "params_EM"],
        rows,
    ))

    cm = analysis["confusion_matrix"]
    out.append("\n[2] Confusion matrix (rows = GT tool, cols = predicted tool at aligned step; counts)")
    abbrev = {c: f"c{i}" for i, c in enumerate(cm["cols_pred"])}
    out.append("    columns: " + ", ".join(f"{abbrev[c]}={c}" for c in cm["cols_pred"]))
    out.append(_table(
        ["gt \\ pred"] + [abbrev[c] for c in cm["cols_pred"]],
        [[g] + [str(c) if c else "." for c in row] for g, row in zip(cm["rows_gt"], cm["counts"])],
    ))
    for tool, e in per_tool.items():
        if e["most_confused_with"]:
            out.append(f"    {tool} -> " + ", ".join(
                f"{m['pred_tool']} ({m['count']}, {m['rate']:.0%})" for m in e["most_confused_with"]))

    out.append("\n[3] Parameter comparison vs GT (only calls where the predicted tool matches GT)")
    for tool, e in per_tool.items():
        p = e.get("parameters")
        if not p:
            continue
        out.append(f"\n  {tool}  (matched calls: {p['num_matched_calls_compared']}, "
                   f"all-params exact: {_fmt(p['all_params_exact_rate'])})")
        num_rows, cat_rows = [], []
        for name, s in p["by_param"].items():
            if s["kind"] == "categorical":
                conf = "; ".join(
                    f"{g}->" + ",".join(f"{pv}:{c}" for pv, c in sorted(row.items(), key=lambda x: -x[1]))
                    for g, row in s["value_confusion"].items()
                )
                cat_rows.append([name, str(s["n_gt"]), _fmt(s["accuracy"]), str(s["n_missing_in_pred"]),
                                 str(s["n_extra_in_pred"]), conf])
            else:
                tol_key = "within_0.1s_rate" if s["kind"] == "timestamp" else "within_10pct_rate"
                num_rows.append([
                    name + (" (s)" if s["kind"] == "timestamp" else ""), str(s["n_gt"]), str(s["n_compared"]),
                    _fmt(s["exact_match_rate"]), _fmt(s["mae"], 4), _fmt(s["median_abs_error"], 4),
                    _fmt(s["rmse"], 4), _fmt(s["max_abs_error"], 4), _fmt(s["mean_signed_error"], 4),
                    _fmt(s.get(tol_key)), str(s["n_missing_in_pred"]), str(s["n_extra_in_pred"]),
                ])
        if num_rows:
            out.append(_table(
                ["  numeric", "n", "cmp", "exact", "MAE", "medAE", "RMSE", "maxAE", "bias", "tol*", "miss", "extra"],
                [["  " + r[0]] + r[1:] for r in num_rows],
            ))
            out.append("    (bias = mean(pred - gt); tol* = within 0.1s for timestamps, within 10% otherwise)")
        if cat_rows:
            out.append(_table(["  categorical", "n", "acc", "miss", "extra", "gt->pred:count"],
                              [["  " + r[0]] + r[1:] for r in cat_rows]))
    out.append("=" * 80)
    return "\n".join(out)
