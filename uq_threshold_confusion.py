#!/usr/bin/env python3
"""Filter paired UQ results by a metric threshold and plot the original -> tool-applied
correctness confusion matrix (T->T, T->F, F->T, F->F).

Inputs are the two result files the paired UQ run writes (qwen25_with_tool_chain_evaluation.py
with --paired_subset_path): one for the original audio, one for the tool-applied audio of the
same items. Either the finished `.json` or the in-progress `.partial.jsonl` works, so a run
can be analysed while it is still going. Items are paired by `id`.

Filtering: `--on` picks the rule (metric is >= --threshold; <= with --below):
    increase      the tool raised the metric AND the tool-applied value >= threshold  (default)
                  (with --below: the tool lowered it AND the tool-applied value <= threshold)
    original      the original-audio value
    tool_applied  the tool-applied value
    both          both values pass
    either        at least one passes
    delta         tool_applied - original  (change caused by the tool)
    abs_delta     |tool_applied - original|
Pairs where a needed value is missing (e.g. P(True) could not be scored) are left out and
counted.

Examples:
    python uq_threshold_confusion.py --threshold 0.8                       # P(True) rose with the tool, and >= 0.8 after it
    python uq_threshold_confusion.py --metric discrete_semantic_entropy --threshold 1.0 --on tool_applied
    python uq_threshold_confusion.py --metric predictive_entropy --on abs_delta --threshold 10 --save-items big_change.json

With no --original/--tool-applied, the files of the local 50k AudioMCQ run are used
(UQ_RESULT_DIR from local_env.sh, run tag --run-tag).

Export (--export-dir DIR): writes the kept items as a dataset in the same layout
gen_2nd_2_data/build_dataset.py produces, so it can be fed straight to the evaluation
scripts (e.g. qwen3_uq_benchmark.py):
    DIR/original_subset.json       dcase_subset.json-shaped entries (as in the source dataset)
    DIR/tool_applied_subset.json   same items, tool-applied audio
    DIR/manifest.json              build_dataset manifest records of the kept items
    DIR/filter.json                the filter, its counts, and each kept item's metric values
    DIR/original/<id>/original.wav, DIR/tool_applied/<id>/tool_applied.wav
                                   copies of the audio (skip with --no-copy-audio), so DIR is
                                   self-contained and can be moved to another machine.
The source dataset dir defaults to the local run's (--subset-dir to override).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
from typing import Dict, Optional

METRICS = [
    "p_true",
    "predictive_entropy",
    "length_normalized_entropy",
    "discrete_semantic_entropy",
    "semantic_entropy",
]
ON_CHOICES = ["increase", "original", "tool_applied", "both", "either", "delta", "abs_delta"]
CELLS = [("T", "T"), ("T", "F"), ("F", "T"), ("F", "F")]

DEFAULT_RUN_TAG = "Qwen2.5-Omni-3B_audiomcq_strong_50k_ml40"
RESULT_SUFFIX = "_uq_results_no_tool_chain_results.json"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(path: Path) -> Dict[str, dict]:
    """id -> result entry, from a finished .json, a .partial.jsonl, or both (a run that
    was resumed after finishing part of its items has both)."""
    path = Path(path)
    candidates = [path]
    if path.suffix == ".json":
        candidates.append(Path(f"{path}.partial.jsonl"))
    found = [p for p in candidates if p.exists()]
    if not found:
        raise SystemExit(f"Not found: {path} (nor {path}.partial.jsonl)")
    by_id: Dict[str, dict] = {}
    for p in found:
        if p.name.endswith(".jsonl"):
            with p.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:  # line cut off by an interrupt
                        continue
                    by_id[entry["id"]] = entry
        else:
            for entry in json.loads(p.read_text(encoding="utf-8")):
                by_id[entry["id"]] = entry
    return by_id


def default_paths(run_tag: str):
    result_dir = os.environ.get("UQ_RESULT_DIR")
    if not result_dir:
        raise SystemExit("Pass --original and --tool-applied (or `source local_env.sh` for the defaults).")
    d = Path(result_dir)
    return (d / f"{run_tag}_original{RESULT_SUFFIX}", d / f"{run_tag}_tool_applied{RESULT_SUFFIX}")


def default_manifest(run_tag: str) -> Optional[Path]:
    data_root = os.environ.get("DATA_ROOT")
    if not data_root or not run_tag.startswith("Qwen2.5-Omni-3B_"):
        return None
    tag = run_tag[len("Qwen2.5-Omni-3B_"):]  # audiomcq_strong_50k_ml40
    base, _, suffix = tag.rpartition("_ml")
    p = Path(data_root) / "datasets" / f"{base}_tool_applied_ml{suffix}" / "manifest.json"
    return p if suffix and p.exists() else None


def default_dataset_dir(run_tag: str) -> Optional[Path]:
    manifest = default_manifest(run_tag)
    return manifest.parent if manifest else None


def load_tools(manifest: Optional[Path]) -> Dict[str, str]:
    """id -> applied tool chain as text, e.g. 'remove_target' or 'clipping+denoise'."""
    if manifest is None:
        return {}
    return {
        rec["id"]: "+".join(step["tool"] for step in rec["tool_chain"]) or "(none)"
        for rec in json.loads(Path(manifest).read_text(encoding="utf-8"))
    }


# ---------------------------------------------------------------------------
# Filtering + counting
# ---------------------------------------------------------------------------

def metric_value(entry: dict, metric: str) -> Optional[float]:
    value = (entry.get("uncertainty") or {}).get(metric)
    return None if value is None else float(value)


def select(pairs, metric: str, threshold: float, on: str, below: bool):
    """Returns (kept, n_missing). Each kept item is a dict with both entries and values."""
    passes = (lambda v: v <= threshold) if below else (lambda v: v >= threshold)
    kept, missing = [], 0
    for item_id, orig, tool in pairs:
        vo, vt = metric_value(orig, metric), metric_value(tool, metric)
        if on == "original":
            needed, score = [vo], vo
        elif on == "tool_applied":
            needed, score = [vt], vt
        else:
            needed, score = [vo, vt], None
        if any(v is None for v in needed):
            missing += 1
            continue
        if on == "increase":
            ok = (vt < vo if below else vt > vo) and passes(vt)
        elif on == "delta":
            ok = passes(vt - vo)
        elif on == "abs_delta":
            ok = passes(abs(vt - vo))
        elif on == "both":
            ok = passes(vo) and passes(vt)
        elif on == "either":
            ok = passes(vo) or passes(vt)
        else:
            ok = passes(score)
        if ok:
            kept.append({"id": item_id, "original": orig, "tool_applied": tool, "v_orig": vo, "v_tool": vt})
    return kept, missing


def confusion(items) -> collections.Counter:
    counts = collections.Counter()
    for it in items:
        a = "T" if it["original"].get("correct") else "F"
        b = "T" if it["tool_applied"].get("correct") else "F"
        counts[(a, b)] += 1
    return counts


def accuracy(items, side: str) -> float:
    return sum(bool(it[side].get("correct")) for it in items) / len(items) if items else float("nan")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def export_items(kept, dataset_dir: Path, export_dir: Path, copy_audio: bool, filter_info: dict) -> None:
    """Write the kept items as a build_dataset-style dataset (see module docstring)."""
    import shutil

    dataset_dir, export_dir = Path(dataset_dir), Path(export_dir).expanduser().resolve()
    keep = {it["id"]: it for it in kept}

    def subset_entries(name):
        entries = json.loads((dataset_dir / name).read_text(encoding="utf-8"))
        by_id = {e["id"]: e for e in entries}
        missing = [i for i in keep if i not in by_id]
        if missing:
            raise SystemExit(f"{len(missing)} kept id(s) not in {dataset_dir / name}, e.g. {missing[:3]}")
        return by_id

    original = subset_entries("original_subset.json")
    tool_applied = subset_entries("tool_applied_subset.json")
    manifest_path = dataset_dir / "manifest.json"
    manifest = {r["id"]: r for r in json.loads(manifest_path.read_text(encoding="utf-8"))} if manifest_path.exists() else {}

    export_dir.mkdir(parents=True, exist_ok=True)
    out_orig, out_tool, out_manifest = [], [], []
    for item_id in keep:  # kept order (= result order)
        o, t = dict(original[item_id]), dict(tool_applied[item_id])
        rec = dict(manifest[item_id]) if item_id in manifest else None
        if copy_audio:
            for entry, side, fname in ((o, "original", "original.wav"), (t, "tool_applied", "tool_applied.wav")):
                src = Path(entry.get("audio_path") or entry["audio_url"])
                dst = export_dir / side / item_id / fname
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                    shutil.copy2(src, dst)
                entry["audio_path"] = entry["audio_url"] = str(dst)
            if rec is not None:
                rec["original_audio"], rec["tool_applied_audio"] = o["audio_path"], t["audio_path"]
                if rec.get("tool_chain"):
                    rec["tool_chain"] = [dict(step) for step in rec["tool_chain"]]
                    rec["tool_chain"][-1]["output_path"] = t["audio_path"]
        out_orig.append(o)
        out_tool.append(t)
        if rec is not None:
            out_manifest.append(rec)

    def write(name, data):
        (export_dir / name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    write("original_subset.json", out_orig)
    write("tool_applied_subset.json", out_tool)
    if out_manifest:
        write("manifest.json", out_manifest)
    metric = filter_info["metric"]
    write("filter.json", {
        **filter_info,
        "source_dataset_dir": str(dataset_dir),
        # Lets a consumer on another machine re-root the absolute audio paths above.
        "export_dir": str(export_dir),
        "audio_copied": copy_audio,
        "items": [{
            "id": it["id"],
            f"{metric}_original": it["v_orig"],
            f"{metric}_tool_applied": it["v_tool"],
            "correct_original": bool(it["original"].get("correct")),
            "correct_tool_applied": bool(it["tool_applied"].get("correct")),
        } for it in kept],
    })
    print(f"exported {len(out_orig):,} item(s) to {export_dir}" + (" (with audio)" if copy_audio else ""))


def describe_filter(metric: str, threshold: float, on: str, below: bool) -> str:
    op = "<=" if below else ">="
    if on == "increase":
        change = "decreased" if below else "increased"
        return f"{metric} {change} by the tool and tool-applied {op} {threshold:g}"
    target = {
        "original": f"{metric} (original)",
        "tool_applied": f"{metric} (tool-applied)",
        "both": f"{metric} (original and tool-applied)",
        "either": f"{metric} (original or tool-applied)",
        "delta": f"Δ{metric} (tool − original)",
        "abs_delta": f"|Δ{metric}|",
    }[on]
    return f"{target} {op} {threshold:g}"


def print_matrix(counts: collections.Counter, n: int, indent: str = "") -> None:
    def cell(a, b):
        c = counts[(a, b)]
        return f"{c:>7} ({c / n:6.1%})" if n else f"{c:>7}"
    print(f"{indent}{'':16}{'tool-applied T':>17}{'tool-applied F':>17}")
    print(f"{indent}{'original T':16}{cell('T', 'T'):>17}{cell('T', 'F'):>17}")
    print(f"{indent}{'original F':16}{cell('F', 'T'):>17}{cell('F', 'F'):>17}")


def plot_matrix(counts: collections.Counter, n: int, title: str, subtitle: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Sequential encoding (magnitude): one hue, light -> dark (reference blue ramp 100 -> 700).
    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("seq_blue", ramp)
    surface, ink, ink2 = "#fcfcfb", "#0b0b0b", "#52514e"

    grid = [[counts[("T", "T")], counts[("T", "F")]], [counts[("F", "T")], counts[("F", "F")]]]
    vmax = max(max(row) for row in grid) or 1
    names = [["T→T", "T→F"], ["F→T", "F→F"]]

    fig, ax = plt.subplots(figsize=(5.6, 5.2), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    gap = 0.02  # surface gap between cells
    for r in range(2):
        for c in range(2):
            v = grid[r][c]
            frac = v / vmax
            color = cmap(frac)
            ax.add_patch(plt.Rectangle((c + gap, 1 - r + gap), 1 - 2 * gap, 1 - 2 * gap,
                                       facecolor=color, edgecolor="none", joinstyle="round"))
            # Text in ink tokens; flips to the light surface ink on the dark half of the ramp.
            txt = surface if frac > 0.55 else ink
            sub = surface if frac > 0.55 else ink2
            ax.text(c + 0.5, 1 - r + 0.60, f"{v:,}", ha="center", va="center", fontsize=20,
                    fontweight="bold", color=txt)
            pct = f"{v / n:.1%}" if n else "–"
            ax.text(c + 0.5, 1 - r + 0.36, f"{names[r][c]}  ·  {pct}", ha="center", va="center",
                    fontsize=10, color=sub)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")
    ax.set_xticks([0.5, 1.5], ["correct (T)", "wrong (F)"], color=ink2, fontsize=10)
    ax.set_yticks([1.5, 0.5], ["correct (T)", "wrong (F)"], color=ink2, fontsize=10, rotation=90, va="center")
    ax.set_xlabel("tool-applied audio", color=ink2, fontsize=10, labelpad=8)
    ax.set_ylabel("original audio", color=ink2, fontsize=10, labelpad=8)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle(title, x=0.06, ha="left", fontsize=12, fontweight="bold", color=ink)
    fig.text(0.06, 0.935, subtitle, ha="left", va="top", fontsize=9, color=ink2, linespacing=1.5)
    fig.subplots_adjust(top=0.76, left=0.14, right=0.96, bottom=0.04)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=surface)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--metric", choices=METRICS, default="p_true")
    ap.add_argument("--on", choices=ON_CHOICES, default="increase", help="filter rule (see above)")
    ap.add_argument("--below", action="store_true", help="keep metric <= threshold instead of >=")
    ap.add_argument("--original", type=Path, help="original-audio results (.json or .partial.jsonl)")
    ap.add_argument("--tool-applied", type=Path, help="tool-applied results (.json or .partial.jsonl)")
    ap.add_argument("--run-tag", default=DEFAULT_RUN_TAG, help="result-file prefix used for the default paths")
    ap.add_argument("--manifest", type=Path, help="dataset manifest.json, for the per-tool breakdown "
                                                  "(default: the local run's manifest if found)")
    ap.add_argument("--out", type=Path, help="figure path (default: next to the results)")
    ap.add_argument("--export-dir", type=Path, help="write the kept items as a build_dataset-style dataset (see above)")
    ap.add_argument("--subset-dir", type=Path, help="source dataset dir with original_subset.json / tool_applied_subset.json "
                                                    "(default: the local run's dataset dir)")
    ap.add_argument("--no-copy-audio", action="store_true", help="with --export-dir: keep the source audio paths instead of copying")
    ap.add_argument("--save-items", type=Path, help="write the filtered items (id, tool, values, correctness) as JSON")
    args = ap.parse_args()

    if (args.original is None) != (args.tool_applied is None):
        raise SystemExit("Give both --original and --tool-applied, or neither.")
    orig_path, tool_path = (args.original, args.tool_applied) if args.original else default_paths(args.run_tag)
    original, tool_applied = load_results(orig_path), load_results(tool_path)
    common = [i for i in original if i in tool_applied]
    pairs = [(i, original[i], tool_applied[i]) for i in common]
    if not pairs:
        raise SystemExit("No items with results on both sides yet.")

    manifest = args.manifest or default_manifest(args.run_tag)
    tools = load_tools(manifest)

    kept, missing = select(pairs, args.metric, args.threshold, args.on, args.below)
    counts = confusion(kept)
    n = len(kept)
    rule = describe_filter(args.metric, args.threshold, args.on, args.below)

    all_items = [{"id": i, "original": o, "tool_applied": t} for i, o, t in pairs]
    print(f"paired items      : {len(pairs):,}  (original {len(original):,}, tool-applied {len(tool_applied):,})")
    print(f"filter            : {rule}")
    print(f"kept              : {n:,} ({n / len(pairs):.1%} of paired)   missing {args.metric}: {missing:,}")
    print(f"accuracy (kept)   : original {accuracy(kept, 'original'):.2%} -> tool-applied {accuracy(kept, 'tool_applied'):.2%}")
    print(f"accuracy (all)    : original {accuracy(all_items, 'original'):.2%} -> tool-applied {accuracy(all_items, 'tool_applied'):.2%}")
    if n:
        flips = counts[("T", "F")] + counts[("F", "T")]
        print(f"answer flips      : {flips:,} ({flips / n:.1%}); net {counts[('F', 'T')] - counts[('T', 'F')]:+,} correct")
    print()
    print_matrix(counts, n)

    if tools and kept:
        print("\nper tool (kept items):")
        print(f"  {'tool':24}{'n':>7}{'T->T':>8}{'T->F':>8}{'F->T':>8}{'F->F':>8}{'acc orig':>10}{'acc tool':>10}")
        by_tool = collections.defaultdict(list)
        for it in kept:
            by_tool[tools.get(it["id"], "(unknown)")].append(it)
        for tool, items in sorted(by_tool.items(), key=lambda kv: -len(kv[1])):
            c = confusion(items)
            print(f"  {tool:24}{len(items):>7}" + "".join(f"{c[k]:>8}" for k in CELLS)
                  + f"{accuracy(items, 'original'):>10.1%}{accuracy(items, 'tool_applied'):>10.1%}")

    out = args.out or (Path(orig_path).parent / "uq_confusion"
                       / f"{args.run_tag}_{args.metric}_{args.on}_{'le' if args.below else 'ge'}{args.threshold:g}.png")
    title = "Original → tool-applied correctness"
    subtitle = (f"Filter: {rule}\n{n:,} of {len(pairs):,} paired items kept · accuracy "
                f"{accuracy(kept, 'original'):.1%} → {accuracy(kept, 'tool_applied'):.1%}")
    plot_matrix(counts, n, title, subtitle, out)
    print(f"\nfigure: {out}")

    if args.export_dir:
        dataset_dir = args.subset_dir or (args.manifest.parent if args.manifest else default_dataset_dir(args.run_tag))
        if dataset_dir is None:
            raise SystemExit("--export-dir needs --subset-dir (the dataset dir with original_subset.json etc.)")
        export_items(kept, dataset_dir, args.export_dir, not args.no_copy_audio, {
            "filter": rule,
            "metric": args.metric,
            "threshold": args.threshold,
            "on": args.on,
            "below": args.below,
            "uq_results": {"original": str(orig_path), "tool_applied": str(tool_path)},
            "n_paired": len(pairs),
            "n_kept": n,
            "confusion": {f"{a}->{b}": counts[(a, b)] for a, b in CELLS},
            "accuracy_original": accuracy(kept, "original"),
            "accuracy_tool_applied": accuracy(kept, "tool_applied"),
        })

    if args.save_items:
        rows = [{
            "id": it["id"],
            "tool": tools.get(it["id"]),
            f"{args.metric}_original": it["v_orig"],
            f"{args.metric}_tool_applied": it["v_tool"],
            "correct_original": bool(it["original"].get("correct")),
            "correct_tool_applied": bool(it["tool_applied"].get("correct")),
            "answer": it["original"].get("answer"),
            "prediction_original": it["original"].get("prediction"),
            "prediction_tool_applied": it["tool_applied"].get("prediction"),
        } for it in kept]
        args.save_items.parent.mkdir(parents=True, exist_ok=True)
        args.save_items.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"items : {args.save_items} ({len(rows):,})")


if __name__ == "__main__":
    main()
