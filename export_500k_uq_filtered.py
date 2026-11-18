#!/usr/bin/env python3
"""Export the 500k items whose UQ metric changed enough with the tool, as one dataset dir in
the same layout as the 50k export (uq_threshold_confusion.py --export-dir), so
qwen3_uq_benchmark.py / run_qwen3_uq_benchmark.sh can run the same experiment on it.

The 500k run differs from the 50k one in two ways, both handled here:
  * the original-audio UQ was computed once per question (pool file, id = base_id) and shared
    by every tool variant (id = <base_id>__v<k>), so pairs are made through manifest base_id;
  * the items live in 10 slice dirs (each with its own subsets/manifest/tool audio).

    python export_500k_uq_filtered.py --threshold 0.2                     # ΔP(True) >= 0.2
    python export_500k_uq_filtered.py --metric p_true --on delta --threshold 0.1

Output: $DATA_ROOT/exports/audiomcq500k_<metric>_<on>_ge<threshold>/ (or --export-dir) with
original_subset.json, tool_applied_subset.json, manifest.json, filter.json, confusion.png and
original/<id>/original.wav + tool_applied/<id>/tool_applied.wav copies. Each item keeps its
variant id on both sides (as in the 50k export), so a question with several kept variants
appears several times in original_subset.json -- the Qwen3 run scores each copy separately.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

from uq_threshold_confusion import (CELLS, METRICS, ON_CHOICES, accuracy, confusion, describe_filter,
                                    export_items, load_results, plot_matrix, print_matrix, select)

MODEL_TAG = "Qwen2.5-Omni-3B-fp8"
SUFFIX = "_uq_results_no_tool_chain_results.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--metric", choices=METRICS, default="p_true")
    ap.add_argument("--on", choices=ON_CHOICES, default="delta")
    ap.add_argument("--below", action="store_true")
    ap.add_argument("--slices-dir", type=Path, help="default: $DATA_ROOT/datasets/audiomcq500k")
    ap.add_argument("--result-dir", type=Path, help="default: $UQ_RESULT_DIR/audiomcq500k")
    ap.add_argument("--export-dir", type=Path)
    ap.add_argument("--no-copy-audio", action="store_true")
    args = ap.parse_args()

    data_root, result_root = os.environ.get("DATA_ROOT"), os.environ.get("UQ_RESULT_DIR")
    if not (args.slices_dir and args.result_dir and args.export_dir) and not (data_root and result_root):
        raise SystemExit("`source local_env.sh` first (or pass --slices-dir, --result-dir, --export-dir).")
    slices_dir = args.slices_dir or Path(data_root) / "datasets" / "audiomcq500k"
    result_dir = args.result_dir or Path(result_root) / "audiomcq500k"
    set_name = f"audiomcq500k_{args.metric}_{args.on}_{'le' if args.below else 'ge'}{args.threshold:g}"
    export_dir = args.export_dir or Path(data_root) / "exports" / set_name

    pool_path = result_dir / f"{MODEL_TAG}_pool_original{SUFFIX}"
    pool = load_results(pool_path)
    print(f"original pool UQ : {len(pool):,} questions")

    pairs, tool_paths = [], []
    orig_by_id, tool_by_id, manifest_by_id, tool_of = {}, {}, {}, {}
    slice_of = {}
    for sdir in sorted(slices_dir.glob("slice_[0-9][0-9]")):
        tool_path = result_dir / f"{MODEL_TAG}_{sdir.name}_tool_applied{SUFFIX}"
        if not tool_path.exists() and not Path(f"{tool_path}.partial.jsonl").exists():
            print(f"  {sdir.name}: no tool-applied UQ yet, skipped")
            continue
        tool_paths.append(str(tool_path))
        tool_res = load_results(tool_path)
        manifest = {r["id"]: r for r in json.loads((sdir / "manifest.json").read_text(encoding="utf-8"))}
        orig_sub = {e["id"]: e for e in json.loads((sdir / "original_subset.json").read_text(encoding="utf-8"))}
        tool_sub = {e["id"]: e for e in json.loads((sdir / "tool_applied_subset.json").read_text(encoding="utf-8"))}
        n0 = len(pairs)
        for vid, t in tool_res.items():
            rec = manifest.get(vid)
            if rec is None or rec["base_id"] not in pool or vid not in orig_sub or vid not in tool_sub:
                continue
            o = dict(pool[rec["base_id"]], id=vid)  # shared original result, under the variant id
            pairs.append((vid, o, t))
            orig_by_id[vid], tool_by_id[vid], manifest_by_id[vid] = orig_sub[vid], tool_sub[vid], rec
            tool_of[vid] = "+".join(s["tool"] for s in rec["tool_chain"]) or "(none)"
            slice_of[vid] = sdir.name
        print(f"  {sdir.name}: {len(pairs) - n0:,} paired")

    kept, missing = select(pairs, args.metric, args.threshold, args.on, args.below)
    # Pruned slices deleted tool audio whose P(True) fell; make sure every kept item still has it.
    gone = [it["id"] for it in kept if not Path(tool_by_id[it["id"]].get("audio_path")
                                                 or tool_by_id[it["id"]]["audio_url"]).exists()]
    if gone:
        print(f"WARNING: {len(gone)} kept item(s) have no tool-applied audio on disk (pruned?), "
              f"left out, e.g. {gone[:3]}")
        kept = [it for it in kept if it["id"] not in set(gone)]

    n = len(kept)
    rule = describe_filter(args.metric, args.threshold, args.on, args.below)
    counts = confusion(kept)
    print(f"\npaired items     : {len(pairs):,}")
    print(f"filter           : {rule}")
    print(f"kept             : {n:,}  (distinct questions {len({manifest_by_id[it['id']]['base_id'] for it in kept}):,})"
          f"   missing {args.metric}: {missing:,}")
    print(f"accuracy (kept)  : original {accuracy(kept, 'original'):.2%} -> tool-applied {accuracy(kept, 'tool_applied'):.2%}\n")
    print_matrix(counts, n)
    by_tool = collections.defaultdict(list)
    for it in kept:
        by_tool[tool_of[it["id"]]].append(it)
    print(f"\n  {'tool':24}{'n':>7}{'T->T':>8}{'T->F':>8}{'F->T':>8}{'F->F':>8}{'acc orig':>10}{'acc tool':>10}")
    for tool, items in sorted(by_tool.items(), key=lambda kv: -len(kv[1])):
        c = confusion(items)
        print(f"  {tool:24}{len(items):>7}" + "".join(f"{c[k]:>8}" for k in CELLS)
              + f"{accuracy(items, 'original'):>10.1%}{accuracy(items, 'tool_applied'):>10.1%}")

    export_items(kept, slices_dir, export_dir, not args.no_copy_audio, {
        "filter": rule,
        "metric": args.metric,
        "threshold": args.threshold,
        "on": args.on,
        "below": args.below,
        "uq_results": {"original": str(pool_path), "tool_applied": tool_paths},
        "pairing": "500k: variant id -> manifest base_id -> shared original pool result",
        "n_paired": len(pairs),
        "n_kept": n,
        "n_distinct_questions": len({manifest_by_id[it["id"]]["base_id"] for it in kept}),
        "confusion": {f"{a}->{b}": counts[(a, b)] for a, b in CELLS},
        "accuracy_original": accuracy(kept, "original"),
        "accuracy_tool_applied": accuracy(kept, "tool_applied"),
    }, sources=(orig_by_id, tool_by_id, manifest_by_id))

    plot_matrix(counts, n, "Original → tool-applied correctness (500k)",
                f"Filter: {rule}\n{n:,} of {len(pairs):,} paired items kept · accuracy "
                f"{accuracy(kept, 'original'):.1%} → {accuracy(kept, 'tool_applied'):.1%}",
                Path(export_dir) / "confusion.png")
    print(f"figure: {Path(export_dir) / 'confusion.png'}")


if __name__ == "__main__":
    main()
