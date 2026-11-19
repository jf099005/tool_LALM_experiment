"""Export N (original, tool-applied) pairs of a build_dataset.py dataset, unfiltered, for the
cluster's full-precision Qwen3-Omni run (qwen3_uq_benchmark.py, vLLM bf16).

The UQ-filtered exports (uq_threshold_confusion.py --export-dir) only keep items a model picked;
this takes a plain random (or first-N) sample, so the cluster's bf16 run measures the tools'
effect without any selection, and the PC's quantized run can be compared on the same items.

Run on the PC (CPU only), then upload with push_exports_to_nano5.sh:
    $PY export_random_subset.py \
        --subset-dir "$DATA_ROOT/datasets/audiomcq_strong_50k_tool_applied_ml40" \
        --n 10000 --seed 0 --export-dir "$DATA_ROOT/exports/audiomcq50k_random10k_s0" \
        --ref-dir "$DATA_ROOT/exp/qwen3_llamacpp/<run dir of the quantized model>"   # optional
    ./push_exports_to_nano5.sh "$DATA_ROOT/exports/audiomcq50k_random10k_s0"

Writes, in --export-dir (same layout as the other exports):
    original_subset.json, tool_applied_subset.json   N items each, same ids and order
    manifest.json                                    the source manifest rows of those ids
    filter.json                                      selection rule, export_dir (for path re-rooting)
                                                     and, with --ref-dir, the reference run's per-item
                                                     p_true_*/correct_* (shown as "reference run")
    original/<id>/original<ext>, tool_applied/<id>/tool_applied<ext>   copied audio
    reference/                                       with --ref-dir: its result files cut to these ids,
                                                     for compare_quant_uq.py on the cluster
"""
from __future__ import annotations

import json
import os
import random
import shutil
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SIDES = ("original", "tool_applied")
RESULT_SUFFIX = "_uq_results_no_tool_chain_results.json"


def load_json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p: Path, obj) -> None:
    tmp = Path(f"{p}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def load_results(path: Path) -> dict:
    """A qwen3_uq_benchmark / qwen25 result file (.json), plus its .partial.jsonl if still running."""
    rows = {}
    if path.exists():
        rows = {r["id"]: r for r in load_json(path)}
    partial = Path(f"{path}.partial.jsonl")
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.setdefault(r["id"], r)
    return rows


def find_result(ref_dir: Path, side: str) -> Path:
    found = sorted(ref_dir.glob(f"*_{side}{RESULT_SUFFIX}")) or \
        [Path(str(p)[: -len(".partial.jsonl")]) for p in sorted(ref_dir.glob(f"*_{side}{RESULT_SUFFIX}.partial.jsonl"))]
    if len(found) != 1:
        raise SystemExit(f"--ref-dir {ref_dir}: expected one *_{side}{RESULT_SUFFIX}, found {len(found)}")
    return found[0]


def main():
    ap = ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--subset-dir", type=Path, required=True,
                    help="build_dataset.py output (original_subset.json, tool_applied_subset.json, manifest.json)")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0, help="random sample seed (ignored with --first)")
    ap.add_argument("--first", action="store_true", help="take the first N items instead of a random sample")
    ap.add_argument("--export-dir", type=Path, required=True)
    ap.add_argument("--ref-dir", type=Path, default=None,
                    help="optional reference run on the same dataset (qwen3_uq_benchmark.py --output_dir, "
                         "e.g. the quantized llama.cpp run): its values go into filter.json and reference/")
    ap.add_argument("--workers", type=int, default=8, help="parallel audio copies")
    args = ap.parse_args()

    src = {s: load_json(args.subset_dir / f"{s}_subset.json") for s in SIDES}
    if [i["id"] for i in src["original"]] != [i["id"] for i in src["tool_applied"]]:
        raise SystemExit("source original/tool_applied subsets must list the same ids in the same order")
    n_all = len(src["original"])
    if args.n > n_all:
        raise SystemExit(f"--n {args.n} > {n_all} items in {args.subset_dir}")
    if args.first:
        keep = list(range(args.n))
    else:
        keep = sorted(random.Random(args.seed).sample(range(n_all), args.n))  # keep source order
    ids = [src["original"][k]["id"] for k in keep]

    manifest_path = args.subset_dir / "manifest.json"
    manifest = {r["id"]: r for r in load_json(manifest_path)} if manifest_path.exists() else {}

    out = args.export_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Audio: copy each side's file to <side>/<id>/<side><ext>, and point the items at the copy.
    jobs, items = [], {s: [] for s in SIDES}
    for k in keep:
        for s in SIDES:
            it = dict(src[s][k])
            a = it.get("audio_path") or it.get("audio_url")
            if not a or not Path(a).exists():
                raise SystemExit(f"{s} audio missing for {it['id']}: {a}")
            dst = out / s / it["id"] / f"{s}{Path(a).suffix or '.wav'}"
            jobs.append((Path(a), dst))
            it["audio_path"] = it["audio_url"] = str(dst)
            items[s].append(it)

    def copy(job):
        a, dst = job
        if dst.exists() and dst.stat().st_size == a.stat().st_size:
            return  # resumable
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a, dst)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(copy, jobs), 1):
            if i % 2000 == 0 or i == len(jobs):
                print(f"copied {i}/{len(jobs)} audio files", flush=True)

    for s in SIDES:
        write_json(out / f"{s}_subset.json", items[s])
    if manifest:  # only tool_chain is used downstream (per-tool breakdown); rows kept as they are
        write_json(out / "manifest.json", [manifest.get(i, {"id": i, "tool_chain": []}) for i in ids])

    flt = {
        "filter": ("first" if args.first else f"random sample, seed {args.seed}") + f": {args.n} of {n_all} items, unfiltered",
        "n_items": args.n, "n_source_items": n_all,
        "source_dataset_dir": str(args.subset_dir.expanduser().resolve()),
        "export_dir": str(out),
        "audio_copied": True,
    }
    if args.ref_dir:
        ref_paths = {s: find_result(args.ref_dir, s) for s in SIDES}
        ref = {s: load_results(ref_paths[s]) for s in SIDES}
        tool_of = {i: "+".join(t["tool"] for t in manifest.get(i, {}).get("tool_chain", [])) for i in ids}
        rows = []
        for i in ids:
            o, t = ref["original"].get(i), ref["tool_applied"].get(i)
            if o is None or t is None:
                continue
            rows.append({
                "id": i, "tool": tool_of.get(i) or "(unknown)",
                "p_true_original": (o.get("uncertainty") or {}).get("p_true"),
                "p_true_tool_applied": (t.get("uncertainty") or {}).get("p_true"),
                "correct_original": bool(o.get("correct")), "correct_tool_applied": bool(t.get("correct")),
            })
        flt["model"] = ref_paths["original"].name[: -len(f"_original{RESULT_SUFFIX}")]
        flt["reference_dir"] = str(args.ref_dir.expanduser().resolve())
        flt["n_reference_items"] = len(rows)
        flt["items"] = rows
        # The reference results cut to the exported ids, in the same file layout, so the cluster can
        # run `compare_quant_uq.py <bf16 output dir> <export>/reference`.
        (out / "reference").mkdir(exist_ok=True)
        id_set = set(ids)
        for s in SIDES:
            write_json(out / "reference" / ref_paths[s].name, [r for i, r in ref[s].items() if i in id_set])
        print(f"reference run {flt['model']}: {len(rows)}/{len(ids)} exported items have both sides")
    write_json(out / "filter.json", flt)
    print(f"exported {len(ids)} pairs -> {out}")


if __name__ == "__main__":
    main()
