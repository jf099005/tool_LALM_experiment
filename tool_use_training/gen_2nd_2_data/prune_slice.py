"""After a slice's UQ run, delete the tool-applied audio of items whose UQ metric went DOWN
with the tool (default metric: P(True)), keeping every record so the audio can be rebuilt.

For each record in <slice>/manifest.json: compare the tool-applied UQ value (--tool-uq) with
the original-audio value of its question (--original-uq, matched on base_id). If
tool < original (strictly), the sample's tool_applied/<id>/ dir is removed. Items with a
missing value on either side are kept.

Writes <slice>/prune_log.jsonl (one line per item: id, base_id, tool, parameters, both metric
values, action = kept / deleted / missing) and <slice>/prune_summary.json. Idempotent: rerunning
only re-deletes what is still on disk. The manifest, subsets and UQ results are never touched.

Rebuilding deleted audio: rerun the slice's build_dataset.py command unchanged except for
`--only-ids <ids.json> --output-dir <new dir>` (ids = the "deleted" ids in prune_log.jsonl);
the per-sample RNG reproduces the same tool parameters. Light tools are deterministic;
SAM-Audio / DeepFilterNet outputs may differ slightly (GPU nondeterminism).
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
from pathlib import Path


def load_uq(path: Path, metric: str) -> dict:
    """id -> metric value, from a finished results .json and/or its .partial.jsonl."""
    out = {}
    candidates = [path, Path(f"{path}.partial.jsonl")]
    for p in candidates:
        if not p.exists():
            continue
        rows = (json.loads(l) for l in p.open(encoding="utf-8") if l.strip()) if p.name.endswith(".jsonl") \
            else json.loads(p.read_text(encoding="utf-8"))
        for r in rows:
            out[r["id"]] = (r.get("uncertainty") or {}).get(metric)
    if not out:
        raise SystemExit(f"No UQ results at {path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slice-dir", type=Path, required=True)
    ap.add_argument("--tool-uq", type=Path, required=True, help="UQ results of the slice's tool_applied_subset")
    ap.add_argument("--original-uq", type=Path, required=True, help="UQ results of the question pool (original audio)")
    ap.add_argument("--metric", default="p_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tool_uq = load_uq(args.tool_uq, args.metric)
    orig_uq = load_uq(args.original_uq, args.metric)
    manifest = json.loads((args.slice_dir / "manifest.json").read_text(encoding="utf-8"))

    counts = collections.Counter()
    freed = 0
    with (args.slice_dir / "prune_log.jsonl").open("w", encoding="utf-8") as log:
        for rec in manifest:
            vt, vo = tool_uq.get(rec["id"]), orig_uq.get(rec.get("base_id"))
            sample_dir = args.slice_dir / "tool_applied" / rec["id"]
            if vt is None or vo is None:
                action = "missing"
            elif vt < vo:
                action = "deleted"
                if sample_dir.exists():
                    freed += sum(f.stat().st_size for f in sample_dir.rglob("*") if f.is_file())
                    if not args.dry_run:
                        shutil.rmtree(sample_dir)
            else:
                action = "kept"
            counts[action] += 1
            step = rec["tool_chain"][0] if rec.get("tool_chain") else {}
            log.write(json.dumps({
                "id": rec["id"], "base_id": rec.get("base_id"), "tool": step.get("tool"),
                "parameters": step.get("parameters"), f"{args.metric}_original": vo,
                f"{args.metric}_tool_applied": vt, "action": action,
            }, ensure_ascii=False) + "\n")

    summary = {"metric": args.metric, "rule": "delete if tool_applied < original", "dry_run": args.dry_run,
               **dict(counts), "freed_gb": round(freed / 1e9, 2)}
    (args.slice_dir / "prune_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
