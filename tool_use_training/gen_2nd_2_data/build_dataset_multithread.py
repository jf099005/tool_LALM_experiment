"""Multithreaded variant of `build_dataset.py` -- same pipeline, same output shape, same
CLI, just runs the per-sample work (`build_one_sample`) across a thread pool instead of a
plain `for` loop. See build_dataset.py's module docstring for the overall strategy this
implements; this file only documents what's different.

Why threads help here: each sample's work is a mix of (a) light in-process DSP
(librosa/numpy/scipy, which release the GIL for their C/Fortran-level FFT/resample calls),
(b) file I/O (copy/move under --output-dir), and (c) for heavy tools (disabled by default in
tool_config.json), a blocking `subprocess.run` into another conda env -- all three release
the GIL while waiting, so a thread pool overlaps them across samples instead of the strictly
serial one-sample-at-a-time original.

The one correctness issue that blocks a naive "just wrap the loop in an executor": the
original script threads a single shared `random.Random(args.seed)` through every
`build_one_sample` call, drawing tool choices/parameters from one continuing stream. Handed
to concurrent workers, calls into that shared, mutable RNG would still be individually
atomic (CPython's GIL), but their *interleaving order* would depend on thread scheduling --
so the same `--seed` would no longer reproduce the same dataset run to run. Fixed here by
giving each sample its own `random.Random` seeded deterministically from
`(args.seed, idx, item_id)` -- `idx` (the sample's position in the sampled-and-repeated
list) is included so that `--repetition > 1` still draws a different tool chain for each
repeated occurrence of the same item, exactly as the shared-stream version did.

Usage: identical to build_dataset.py, plus one extra flag:
    python build_dataset_multithread.py \\
        --input-json /home/u1501463/tool_use_LALM/dcase_subset.json \\
        --num-samples 200 \\
        --output-dir /work/u1501463/gen_2nd_2_stage_tool_applied \\
        --workers 8 \\
        --seed 0
"""

from __future__ import annotations

import argparse
import glob as glob_module
import json
import os
import random
import sys
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import tool_appliers as ta  # noqa: E402

# Reuse build_dataset.py's step-1/step-2 helpers verbatim (sampling, dcase-entry shaping,
# weighted draw, tool-chain application, single-sample build) rather than forking them --
# only `main()` below differs, to drive them from a thread pool.
from build_dataset import (  # noqa: E402
    DEFAULT_AUDIOSET_DIR,  # noqa: F401
    DEFAULT_VCTK_DIR,  # noqa: F401
    build_one_sample,
    collect_raw_audio_items,
    load_audiomcq_items,
    load_qa_items,
    sample_items,
    to_dcase_entry,
)


# ---------------------------------------------------------------------------
# Per-task RNG + worker
# ---------------------------------------------------------------------------

def _make_task_rng(seed: int, idx: int, item_id: str) -> random.Random:
    """Deterministic, per-task RNG so concurrent workers don't share (and race on) one
    mutable random.Random stream -- see module docstring for why that would otherwise
    break --seed reproducibility."""
    return random.Random(f"{seed}-{idx}-{item_id}")


def _process_item(
    idx: int,
    total: int,
    item: Dict[str, Any],
    args: argparse.Namespace,
    tool_names: List[str],
    tool_weights_list: List[float],
    tool_config: Dict[str, Any],
    min_tools: int,
    max_tools: int,
    print_lock: threading.Lock,
) -> Optional[Dict[str, Any]]:
    rng = _make_task_rng(args.seed, idx, str(item["id"]))
    for attempt in range(args.max_attempts_per_sample):
        try:
            record = build_one_sample(
                item, args.output_dir, tool_names, tool_weights_list, tool_config, min_tools, max_tools, rng
            )
            ops = [step["tool"] for step in record["tool_chain"]]
            with print_lock:
                print(f"[{idx}/{total}] {item['id']}: applied {ops}", file=sys.stderr)
            return record
        except Exception:
            with print_lock:
                print(f"Attempt {attempt + 1} failed for {item['id']}:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
    with print_lock:
        print(f"Giving up on {item['id']} after {args.max_attempts_per_sample} attempts.", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multithreaded build_dataset.py: sample real audio and apply a randomly "
        "drawn tools/ call to build (original, tool_applied) pairs."
    )
    parser.add_argument("--tool-config", type=Path, default=Path(__file__).resolve().parent / "tool_config.json")
    parser.add_argument("--input-json", type=Path, nargs="*", default=[],
                         help="dcase_subset.json-shaped QA list(s) to sample from (e.g. a DCASE 2025 subset).")
    parser.add_argument("--sources", nargs="*",
                         choices=["audioset", "vctk", "clotho", "audiocaps", "compa_r"], default=[],
                         help="Raw audio pools (no QA attached) to sample from.")
    parser.add_argument("--audiomcq", nargs="*", choices=["clotho", "audiocaps", "compa_r"], default=[],
                         help="Pull QA-attached items (real question/choices/answer) from AudioMCQ's "
                         "per-source-dataset jsonl exports instead of the corresponding --sources pool.")
    parser.add_argument("--audio-glob", nargs="*", default=[],
                         help="Extra glob pattern(s) for other raw audio pools, e.g. '/work/u1501463/my_corpus/**/*.wav'.")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--repetition", type=int, default=1, help="Repeat the whole sample set this many times (for debugging).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dedupe-by-audio", action="store_true")
    parser.add_argument("--min-tools", type=int, default=None, help="Default: tool_config.json's min_tools.")
    parser.add_argument("--max-tools", type=int, default=None, help="Default: tool_config.json's max_tools.")
    parser.add_argument("--output-dir", type=Path, default=Path("/work/u1501463/gen_2nd_2_stage_tool_applied"))
    parser.add_argument("--max-attempts-per-sample", type=int, default=3)
    parser.add_argument("--manifest-file", type=Path, default=None, help="Default: <output-dir>/manifest.json")
    parser.add_argument("--original-subset-file", type=Path, default=None, help="Default: <output-dir>/original_subset.json")
    parser.add_argument("--tool-applied-subset-file", type=Path, default=None, help="Default: <output-dir>/tool_applied_subset.json")
    parser.add_argument("--workers", type=int, default=4,
                         help="Thread pool size. Heavy (subprocess-dispatched) tools and "
                         "file I/O benefit most from more workers; the default light-tool "
                         "set (clipping/denoise/pitch_shift/time_stretch/normalize family) "
                         "is CPU-bound in-process DSP, so past a handful of workers you're "
                         "mostly bounded by cores, not I/O wait.")
    args = parser.parse_args()

    tool_config = ta.load_tool_config(args.tool_config)
    min_tools = args.min_tools if args.min_tools is not None else int(tool_config.get("min_tools", 1))
    max_tools = args.max_tools if args.max_tools is not None else int(tool_config.get("max_tools", 1))

    tool_names = ta.available_tool_names(tool_config)
    if not tool_names:
        raise SystemExit("No enabled+available tools in tool_config.json -- check `enabled` flags and heavy-tool `env` paths.")
    tool_weights_list = ta.tool_weights(tool_config, tool_names)
    print(f"Available tools ({len(tool_names)}): {tool_names}", file=sys.stderr)

    items = (
        load_qa_items(args.input_json)
        + load_audiomcq_items(args.audiomcq)
        + collect_raw_audio_items(args.sources, args.audio_glob)
    )
    if not items:
        raise SystemExit("No input items -- pass --input-json, --audiomcq, and/or --sources/--audio-glob.")

    for item in items:
        item.setdefault("id", f"sample_{uuid.uuid4().hex[:12]}")

    sampled = sample_items(items, args.num_samples, args.seed, args.dedupe_by_audio)
    print(f"Sampled {len(sampled)}/{len(items)} item(s)", file=sys.stderr)
    sampled = sampled * args.repetition
    print(f"After repetition={args.repetition}, {len(sampled)} total item(s)", file=sys.stderr)

    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = args.manifest_file or (args.output_dir / "manifest.json")
    original_subset_file = args.original_subset_file or (args.output_dir / "original_subset.json")
    tool_applied_subset_file = args.tool_applied_subset_file or (args.output_dir / "tool_applied_subset.json")

    print(f"Running with {args.workers} worker thread(s)", file=sys.stderr)
    print_lock = threading.Lock()
    results: List[Optional[Dict[str, Any]]] = [None] * len(sampled)
    # Heavy ML tools (if any are enabled) share one persistent server per conda env, so their
    # GPU calls are serialized however many worker threads are waiting on them.
    with ta.heavy_tool_pool.serve(ta.heavy_env_pythons(tool_config, tool_names)), \
            ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {
            executor.submit(
                _process_item, idx, len(sampled), item, args, tool_names, tool_weights_list,
                tool_config, min_tools, max_tools, print_lock,
            ): idx - 1
            for idx, item in enumerate(sampled, start=1)
        }
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx)):
            results[future_to_idx[future]] = future.result()

    # Keep manifest order matching the original sampled order (not completion order), same
    # as the sequential script would have produced.
    records: List[Dict[str, Any]] = [r for r in results if r is not None]

    qa_by_id = {item["id"]: item for item in sampled}

    with manifest_file.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)
    print(f"Wrote manifest for {len(records)} samples to {manifest_file}", file=sys.stderr)

    original_subset = [to_dcase_entry(qa_by_id[rec["id"]], rec["original_audio"]) for rec in records]
    with original_subset_file.open("w", encoding="utf-8") as handle:
        json.dump(original_subset, handle, indent=2, ensure_ascii=False)
    print(f"Wrote original subset ({len(original_subset)} items) to {original_subset_file}", file=sys.stderr)

    tool_applied_subset = [to_dcase_entry(qa_by_id[rec["id"]], rec["tool_applied_audio"]) for rec in records]
    with tool_applied_subset_file.open("w", encoding="utf-8") as handle:
        json.dump(tool_applied_subset, handle, indent=2, ensure_ascii=False)
    print(f"Wrote tool_applied subset ({len(tool_applied_subset)} items) to {tool_applied_subset_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
