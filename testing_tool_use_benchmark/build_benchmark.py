"""Build a held-out benchmark for evaluating a trained LALM's tool-use ability.

Same construction recipe as tool_use_training/gen_1st_stage_data/build_dataset.py
(pick a source audio A, draw a tool chain, apply it to get target audio B), but the
output schema is eval-oriented: it keeps the per-step ground-truth tool calls
*and* per-step audio paths so a grader can later (a) check whether the model's
own predicted tool calls execute, and (b) compare the model's reconstructed
audio against the ground-truth intermediate/final audio.

Run this under the `ms-swift` conda env (it has librosa/soundfile/numpy/scipy,
which is enough for every tool except the heavy ML-backed ones that depend on
deepfilternet/audiosr/sam_audio -- those are simply absent from the toolset if
unavailable, same as build_dataset.py).

Usage:
    ~/miniconda3/envs/ms-swift/bin/python build_benchmark.py \
        --num-samples 100 --seed 1234 \
        --output-dir /work/u1501463/tool_use_benchmark_audio \
        --output-file benchmark.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
import uuid
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tool_use_training.gen_1st_stage_data.build_dataset import build_one_sample, collect_source_files  # noqa: E402
from tools import tools_registry as tool_registry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a held-out tool-use benchmark.")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--min-tools", type=int, default=1)
    parser.add_argument("--max-tools", type=int, default=4)
    parser.add_argument("--sources", nargs="+", choices=["audioset", "vctk"], default=["audioset"])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/work/u1501463/tool_use_benchmark_audio"),
        help="Directory to write generated audio (A/B + intermediates) into.",
    )
    parser.add_argument(
        "--output-file", type=Path, default=Path(__file__).resolve().parent / "benchmark.json",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Use a seed disjoint from training generation.")
    parser.add_argument("--max-attempts-per-sample", type=int, default=5)
    parser.add_argument(
        "--held-out-fraction", type=float, default=1.0,
        help="Fraction of the tail of the sorted source file list to draw from, so the "
        "benchmark doesn't reuse the same source files as a prior train split that used "
        "the head of the list. 1.0 = use all files.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    source_files = collect_source_files(args.sources)
    if not source_files:
        raise SystemExit(f"No source audio files found for sources={args.sources}")
    if args.held_out_fraction < 1.0:
        cut = int(len(source_files) * (1.0 - args.held_out_fraction))
        source_files = source_files[cut:]

    tool_names = tool_registry.available_tool_names()
    if not tool_names:
        raise SystemExit("No tools available in the current environment -- check tool_registry dependencies.")
    print(f"Available tools ({len(tool_names)}): {tool_names}", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset: List[dict] = []
    while len(dataset) < args.num_samples:
        source_file = rng.choice(source_files)
        sample_id = f"bench_{uuid.uuid4().hex[:12]}"

        for attempt in range(args.max_attempts_per_sample):
            try:
                entry = build_one_sample(
                    sample_id=sample_id,
                    source_file=source_file,
                    work_dir=args.output_dir,
                    tool_names=tool_names,
                    min_tools=args.min_tools,
                    max_tools=args.max_tools,
                    rng=rng,
                )
                dataset.append(entry)
                print(f"[{len(dataset)}/{args.num_samples}] {sample_id}: {entry['num_steps']} steps", file=sys.stderr)
                break
            except Exception:
                print(f"Attempt {attempt + 1} failed for {sample_id} ({source_file}):", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                source_file = rng.choice(source_files)
        else:
            print(f"Giving up on {sample_id} after {args.max_attempts_per_sample} attempts.", file=sys.stderr)

    with args.output_file.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, indent=2)

    print(f"Wrote {len(dataset)} benchmark samples to {args.output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
