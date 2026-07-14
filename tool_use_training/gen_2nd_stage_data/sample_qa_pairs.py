"""Randomly sample a subset of QA pairs to feed into `build_by_disturb.py`.

Deliberately kept separate from `build_by_disturb.py`: that script disturbs+recovers
whatever list of QA pairs it's handed, and shouldn't own the sampling policy (how many
items, from which benchmark file(s), with what class balance, ...). This script is
just a thin, swappable front-end for producing that input list, e.g. from a big
DCASE/MMAU-shaped JSON.

Usage:
    python sample_qa_pairs.py --input dcase_subset.json --num-samples 200 \\
        --output sampled_qa_pairs.json --seed 0

Multiple --input files can be given; they're concatenated before sampling (each item
must already be dcase_subset.json-shaped: question/choice/answer/id/audio_path, ...).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def load_items(paths: List[Path]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise SystemExit(f"{path} must contain a JSON list of QA-pair dicts.")
        items.extend(data)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomly sample QA pairs for build_by_disturb.py.")
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="One or more source JSON files (dcase_subset.json-shaped lists).")
    parser.add_argument("--output", type=Path, required=True, help="Path to write the sampled JSON list.")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dedupe-by-audio", action="store_true",
                         help="Drop duplicate items that share the same audio_path/audio_url before sampling.")
    args = parser.parse_args()

    items = load_items(args.input)
    if args.dedupe_by_audio:
        seen = set()
        deduped = []
        for item in items:
            key = item.get("audio_path") or item.get("audio_url")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        items = deduped

    rng = random.Random(args.seed)
    k = min(args.num_samples, len(items))
    sample = rng.sample(items, k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(sample, handle, indent=2, ensure_ascii=False)
    print(f"Sampled {len(sample)}/{len(items)} QA pairs from {len(args.input)} file(s) -> {args.output}")


if __name__ == "__main__":
    main()
