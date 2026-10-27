#!/usr/bin/env python
"""CLI wrapper around `probe_lib.run_probe` -- run ONE (adapter x dtype x
audio) variant in its own process (so GPU memory and dtype fully reset
between variants) and optionally dump the result as JSON. Invoked by
`run_all.sh` for the three isolation tests; also fine to call by hand:

    python probe_first_token.py --no-audio
    python probe_first_token.py --no-adapter
    python probe_first_token.py --dtype fp32
"""

import argparse
import json

from probe_lib import DEFAULT_ADAPTER, DEFAULT_MODEL, DEFAULT_SAMPLE_ID, print_probe_result, run_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="LoRA adapter dir")
    parser.add_argument("--no-adapter", action="store_true", help="run the raw base checkpoint, no LoRA")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--no-audio", action="store_true", help="strip audio, text-only prompt")
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--out", help="optional path to dump the result as JSON")
    args = parser.parse_args()

    result = run_probe(
        model=args.model,
        adapter_dir=None if args.no_adapter else args.adapter,
        dtype=args.dtype,
        use_audio=not args.no_audio,
        sample_id=args.sample_id,
        topk=args.topk,
    )
    print_probe_result(result)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
