"""Re-run the per-tool analysis (tool_analysis.py) on an existing results file.

No model / GPU / audio involved -- just reads JSON. Results files written
before `gt_calls` was stored per sample only carry GT tool *names*; pass
--benchmark-file so GT parameters can be looked up by sample id for the
parameter comparison.

Usage:
    python analyze_results.py results_qwen3_200.json --benchmark-file benchmark.json
Writes <results>_tool_analysis.json / .txt next to the results file (or
under --output-prefix) and prints the text report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tool_analysis import analyze_per_tool, format_tool_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_file", type=Path)
    parser.add_argument("--benchmark-file", type=Path, default=None,
                        help="Backfill GT parameters by sample id for results lacking 'gt_calls'.")
    parser.add_argument("--output-prefix", type=Path, default=None,
                        help="Default: <results_file without .json>.")
    args = parser.parse_args()

    with args.results_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"] if isinstance(data, dict) else data

    gt_calls_by_id = None
    if args.benchmark_file:
        with args.benchmark_file.open("r", encoding="utf-8") as f:
            gt_calls_by_id = {e["id"]: e["answer"]["tool_calls"] for e in json.load(f)}

    analysis = analyze_per_tool(results, gt_calls_by_id=gt_calls_by_id, include_pairs=True)
    report = format_tool_analysis(analysis)

    prefix = args.output_prefix or args.results_file.with_suffix("")
    with Path(f"{prefix}_tool_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    with Path(f"{prefix}_tool_analysis.txt").open("w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
