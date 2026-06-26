from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from tool_execute import get_tool_class


def read_json_file(path: str) -> Any:
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"JSON input file not found: {path}")

    with candidate.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_result(result: Any, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a batch of tool calls from a JSON file.")
    parser.add_argument("--tool-name", required=True, help="Name of the tool to execute.")
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to a JSON file containing a list of tool parameter dictionaries.",
    )
    parser.add_argument(
        "--output-file",
        default="tool_batch_results.json",
        help="Path to write the batch execution result JSON.",
    )

    args = parser.parse_args()

    try:
        batch_parameters = read_json_file(args.input_file)
    except Exception as exc:
        print(f"Failed to read input file: {exc}", file=sys.stderr)
        sys.exit(1)

    tool_cls = get_tool_class(args.tool_name)

    try:
        result = tool_cls.execute_batch(batch_parameters)
    except Exception as exc:
        result = {
            "tool_name": args.tool_name,
            "input_file": args.input_file,
            "status": "error",
            "error": str(exc),
        }

    output_path = Path(args.output_file)
    write_result(result, output_path)
    print(f"Saved batch result to {output_path.resolve()}")


if __name__ == "__main__":
    main()
