#!/usr/bin/env python3
"""Validate tool_schedules.json against tool parameter schemas.

Structure expected:
  [ [metadata_obj, [tool_call, ...]], ... ]

Each tool_call: {"tool": "<name>", "parameters": {...}}

Validation mirrors abstract_tool.Tool.validate_parameters() and _validate_timestamp()
without importing any heavy ML dependencies.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Known tools (mirrors tool_execute._TOOL_MODULES)
# ---------------------------------------------------------------------------
KNOWN_TOOLS = {
    "asr",
    "clipping",
    "denoise",
    "amplitude_normalize",
    "loudness_normalize",
    "remove_dc_offset",
    "spectral_normalize",
    "trim_silence",
    "pre_emphasis",
    "source_separation",
    "extract_target",
    "remove_target",
}

# ---------------------------------------------------------------------------
# Inlined parameter schemas (copied verbatim from each tool's parameter_schema)
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "asr": {
        "properties": {
            "audio_path": {"type": "string"},
            "audio_begin": {"type": "string", "format": "HH:MM:SS.mmm"},
            "audio_end": {"type": "string", "format": "HH:MM:SS.mmm"},
            "language": {"type": "string"},
        },
        "required": ["audio_path", "audio_begin", "audio_end"],
    },
    "clipping": {
        "properties": {
            "audio_path": {"type": "string"},
            "audio_begin": {"type": "string", "format": "HH:MM:SS.mmm"},
            "audio_end": {"type": "string", "format": "HH:MM:SS.mmm"},
        },
        "required": ["audio_path", "audio_begin", "audio_end"],
    },
    "denoise": {
        "properties": {
            "audio_path": {"type": "string"},
            "audio_begin": {"type": "string", "format": "HH:MM:SS.mmm"},
            "audio_end": {"type": "string", "format": "HH:MM:SS.mmm"},
            "algorithm": {
                "type": "string",
                "enum": ["spectral_subtraction", "wiener", "echo_cancellation", "adaptive"],
            },
            "noise_factor": {"type": "number"},
            "sensitivity": {"type": "number"},
        },
        "required": ["audio_path", "audio_begin", "audio_end", "algorithm"],
    },
    "amplitude_normalize": {
        "properties": {
            "audio_path": {"type": "string"},
            "audio_begin": {"type": "string", "format": "HH:MM:SS.mmm"},
            "audio_end": {"type": "string", "format": "HH:MM:SS.mmm"},
            "target_level": {"type": "number"},
            "method": {"type": "string", "enum": ["peak", "rms"]},
        },
        "required": ["audio_path", "audio_begin", "audio_end"],
    },
    "loudness_normalize": {
        "properties": {
            "audio_path": {"type": "string"},
            "audio_begin": {"type": "string", "format": "HH:MM:SS.mmm"},
            "audio_end": {"type": "string", "format": "HH:MM:SS.mmm"},
            "target_lufs": {"type": "number"},
        },
        "required": ["audio_path", "audio_begin", "audio_end"],
    },
    "remove_dc_offset": {
        "properties": {
            "audio_path": {"type": "string"},
            "audio_begin": {"type": "string", "format": "HH:MM:SS.mmm"},
            "audio_end": {"type": "string", "format": "HH:MM:SS.mmm"},
        },
        "required": ["audio_path", "audio_begin", "audio_end"],
    },
    "spectral_normalize": {
        "properties": {
            "audio_path": {"type": "string"},
            "audio_begin": {"type": "string", "format": "HH:MM:SS.mmm"},
            "audio_end": {"type": "string", "format": "HH:MM:SS.mmm"},
            "strength": {"type": "number"},
        },
        "required": ["audio_path", "audio_begin", "audio_end"],
    },
    "trim_silence": {
        "properties": {
            "audio_path": {"type": "string"},
            "audio_begin": {"type": "string", "format": "HH:MM:SS.mmm"},
            "audio_end": {"type": "string", "format": "HH:MM:SS.mmm"},
            "threshold_db": {"type": "number"},
            "frame_length": {"type": "integer"},
            "hop_length": {"type": "integer"},
        },
        "required": ["audio_path", "audio_begin", "audio_end"],
    },
    "pre_emphasis": {
        "properties": {
            "audio_path": {"type": "string"},
            "audio_begin": {"type": "string", "format": "HH:MM:SS.mmm"},
            "audio_end": {"type": "string", "format": "HH:MM:SS.mmm"},
            "coef": {"type": "number"},
        },
        "required": ["audio_path", "audio_begin", "audio_end"],
    },
    "source_separation": {
        "properties": {
            "audio_path": {"type": "string"},
            "audio_begin": {"type": "string", "format": "HH:MM:SS.mmm"},
            "audio_end": {"type": "string", "format": "HH:MM:SS.mmm"},
            "target_description": {"type": "string"},
            "save_residual": {"type": "boolean"},
        },
        "required": ["audio_path", "audio_begin", "audio_end"],
    },
    "extract_target": {
        "properties": {
            "audio_path": {"type": "string"},
            "audio_begin": {"type": "string", "format": "HH:MM:SS.mmm"},
            "audio_end": {"type": "string", "format": "HH:MM:SS.mmm"},
            "target_description": {"type": "string"},
        },
        "required": ["audio_path", "audio_begin", "audio_end", "target_description"],
    },
    "remove_target": {
        "properties": {
            "audio_path": {"type": "string"},
            "audio_begin": {"type": "string", "format": "HH:MM:SS.mmm"},
            "audio_end": {"type": "string", "format": "HH:MM:SS.mmm"},
            "target_description": {"type": "string"},
        },
        "required": ["audio_path", "audio_begin", "audio_end", "target_description"],
    },
}

# Required keys in the metadata object (element 0 of each entry).
METADATA_REQUIRED_KEYS = {"question", "choice", "answer", "id", "audio_path"}


# ---------------------------------------------------------------------------
# Validation helpers (mirror abstract_tool.Tool logic)
# ---------------------------------------------------------------------------

def _validate_timestamp(name: str, value: Any) -> Optional[str]:
    """Return error string or None. Mirrors Tool._validate_timestamp()."""
    if not isinstance(value, str):
        return f"'{name}' must be a string in HH:MM:SS.mmm format, got {type(value).__name__}"
    try:
        datetime.strptime(value, "%H:%M:%S.%f")
    except ValueError as exc:
        return f"'{name}' timestamp format error: {exc}"
    return None


def _validate_parameters(tool_name: str, parameters: Any) -> List[str]:
    """Return list of error strings for the given tool parameters."""
    errors: List[str] = []

    if not isinstance(parameters, dict):
        return [f"'parameters' must be a dict, got {type(parameters).__name__}"]

    schema = TOOL_SCHEMAS[tool_name]
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    missing = [k for k in required if k not in parameters]
    if missing:
        errors.append(f"Missing required parameters: {missing}")

    for param_name, value in parameters.items():
        if param_name not in properties:
            errors.append(f"Unexpected parameter: '{param_name}'")
            continue

        spec = properties[param_name]
        expected_type = spec.get("type")

        # Mirror abstract_tool.validate_parameters: only array and string are type-checked
        if expected_type == "array":
            if not isinstance(value, list):
                errors.append(f"Parameter '{param_name}' must be an array")
        elif expected_type == "string":
            if not isinstance(value, str):
                errors.append(f"Parameter '{param_name}' must be a string")

        if spec.get("format") == "HH:MM:SS.mmm":
            err = _validate_timestamp(param_name, value)
            if err:
                errors.append(err)

        if spec.get("enum") is not None and value is not None:
            allowed = spec["enum"]
            if isinstance(value, list):
                invalid = [v for v in value if v not in allowed]
                if invalid:
                    errors.append(
                        f"Parameter '{param_name}' contains invalid values {invalid}. Allowed: {allowed}"
                    )
            elif value not in allowed:
                errors.append(
                    f"Parameter '{param_name}' has invalid value '{value}'. Allowed: {allowed}"
                )

    return errors


def _validate_tool_call(tc: Any, tc_idx: int) -> List[str]:
    """Validate a single tool call dict. Returns list of errors."""
    errors: List[str] = []
    prefix = f"tool_call[{tc_idx}]"

    if not isinstance(tc, dict):
        return [f"{prefix}: must be a dict, got {type(tc).__name__}"]

    tool_name = tc.get("tool")
    if tool_name is None:
        errors.append(f"{prefix}: missing 'tool' key")
    elif not isinstance(tool_name, str):
        errors.append(f"{prefix}: 'tool' must be a string, got {type(tool_name).__name__}")
        tool_name = None
    elif tool_name not in KNOWN_TOOLS:
        errors.append(f"{prefix}: unknown tool '{tool_name}'. Known: {sorted(KNOWN_TOOLS)}")
        tool_name = None

    if "parameters" not in tc:
        errors.append(f"{prefix}: missing 'parameters' key")
    elif tool_name is not None:
        param_errors = _validate_parameters(tool_name, tc["parameters"])
        for pe in param_errors:
            errors.append(f"{prefix}(tool={tool_name}): {pe}")

    return errors


def _validate_metadata(meta: Any) -> List[str]:
    """Validate the metadata object (entry[0]). Returns list of errors."""
    errors: List[str] = []
    if not isinstance(meta, dict):
        return [f"metadata must be a dict, got {type(meta).__name__}"]
    missing = [k for k in METADATA_REQUIRED_KEYS if k not in meta]
    if missing:
        errors.append(f"metadata missing required keys: {missing}")
    if "choice" in meta and not isinstance(meta["choice"], list):
        errors.append("metadata 'choice' must be a list")
    return errors


def _validate_entry(entry: Any, idx: int) -> List[str]:
    """Validate one top-level entry. Returns list of errors."""
    errors: List[str] = []
    prefix = f"entry[{idx}]"

    if not isinstance(entry, list):
        return [f"{prefix}: must be a 2-element list, got {type(entry).__name__}"]

    if len(entry) != 2:
        errors.append(f"{prefix}: must have exactly 2 elements, got {len(entry)}")
        return errors

    metadata, tool_calls = entry

    for err in _validate_metadata(metadata):
        errors.append(f"{prefix}.metadata: {err}")

    if not isinstance(tool_calls, list):
        errors.append(f"{prefix}.tool_calls: must be a list, got {type(tool_calls).__name__}")
    else:
        if len(tool_calls) == 0:
            errors.append(f"{prefix}.tool_calls: list is empty")
        for tc_idx, tc in enumerate(tool_calls):
            errors.extend(_validate_tool_call(tc, tc_idx))

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate(path: str) -> Tuple[int, int, int]:
    """Validate the schedules file. Returns (total_entries, error_entries, total_errors)."""
    p = Path(path)
    if not p.exists():
        print(f"ERROR: File not found: {p}", file=sys.stderr)
        sys.exit(1)

    with p.open("r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"ERROR: Invalid JSON: {exc}", file=sys.stderr)
            sys.exit(1)

    if not isinstance(data, list):
        print(f"ERROR: Top-level structure must be a list, got {type(data).__name__}", file=sys.stderr)
        sys.exit(1)

    total_entries = len(data)
    error_entries = 0
    total_errors = 0

    for idx, entry in enumerate(data):
        entry_errors = _validate_entry(entry, idx)
        if entry_errors:
            error_entries += 1
            total_errors += len(entry_errors)
            print(f"\n--- entry[{idx}] INVALID ({len(entry_errors)} error(s)) ---")
            for err in entry_errors:
                print(f"  [ERR] {err}")

    print(f"\n{'='*60}")
    print(f"File   : {p.resolve()}")
    print(f"Entries: {total_entries}")
    if total_errors == 0:
        print("Result : OK — all entries are valid")
    else:
        print(f"Result : FAILED — {error_entries}/{total_entries} entries have errors ({total_errors} total)")

    return total_entries, error_entries, total_errors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate tool_schedules.json legality.")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(Path(__file__).resolve().parent / "tool_schedules.json"),
        help="Path to tool_schedules.json (default: ./tool_schedules.json)",
    )
    args = parser.parse_args()

    _, _, errs = validate(args.path)
    sys.exit(0 if errs == 0 else 1)
