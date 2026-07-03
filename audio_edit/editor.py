"""Unified audio-editing API and CLI.

Supports five augmentation operations, each importable directly and each
runnable as a CLI subcommand:

  1. add_noise                -- mix white/pink noise across the whole clip
  2. pad_noise                -- extend the clip with noise at start/end/both
  3. insert_background_event  -- overlay a labeled real-world sound event
                                  (e.g. "Dog", "Wind") at a timestamp
  4. change_speed             -- time-stretch (speed up/down)
  5. change_pitch             -- pitch-shift up/down

Every parameter has a sensible default (documented in each op's own module)
but can be overridden explicitly, per-call or via the CLI. Speed/pitch are not
reimplemented here -- they delegate to the existing `tools.pitch_time` tools.

Python API:

    from audio_edit.editor import edit_audio
    result = edit_audio("insert_background_event", "in.wav", label="Dog", timestamp="00:00:02.500")

CLI:

    python audio_edit/editor.py insert_background_event --audio_path in.wav --label Dog
    python audio_edit/editor.py --list
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audio_edit.audio_add_noise import (  # noqa: E402
    DEFAULT_NOISE_TYPE as ADD_NOISE_DEFAULT_TYPE,
    DEFAULT_SNR_DB as ADD_NOISE_DEFAULT_SNR,
    AddNoiseTool,
    add_noise,
)
from audio_edit.pad_noise import (  # noqa: E402
    DEFAULT_DURATION_SEC as PAD_DEFAULT_DURATION,
    DEFAULT_FADE_MS as PAD_DEFAULT_FADE_MS,
    DEFAULT_NOISE_TYPE as PAD_DEFAULT_NOISE_TYPE,
    DEFAULT_POSITION as PAD_DEFAULT_POSITION,
    DEFAULT_SNR_DB as PAD_DEFAULT_SNR,
    PadNoiseTool,
    pad_noise,
)
from audio_edit.background_event import (  # noqa: E402
    DEFAULT_FADE_MS as EVENT_DEFAULT_FADE_MS,
    DEFAULT_LABEL as EVENT_DEFAULT_LABEL,
    DEFAULT_SNR_DB as EVENT_DEFAULT_SNR,
    InsertBackgroundEventTool,
    insert_background_event,
)
from tools import generate_tool_descriptions  # noqa: E402
from tools.pitch_time import (  # noqa: E402
    PITCH_SHIFT_STEPS,
    TIME_STRETCH_RATES,
    PitchShiftTool,
    TimeStretchTool,
)

# Reuse the same fixed offsets the dataset-synthesis pipeline uses elsewhere in
# this repo (tools/pitch_time.py), so "pick a default" and "pick a ground-truth
# augmentation strength" stay in sync.
DEFAULT_PITCH_STEPS = PITCH_SHIFT_STEPS
DEFAULT_STRETCH_RATES = TIME_STRETCH_RATES


def change_pitch(
    audio_path: str,
    output_path: Optional[str] = None,
    *,
    n_steps: Optional[float] = None,
    bins_per_octave: int = 12,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Pitch-shift by `n_steps` semitones (delegates to `tools.pitch_time.PitchShiftTool`)."""
    if n_steps is None:
        n_steps = random.Random(seed).choice(DEFAULT_PITCH_STEPS)

    params: Dict[str, Any] = {
        "audio_path": str(audio_path),
        "n_steps": n_steps,
        "bins_per_octave": bins_per_octave,
    }
    if output_path:
        params["output_path"] = str(output_path)
    return PitchShiftTool.execute(params)


def change_speed(
    audio_path: str,
    output_path: Optional[str] = None,
    *,
    rate: Optional[float] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Time-stretch by `rate` (>1 speeds up, <1 slows down); delegates to `tools.pitch_time.TimeStretchTool`."""
    if rate is None:
        rate = random.Random(seed).choice(DEFAULT_STRETCH_RATES)

    params: Dict[str, Any] = {"audio_path": str(audio_path), "rate": rate}
    if output_path:
        params["output_path"] = str(output_path)
    return TimeStretchTool.execute(params)


OPERATIONS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "add_noise": add_noise,
    "pad_noise": pad_noise,
    "insert_background_event": insert_background_event,
    "change_speed": change_speed,
    "change_pitch": change_pitch,
}

_DESCRIBED_TOOLS = [
    AddNoiseTool,
    PadNoiseTool,
    InsertBackgroundEventTool,
    TimeStretchTool,
    PitchShiftTool,
]


def describe_operations() -> str:
    """Human-readable description of every operation and its parameters/defaults."""
    return generate_tool_descriptions(_DESCRIBED_TOOLS)


def edit_audio(operation: str, audio_path: str, output_path: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
    """Unified entry point: dispatch `operation` on `audio_path` with keyword overrides.

    All parameters beyond `operation`/`audio_path`/`output_path` are optional --
    each operation fills in a documented default for anything not passed.
    """
    if operation not in OPERATIONS:
        raise ValueError(f"Unknown operation '{operation}'. Available: {sorted(OPERATIONS)}")
    return OPERATIONS[operation](audio_path, output_path, **kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified audio-editing CLI: noise, padding, background events, speed, pitch.",
    )
    parser.add_argument("--list", action="store_true", help="List available operations and their parameters, then exit.")
    subparsers = parser.add_subparsers(dest="operation")

    p_add_noise = subparsers.add_parser("add_noise", help=AddNoiseTool.description())
    p_add_noise.add_argument("--audio_path", required=True)
    p_add_noise.add_argument("--output_path", default=None)
    p_add_noise.add_argument("--noise_type", choices=["white", "pink"], default=ADD_NOISE_DEFAULT_TYPE)
    p_add_noise.add_argument("--snr_db", type=float, default=ADD_NOISE_DEFAULT_SNR)
    p_add_noise.add_argument("--seed", type=int, default=None)

    p_pad_noise = subparsers.add_parser("pad_noise", help=PadNoiseTool.description())
    p_pad_noise.add_argument("--audio_path", required=True)
    p_pad_noise.add_argument("--output_path", default=None)
    p_pad_noise.add_argument("--position", choices=["start", "end", "both"], default=PAD_DEFAULT_POSITION)
    p_pad_noise.add_argument("--duration_sec", type=float, default=PAD_DEFAULT_DURATION)
    p_pad_noise.add_argument("--noise_type", choices=["white", "pink"], default=PAD_DEFAULT_NOISE_TYPE)
    p_pad_noise.add_argument("--snr_db", type=float, default=PAD_DEFAULT_SNR)
    p_pad_noise.add_argument("--fade_ms", type=float, default=PAD_DEFAULT_FADE_MS)
    p_pad_noise.add_argument("--seed", type=int, default=None)

    p_event = subparsers.add_parser("insert_background_event", help=InsertBackgroundEventTool.description())
    p_event.add_argument("--audio_path", required=True)
    p_event.add_argument("--output_path", default=None)
    p_event.add_argument("--label", default=EVENT_DEFAULT_LABEL)
    p_event.add_argument("--timestamp", default=None, help="HH:MM:SS.mmm onset; random if omitted")
    p_event.add_argument("--duration", type=float, default=None)
    p_event.add_argument("--snr_db", type=float, default=EVENT_DEFAULT_SNR)
    p_event.add_argument("--fade_ms", type=float, default=EVENT_DEFAULT_FADE_MS)
    p_event.add_argument("--audio_dir", default=None)
    p_event.add_argument("--metadata_path", default=None)
    p_event.add_argument("--seed", type=int, default=None)

    p_speed = subparsers.add_parser("change_speed", help=TimeStretchTool.description())
    p_speed.add_argument("--audio_path", required=True)
    p_speed.add_argument("--output_path", default=None)
    p_speed.add_argument("--rate", type=float, default=None, help=f"Default: random.choice({DEFAULT_STRETCH_RATES})")
    p_speed.add_argument("--seed", type=int, default=None)

    p_pitch = subparsers.add_parser("change_pitch", help=PitchShiftTool.description())
    p_pitch.add_argument("--audio_path", required=True)
    p_pitch.add_argument("--output_path", default=None)
    p_pitch.add_argument("--n_steps", type=float, default=None, help=f"Default: random.choice({DEFAULT_PITCH_STEPS})")
    p_pitch.add_argument("--bins_per_octave", type=int, default=12)
    p_pitch.add_argument("--seed", type=int, default=None)

    return parser


def main(argv: Optional[list] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list or not args.operation:
        print(describe_operations())
        if not args.operation:
            return
        return

    kwargs = {k: v for k, v in vars(args).items() if k not in ("operation", "audio_path", "output_path", "list")}
    result = edit_audio(args.operation, args.audio_path, args.output_path, **kwargs)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
