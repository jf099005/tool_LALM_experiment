"""Extend a clip with noise padding at the start, end, or both (does not overlay
existing audio -- it lengthens the clip). Useful for training models to be robust
to leading/trailing ambient noise instead of hard digital silence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audio_edit._audio_utils import (  # noqa: E402
    AudioEditError,
    db_to_ratio,
    default_output_path,
    fade_in_out,
    generate_noise,
    load_audio,
    rms,
    save_audio,
)
from tools.abstract_tool import Tool, ToolValidationError  # noqa: E402

DEFAULT_POSITION = "both"
DEFAULT_DURATION_SEC = 0.5
DEFAULT_NOISE_TYPE = "white"
# Padding noise sits well below the clip's own level so it reads as ambient
# room tone rather than an audible event, per the "silence padding with
# low-level noise" convention used in ASR robustness augmentation recipes.
DEFAULT_SNR_DB = 20.0
DEFAULT_FADE_MS = 10.0


def pad_noise(
    audio_path: str,
    output_path: Optional[str] = None,
    *,
    position: str = DEFAULT_POSITION,
    duration_sec: float = DEFAULT_DURATION_SEC,
    noise_type: str = DEFAULT_NOISE_TYPE,
    snr_db: float = DEFAULT_SNR_DB,
    fade_ms: float = DEFAULT_FADE_MS,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Prepend/append `duration_sec` of noise, quiet relative to the clip by `snr_db`.

    `position` is one of "start", "end", "both". The pad is faded into the real
    audio (`fade_ms` raised-cosine ramp) to avoid a boundary click.
    """
    if position not in ("start", "end", "both"):
        raise AudioEditError(f"position must be one of 'start', 'end', 'both', got '{position}'")
    if duration_sec <= 0:
        raise AudioEditError("duration_sec must be greater than 0.")

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise AudioEditError(f"Audio file not found: {audio_path}")

    audio, sr = load_audio(audio_path, sr=None, mono=True)
    rng = np.random.default_rng(seed)
    ref_rms = rms(audio)
    pad_len = int(sr * duration_sec)

    def _make_pad() -> np.ndarray:
        pad = generate_noise(pad_len, noise_type=noise_type, rng=rng)
        target_rms = ref_rms / db_to_ratio(snr_db) if ref_rms > 0 else 0.01
        pad = pad * target_rms
        return fade_in_out(pad, sr, fade_ms)

    segments = []
    if position in ("start", "both"):
        segments.append(_make_pad())
    segments.append(audio)
    if position in ("end", "both"):
        segments.append(_make_pad())

    padded = np.concatenate(segments)

    out_path = Path(output_path) if output_path else default_output_path(audio_path, "pad_noise")
    save_audio(out_path, padded, sr)

    return {
        "audio_path": str(audio_path),
        "position": position,
        "duration_sec": duration_sec,
        "noise_type": noise_type,
        "snr_db": snr_db,
        "fade_ms": fade_ms,
        "status": "success",
        "output_path": str(out_path),
        "message": "Noise padding completed successfully.",
    }


class PadNoiseTool(Tool):
    @classmethod
    def name(cls) -> str:
        return "pad_noise"

    @classmethod
    def description(cls) -> str:
        return (
            "Extend a WAV/MP3 clip with low-level noise padding at the start, end, or "
            "both, fading into the real audio to avoid a click. Requires audio_path; "
            "position, duration_sec, noise_type, snr_db, and fade_ms all have defaults."
        )

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"},
                "position": {
                    "type": "string",
                    "enum": ["start", "end", "both"],
                    "description": f"Where to add padding (default '{DEFAULT_POSITION}').",
                },
                "duration_sec": {
                    "type": "number",
                    "description": f"Padding length per side, in seconds (default {DEFAULT_DURATION_SEC}).",
                },
                "noise_type": {"type": "string", "enum": ["white", "pink"]},
                "snr_db": {
                    "type": "number",
                    "description": f"How much quieter the pad is than the clip, in dB (default {DEFAULT_SNR_DB}).",
                },
                "fade_ms": {
                    "type": "number",
                    "description": f"Fade duration at the pad/audio boundary, in ms (default {DEFAULT_FADE_MS}).",
                },
                "output_path": {"type": "string"},
            },
            "required": ["audio_path"],
        }

    @classmethod
    def execute(cls, parameters: Dict[str, Any]) -> Dict[str, Any]:
        cls.validate_parameters(parameters)
        try:
            return pad_noise(
                parameters["audio_path"],
                parameters.get("output_path"),
                position=parameters.get("position", DEFAULT_POSITION),
                duration_sec=float(parameters.get("duration_sec", DEFAULT_DURATION_SEC)),
                noise_type=parameters.get("noise_type", DEFAULT_NOISE_TYPE),
                snr_db=float(parameters.get("snr_db", DEFAULT_SNR_DB)),
                fade_ms=float(parameters.get("fade_ms", DEFAULT_FADE_MS)),
            )
        except AudioEditError as exc:
            raise ToolValidationError(str(exc)) from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pad an audio clip with noise at the start/end/both.")
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--position", choices=["start", "end", "both"], default=DEFAULT_POSITION)
    parser.add_argument("--duration_sec", type=float, default=DEFAULT_DURATION_SEC)
    parser.add_argument("--noise_type", choices=["white", "pink"], default=DEFAULT_NOISE_TYPE)
    parser.add_argument("--snr_db", type=float, default=DEFAULT_SNR_DB)
    parser.add_argument("--fade_ms", type=float, default=DEFAULT_FADE_MS)
    args = parser.parse_args()

    result = pad_noise(
        args.input_file,
        args.output_file,
        position=args.position,
        duration_sec=args.duration_sec,
        noise_type=args.noise_type,
        snr_db=args.snr_db,
        fade_ms=args.fade_ms,
    )
    print(f"處理完成！已儲存至: {result['output_path']}")
