"""Add synthetic (white/pink) noise across an entire clip at a target SNR."""

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
    default_output_path,
    generate_noise,
    load_audio,
    save_audio,
    scale_to_snr,
)
from tools.abstract_tool import Tool, ToolValidationError  # noqa: E402

# Defaults follow common augmentation-library practice (e.g. audiomentations'
# AddGaussianSNR, Kaldi/MUSAN's babble-noise recipe): white noise mixed in at a
# moderate SNR so it is audible but does not swamp the signal.
DEFAULT_NOISE_TYPE = "white"
DEFAULT_SNR_DB = 15.0


def add_gaussian_noise(file_path: str, output_path: str, noise_factor: float = 0.005) -> str:
    """Legacy amplitude-scaled Gaussian noise helper (kept for backward compatibility)."""
    audio, sr = load_audio(Path(file_path), sr=None, mono=True)
    noise = np.random.randn(len(audio)).astype(np.float32)
    augmented = np.clip(audio + noise_factor * noise, -1.0, 1.0)
    save_audio(Path(output_path), augmented, sr, ceiling=1.0)
    return output_path


def add_noise(
    audio_path: str,
    output_path: Optional[str] = None,
    *,
    noise_type: str = DEFAULT_NOISE_TYPE,
    snr_db: float = DEFAULT_SNR_DB,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Mix synthetic noise into the whole clip at a target signal-to-noise ratio (dB).

    Defaults (white noise, 15 dB SNR) are reasonable for general augmentation;
    override `noise_type` ("white"/"pink") and `snr_db` for other regimes.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise AudioEditError(f"Audio file not found: {audio_path}")

    audio, sr = load_audio(audio_path, sr=None, mono=True)
    rng = np.random.default_rng(seed)
    noise = generate_noise(audio.shape[-1], noise_type=noise_type, rng=rng)
    noise = scale_to_snr(audio, noise, snr_db)
    mixed = audio + noise

    out_path = Path(output_path) if output_path else default_output_path(audio_path, "add_noise")
    save_audio(out_path, mixed, sr)

    return {
        "audio_path": str(audio_path),
        "noise_type": noise_type,
        "snr_db": snr_db,
        "status": "success",
        "output_path": str(out_path),
        "message": "Noise addition completed successfully.",
    }


class AddNoiseTool(Tool):
    @classmethod
    def name(cls) -> str:
        return "add_noise"

    @classmethod
    def description(cls) -> str:
        return (
            "Mix synthetic white or pink noise into an entire WAV/MP3 clip at a target "
            "signal-to-noise ratio (dB). Requires audio_path; noise_type and snr_db have "
            "sensible defaults."
        )

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"},
                "noise_type": {
                    "type": "string",
                    "enum": ["white", "pink"],
                    "description": f"Noise color (default '{DEFAULT_NOISE_TYPE}').",
                },
                "snr_db": {
                    "type": "number",
                    "description": f"Target signal-to-noise ratio in dB (default {DEFAULT_SNR_DB}).",
                },
                "output_path": {"type": "string"},
            },
            "required": ["audio_path"],
        }

    @classmethod
    def execute(cls, parameters: Dict[str, Any]) -> Dict[str, Any]:
        cls.validate_parameters(parameters)
        try:
            return add_noise(
                parameters["audio_path"],
                parameters.get("output_path"),
                noise_type=parameters.get("noise_type", DEFAULT_NOISE_TYPE),
                snr_db=float(parameters.get("snr_db", DEFAULT_SNR_DB)),
            )
        except AudioEditError as exc:
            raise ToolValidationError(str(exc)) from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mix synthetic noise into an audio file at a target SNR.")
    parser.add_argument("input_file", type=str, help="Path to the input audio file")
    parser.add_argument("output_file", type=str, help="Path to write the noisy audio file")
    parser.add_argument("--noise_type", choices=["white", "pink"], default=DEFAULT_NOISE_TYPE)
    parser.add_argument("--snr_db", type=float, default=DEFAULT_SNR_DB, help=f"Target SNR in dB (default {DEFAULT_SNR_DB})")
    args = parser.parse_args()

    result = add_noise(args.input_file, args.output_file, noise_type=args.noise_type, snr_db=args.snr_db)
    print(f"處理完成！已儲存至: {result['output_path']}")
