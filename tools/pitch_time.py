from __future__ import annotations

import wave
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from .abstract_tool import Tool, ToolValidationError

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load a mono audio signal from a WAV or MP3 (or any librosa-supported) file."""
    if librosa is None:
        raise ToolValidationError(
            "librosa is required to load audio files. Install it with `pip install librosa`."
        )

    try:
        audio, sample_rate = librosa.load(str(path), sr=None, mono=True)
    except Exception as exc:
        raise ToolValidationError(f"Failed to open audio file '{path}': {exc}") from exc

    audio = np.asarray(audio, dtype=np.float32)
    if audio.shape[0] == 0:
        raise ToolValidationError(f"Audio file '{path}' appears to be empty.")

    return audio, sample_rate


def _save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak

    samples = np.clip(audio, -1.0, 1.0)
    int_samples = (samples * np.iinfo(np.int16).max).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int_samples.tobytes())


class PitchShiftTool(Tool):
    @classmethod
    def name(cls) -> str:
        return "pitch_shift"

    @classmethod
    def description(cls) -> str:
        return (
            "Shift the pitch of an audio file (WAV or MP3) by a number of semitones "
            "without changing its duration, using librosa. Requires audio_path and "
            "n_steps (semitones; positive shifts up, negative shifts down)."
        )

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"},
                "n_steps": {
                    "type": "number",
                    "description": "Number of semitones to shift. Positive values shift up, negative values shift down.",
                },
                "bins_per_octave": {
                    "type": "integer",
                    "description": "Number of steps per octave (default 12).",
                },
                "output_path": {"type": "string"},
            },
            "required": ["audio_path", "n_steps"],
        }

    @classmethod
    def execute(cls, parameters: Dict[str, Any]) -> Dict[str, Any]:
        cls.validate_parameters(parameters)

        if librosa is None:
            raise ToolValidationError(
                "librosa is required for pitch shifting. Install it with `pip install librosa`."
            )

        audio_path = parameters["audio_path"]
        n_steps = float(parameters["n_steps"])
        bins_per_octave = int(parameters.get("bins_per_octave", 12))
        output_path = parameters.get("output_path")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise ToolValidationError(f"Audio file not found: {audio_path}")

        audio, sr = _load_audio(audio_path)
        shifted = librosa.effects.pitch_shift(
            y=audio, sr=sr, n_steps=n_steps, bins_per_octave=bins_per_octave
        )

        if output_path is None:
            output_path = audio_path.parent / f"{audio_path.stem}_{cls.name()}.wav"
        else:
            output_path = Path(output_path)

        _save_wav(output_path, shifted, sr)

        return {
            "audio_path": str(audio_path),
            "n_steps": n_steps,
            "bins_per_octave": bins_per_octave,
            "status": "success",
            "output_path": str(output_path),
            "message": "Pitch shift completed successfully.",
        }

    @classmethod
    def execute_batch(cls, batch_parameters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(batch_parameters, list):
            raise ToolValidationError("Batch parameters must be a list of parameter dictionaries.")

        results: List[Dict[str, Any]] = []
        for parameters in tqdm(batch_parameters):
            if not isinstance(parameters, dict):
                raise ToolValidationError("Each batch item must be a parameter dictionary.")
            try:
                results.append(cls.execute(parameters))
            except Exception as exc:
                results.append({
                    "audio_path": parameters.get("audio_path"),
                    "status": "failure",
                    "output_path": None,
                    "message": str(exc),
                })

        return results


class TimeStretchTool(Tool):
    @classmethod
    def name(cls) -> str:
        return "time_stretch"

    @classmethod
    def description(cls) -> str:
        return (
            "Time-stretch an audio file (WAV or MP3) by a fixed rate without changing "
            "its pitch, using librosa. Requires audio_path and rate (rate > 1 speeds "
            "up/shortens, rate < 1 slows down/lengthens)."
        )

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"},
                "rate": {
                    "type": "number",
                    "description": "Stretch factor. Values > 1 speed up (shorten); values < 1 slow down (lengthen).",
                },
                "output_path": {"type": "string"},
            },
            "required": ["audio_path", "rate"],
        }

    @classmethod
    def execute(cls, parameters: Dict[str, Any]) -> Dict[str, Any]:
        cls.validate_parameters(parameters)

        if librosa is None:
            raise ToolValidationError(
                "librosa is required for time stretching. Install it with `pip install librosa`."
            )

        audio_path = parameters["audio_path"]
        rate = float(parameters["rate"])
        output_path = parameters.get("output_path")

        if rate <= 0:
            raise ToolValidationError("rate must be greater than 0.")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise ToolValidationError(f"Audio file not found: {audio_path}")

        audio, sr = _load_audio(audio_path)
        stretched = librosa.effects.time_stretch(y=audio, rate=rate)

        if output_path is None:
            output_path = audio_path.parent / f"{audio_path.stem}_{cls.name()}.wav"
        else:
            output_path = Path(output_path)

        _save_wav(output_path, stretched, sr)

        return {
            "audio_path": str(audio_path),
            "rate": rate,
            "status": "success",
            "output_path": str(output_path),
            "message": "Time stretch completed successfully.",
        }

    @classmethod
    def execute_batch(cls, batch_parameters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(batch_parameters, list):
            raise ToolValidationError("Batch parameters must be a list of parameter dictionaries.")

        results: List[Dict[str, Any]] = []
        for parameters in tqdm(batch_parameters):
            if not isinstance(parameters, dict):
                raise ToolValidationError("Each batch item must be a parameter dictionary.")
            try:
                results.append(cls.execute(parameters))
            except Exception as exc:
                results.append({
                    "audio_path": parameters.get("audio_path"),
                    "status": "failure",
                    "output_path": None,
                    "message": str(exc),
                })

        return results
