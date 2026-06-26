from __future__ import annotations

import wave
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy import signal

from .abstract_tool import Tool, ToolValidationError

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None


class NormalizeToolBase(Tool):
    @staticmethod
    def _load_wav_segment(path: Path, begin_seconds: float, end_seconds: float) -> tuple[np.ndarray, int]:
        with wave.open(str(path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()

            start_frame = int(begin_seconds * sample_rate)
            end_frame = int(end_seconds * sample_rate)
            if start_frame < 0 or end_frame > n_frames:
                raise ToolValidationError(
                    f"Requested clip range [{begin_seconds:.3f}, {end_seconds:.3f}] is out of bounds for file duration {n_frames / sample_rate:.3f}s."
                )

            wav_file.setpos(start_frame)
            frames = wav_file.readframes(end_frame - start_frame)

        dtype = NormalizeToolBase._dtype_from_width(sampwidth)
        audio = np.frombuffer(frames, dtype=dtype)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)
            audio = np.mean(audio, axis=1)

        audio = audio.astype(np.float32)
        if np.issubdtype(dtype, np.integer):
            max_val = float(np.iinfo(dtype).max)
            audio = audio / max_val

        return audio, sample_rate

    @staticmethod
    def _dtype_from_width(width: int) -> Any:
        if width == 1:
            return np.int8
        if width == 2:
            return np.int16
        if width == 4:
            return np.int32
        raise ToolValidationError(f"Unsupported WAV sample width: {width}")

    @staticmethod
    def _parse_timestamp(value: str) -> float:
        try:
            time_text, millis_text = value.split(".")
            hours, minutes, seconds = [int(part) for part in time_text.split(":")]
            millis = int(millis_text.ljust(3, "0")[:3])
        except ValueError as exc:
            raise ToolValidationError(
                f"Invalid timestamp format '{value}'. Expected HH:MM:SS.mmm"
            ) from exc

        return hours * 3600 + minutes * 60 + seconds + millis / 1000.0

    @staticmethod
    def _ensure_wav_file(path: Path) -> None:
        if path.suffix.lower() != ".wav":
            raise ToolValidationError("This normalization tool supports only WAV audio files.")

    @classmethod
    def _build_output_path(cls, audio_path: Path, suffix: str, audio_begin: str, audio_end: str) -> Path:
        return (
            audio_path.parent
            / f"{audio_path.stem}_{suffix}_{audio_begin.replace(':', '-')}_{audio_end.replace(':', '-')}.wav"
        )

    @staticmethod
    def _save_wav(path: Path, wav: np.ndarray, sample_rate: int) -> None:
        peak = np.max(np.abs(wav))
        if peak > 1.0:
            wav = wav / peak

        samples = np.clip(wav, -1.0, 1.0)
        int_samples = (samples * np.iinfo(np.int16).max).astype(np.int16)

        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(int_samples.tobytes())


class AmplitudeNormalizeTool(NormalizeToolBase):
    @classmethod
    def name(cls) -> str:
        return "amplitude_normalize"

    @classmethod
    def description(cls) -> str:
        return (
            "Normalize the amplitude of a WAV audio segment to a consistent peak or RMS level. "
            "Requires audio_path and HH:MM:SS.mmm timestamps for audio_begin and audio_end."
        )

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"},
                "audio_begin": {"type": "string", "format": "HH:MM:SS.mmm"},
                "audio_end": {"type": "string", "format": "HH:MM:SS.mmm"},
                "target_level": {"type": "number"},
                "method": {"type": "string", "enum": ["peak", "rms"]},
                "output_path": {"type": "string"},
            },
            "required": ["audio_path", "audio_begin", "audio_end"],
        }

    @classmethod
    def execute(cls, parameters: Dict[str, Any]) -> Dict[str, Any]:
        cls.validate_parameters(parameters)

        audio_path = parameters["audio_path"]
        audio_begin = parameters["audio_begin"]
        audio_end = parameters["audio_end"]
        target_level = float(parameters.get("target_level", 0.9))
        method = parameters.get("method", "peak")
        output_path = parameters.get("output_path")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise ToolValidationError(f"Audio file not found: {audio_path}")
        cls._ensure_wav_file(audio_path)

        begin_seconds = cls._parse_timestamp(audio_begin)
        end_seconds = cls._parse_timestamp(audio_end)
        if end_seconds <= begin_seconds:
            raise ToolValidationError("audio_end must be greater than audio_begin.")

        audio, sr = cls._load_wav_segment(audio_path, begin_seconds, end_seconds)
        normalized = cls._normalize(audio, target_level=target_level, method=method)

        if output_path is None:
            output_path = cls._build_output_path(audio_path, cls.name(), audio_begin, audio_end)
        else:
            output_path = Path(output_path)

        cls._save_wav(output_path, normalized, sr)

        return {
            "audio_path": str(audio_path),
            "audio_begin": audio_begin,
            "audio_end": audio_end,
            "target_level": target_level,
            "method": method,
            "status": "success",
            "output_path": str(output_path),
            "message": "Amplitude normalization completed successfully.",
        }

    @staticmethod
    def _normalize(audio: np.ndarray, target_level: float, method: str) -> np.ndarray:
        if method == "peak":
            max_val = np.max(np.abs(audio))
            return audio / max_val * target_level if max_val > 0 else audio

        if method == "rms":
            rms = np.sqrt(np.mean(audio ** 2))
            return audio / rms * (target_level / np.sqrt(2)) if rms > 0 else audio

        raise ToolValidationError(f"Unknown normalization method: {method}")


class LoudnessNormalizeTool(NormalizeToolBase):
    @classmethod
    def name(cls) -> str:
        return "loudness_normalize"

    @classmethod
    def description(cls) -> str:
        return (
            "Normalize a WAV audio segment to a target loudness level in LUFS. "
            "Requires audio_path and HH:MM:SS.mmm timestamps for audio_begin and audio_end."
        )

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"},
                "audio_begin": {"type": "string", "format": "HH:MM:SS.mmm"},
                "audio_end": {"type": "string", "format": "HH:MM:SS.mmm"},
                "target_lufs": {"type": "number"},
                "output_path": {"type": "string"},
            },
            "required": ["audio_path", "audio_begin", "audio_end"],
        }

    @classmethod
    def execute(cls, parameters: Dict[str, Any]) -> Dict[str, Any]:
        cls.validate_parameters(parameters)

        audio_path = parameters["audio_path"]
        audio_begin = parameters["audio_begin"]
        audio_end = parameters["audio_end"]
        target_lufs = float(parameters.get("target_lufs", -23.0))
        output_path = parameters.get("output_path")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise ToolValidationError(f"Audio file not found: {audio_path}")
        cls._ensure_wav_file(audio_path)

        begin_seconds = cls._parse_timestamp(audio_begin)
        end_seconds = cls._parse_timestamp(audio_end)
        if end_seconds <= begin_seconds:
            raise ToolValidationError("audio_end must be greater than audio_begin.")

        audio, sr = cls._load_wav_segment(audio_path, begin_seconds, end_seconds)
        normalized = cls._loudness_normalize(audio, sr, target_lufs=target_lufs)

        if output_path is None:
            output_path = cls._build_output_path(audio_path, cls.name(), audio_begin, audio_end)
        else:
            output_path = Path(output_path)

        cls._save_wav(output_path, normalized, sr)

        return {
            "audio_path": str(audio_path),
            "audio_begin": audio_begin,
            "audio_end": audio_end,
            "target_lufs": target_lufs,
            "status": "success",
            "output_path": str(output_path),
            "message": "Loudness normalization completed successfully.",
        }

    @staticmethod
    def _loudness_normalize(audio: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
        sos = signal.butter(2, [20, min(20000, sr / 2 - 1)], "bandpass", fs=sr, output="sos")
        weighted = signal.sosfilt(sos, audio)
        rms = np.sqrt(np.mean(weighted ** 2))

        if rms <= 0:
            return audio

        current_lufs = -23.0 + 20.0 * np.log10(rms / 0.1)
        gain_db = target_lufs - current_lufs
        gain_linear = 10.0 ** (gain_db / 20.0)
        normalized = audio * gain_linear
        max_val = np.max(np.abs(normalized))
        if max_val > 0.95:
            normalized = normalized / max_val * 0.95

        return normalized


class DCOffsetRemovalTool(NormalizeToolBase):
    @classmethod
    def name(cls) -> str:
        return "remove_dc_offset"

    @classmethod
    def description(cls) -> str:
        return (
            "Remove DC bias from a WAV audio segment. Requires audio_path and HH:MM:SS.mmm timestamps "
            "for audio_begin and audio_end."
        )

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"},
                "audio_begin": {"type": "string", "format": "HH:MM:SS.mmm"},
                "audio_end": {"type": "string", "format": "HH:MM:SS.mmm"},
                "output_path": {"type": "string"},
            },
            "required": ["audio_path", "audio_begin", "audio_end"],
        }

    @classmethod
    def execute(cls, parameters: Dict[str, Any]) -> Dict[str, Any]:
        cls.validate_parameters(parameters)

        audio_path = parameters["audio_path"]
        audio_begin = parameters["audio_begin"]
        audio_end = parameters["audio_end"]
        output_path = parameters.get("output_path")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise ToolValidationError(f"Audio file not found: {audio_path}")
        cls._ensure_wav_file(audio_path)

        begin_seconds = cls._parse_timestamp(audio_begin)
        end_seconds = cls._parse_timestamp(audio_end)
        if end_seconds <= begin_seconds:
            raise ToolValidationError("audio_end must be greater than audio_begin.")

        audio, sr = cls._load_wav_segment(audio_path, begin_seconds, end_seconds)
        normalized = cls._remove_dc_offset(audio)

        if output_path is None:
            output_path = cls._build_output_path(audio_path, cls.name(), audio_begin, audio_end)
        else:
            output_path = Path(output_path)

        cls._save_wav(output_path, normalized, sr)

        return {
            "audio_path": str(audio_path),
            "audio_begin": audio_begin,
            "audio_end": audio_end,
            "status": "success",
            "output_path": str(output_path),
            "message": "DC offset removal completed successfully.",
        }

    @staticmethod
    def _remove_dc_offset(audio: np.ndarray) -> np.ndarray:
        return audio - np.mean(audio)


class SpectralNormalizeTool(NormalizeToolBase):
    @classmethod
    def name(cls) -> str:
        return "spectral_normalize"

    @classmethod
    def description(cls) -> str:
        return (
            "Adjust the spectral energy distribution of a WAV audio segment to make frequency band energy more balanced. "
            "Requires audio_path and HH:MM:SS.mmm timestamps for audio_begin and audio_end."
        )

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"},
                "audio_begin": {"type": "string", "format": "HH:MM:SS.mmm"},
                "audio_end": {"type": "string", "format": "HH:MM:SS.mmm"},
                "strength": {"type": "number"},
                "output_path": {"type": "string"},
            },
            "required": ["audio_path", "audio_begin", "audio_end"],
        }

    @classmethod
    def execute(cls, parameters: Dict[str, Any]) -> Dict[str, Any]:
        cls.validate_parameters(parameters)

        if librosa is None:
            raise ToolValidationError(
                "librosa is required for spectral normalization. Install it with `pip install librosa`."
            )

        audio_path = parameters["audio_path"]
        audio_begin = parameters["audio_begin"]
        audio_end = parameters["audio_end"]
        strength = float(parameters.get("strength", 0.5))
        output_path = parameters.get("output_path")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise ToolValidationError(f"Audio file not found: {audio_path}")
        cls._ensure_wav_file(audio_path)

        begin_seconds = cls._parse_timestamp(audio_begin)
        end_seconds = cls._parse_timestamp(audio_end)
        if end_seconds <= begin_seconds:
            raise ToolValidationError("audio_end must be greater than audio_begin.")

        audio, sr = cls._load_wav_segment(audio_path, begin_seconds, end_seconds)
        normalized = cls._spectral_normalize(audio, sr, strength=strength)

        if output_path is None:
            output_path = cls._build_output_path(audio_path, cls.name(), audio_begin, audio_end)
        else:
            output_path = Path(output_path)

        cls._save_wav(output_path, normalized, sr)

        return {
            "audio_path": str(audio_path),
            "audio_begin": audio_begin,
            "audio_end": audio_end,
            "strength": strength,
            "status": "success",
            "output_path": str(output_path),
            "message": "Spectral normalization completed successfully.",
        }

    @staticmethod
    def _spectral_normalize(audio: np.ndarray, sr: int, strength: float = 0.5) -> np.ndarray:
        n_fft = 2048
        hop_length = 512
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(D)
        phase = np.angle(D)

        mean_energy = np.mean(magnitude, axis=1, keepdims=True) + 1e-10
        target_energy = np.median(mean_energy)
        normalization_factor = (target_energy / mean_energy) ** strength

        magnitude_normalized = magnitude * normalization_factor
        D_normalized = magnitude_normalized * np.exp(1j * phase)
        audio_normalized = librosa.istft(D_normalized, hop_length=hop_length, length=len(audio))

        max_val = np.max(np.abs(audio_normalized))
        if max_val > 0:
            audio_normalized = audio_normalized / max_val * 0.95

        return audio_normalized


class TrimSilenceTool(NormalizeToolBase):
    @classmethod
    def name(cls) -> str:
        return "trim_silence"

    @classmethod
    def description(cls) -> str:
        return (
            "Trim silence from the beginning and end of a WAV audio segment. "
            "Requires audio_path and HH:MM:SS.mmm timestamps for audio_begin and audio_end."
        )

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"},
                "audio_begin": {"type": "string", "format": "HH:MM:SS.mmm"},
                "audio_end": {"type": "string", "format": "HH:MM:SS.mmm"},
                "threshold_db": {"type": "number"},
                "frame_length": {"type": "integer"},
                "hop_length": {"type": "integer"},
                "output_path": {"type": "string"},
            },
            "required": ["audio_path", "audio_begin", "audio_end"],
        }

    @classmethod
    def execute(cls, parameters: Dict[str, Any]) -> Dict[str, Any]:
        cls.validate_parameters(parameters)

        if librosa is None:
            raise ToolValidationError(
                "librosa is required for silence trimming. Install it with `pip install librosa`."
            )

        audio_path = parameters["audio_path"]
        audio_begin = parameters["audio_begin"]
        audio_end = parameters["audio_end"]
        threshold_db = float(parameters.get("threshold_db", 40.0))
        frame_length = int(parameters.get("frame_length", 2048))
        hop_length = int(parameters.get("hop_length", 512))
        output_path = parameters.get("output_path")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise ToolValidationError(f"Audio file not found: {audio_path}")
        cls._ensure_wav_file(audio_path)

        begin_seconds = cls._parse_timestamp(audio_begin)
        end_seconds = cls._parse_timestamp(audio_end)
        if end_seconds <= begin_seconds:
            raise ToolValidationError("audio_end must be greater than audio_begin.")

        audio, sr = cls._load_wav_segment(audio_path, begin_seconds, end_seconds)
        trimmed, _ = librosa.effects.trim(
            audio,
            top_db=threshold_db,
            frame_length=frame_length,
            hop_length=hop_length,
        )

        if output_path is None:
            output_path = cls._build_output_path(audio_path, cls.name(), audio_begin, audio_end)
        else:
            output_path = Path(output_path)

        cls._save_wav(output_path, trimmed, sr)

        return {
            "audio_path": str(audio_path),
            "audio_begin": audio_begin,
            "audio_end": audio_end,
            "threshold_db": threshold_db,
            "frame_length": frame_length,
            "hop_length": hop_length,
            "status": "success",
            "output_path": str(output_path),
            "message": "Silence trimming completed successfully.",
        }


class PreEmphasisTool(NormalizeToolBase):
    @classmethod
    def name(cls) -> str:
        return "pre_emphasis"

    @classmethod
    def description(cls) -> str:
        return (
            "Apply a pre-emphasis filter to a WAV audio segment to boost high-frequency content and improve spectral balance. "
            "Requires audio_path and HH:MM:SS.mmm timestamps for audio_begin and audio_end."
        )

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"},
                "audio_begin": {"type": "string", "format": "HH:MM:SS.mmm"},
                "audio_end": {"type": "string", "format": "HH:MM:SS.mmm"},
                "coef": {"type": "number"},
                "output_path": {"type": "string"},
            },
            "required": ["audio_path", "audio_begin", "audio_end"],
        }

    @classmethod
    def execute(cls, parameters: Dict[str, Any]) -> Dict[str, Any]:
        cls.validate_parameters(parameters)

        audio_path = parameters["audio_path"]
        audio_begin = parameters["audio_begin"]
        audio_end = parameters["audio_end"]
        coef = float(parameters.get("coef", 0.97))
        output_path = parameters.get("output_path")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise ToolValidationError(f"Audio file not found: {audio_path}")
        cls._ensure_wav_file(audio_path)

        begin_seconds = cls._parse_timestamp(audio_begin)
        end_seconds = cls._parse_timestamp(audio_end)
        if end_seconds <= begin_seconds:
            raise ToolValidationError("audio_end must be greater than audio_begin.")

        audio, sr = cls._load_wav_segment(audio_path, begin_seconds, end_seconds)
        emphasized = cls._pre_emphasis(audio, coef=coef)

        if output_path is None:
            output_path = cls._build_output_path(audio_path, cls.name(), audio_begin, audio_end)
        else:
            output_path = Path(output_path)

        cls._save_wav(output_path, emphasized, sr)

        return {
            "audio_path": str(audio_path),
            "audio_begin": audio_begin,
            "audio_end": audio_end,
            "coef": coef,
            "status": "success",
            "output_path": str(output_path),
            "message": "Pre-emphasis completed successfully.",
        }

    @staticmethod
    def _pre_emphasis(audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
        return np.append(audio[:1], audio[1:] - coef * audio[:-1])
