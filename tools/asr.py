from __future__ import annotations

import wave
from tqdm import tqdm

from pathlib import Path
from typing import Any, Dict, List, Optional

from .abstract_tool import Tool, ToolValidationError

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import whisper
except ImportError:  # pragma: no cover
    whisper = None

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None


class ASRTool(Tool):
    _model: Optional[Any] = None

    @classmethod
    def name(cls) -> str:
        return "asr"

    @classmethod
    def description(cls) -> str:
        return (
            "Transcribe speech from a WAV audio segment into text. "
            "Requires audio_path plus HH:MM:SS.mmm timestamps for audio_begin and audio_end, "
            "with optional language selection."
        )

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"},
                "audio_begin": {
                    "type": "string",
                    "format": "HH:MM:SS.mmm",
                },
                "audio_end": {
                    "type": "string",
                    "format": "HH:MM:SS.mmm",
                },
                "language": {"type": "string"},
            },
            "required": ["audio_path", "audio_begin", "audio_end"],
        }

    @classmethod
    def execute(cls, parameters: Dict[str, Any]) -> Dict[str, Any]:
        cls.validate_parameters(parameters)

        if whisper is None:
            raise ToolValidationError(
                "Whisper is required for ASR. Install it with `pip install -U openai-whisper`."
            )

        if np is None:
            raise ToolValidationError(
                "NumPy is required for ASR. Install it with `pip install numpy`."
            )

        model = cls._get_model()
        return cls._execute_single(parameters, model)

    @classmethod
    def execute_batch(cls, batch_parameters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(batch_parameters, list):
            raise ToolValidationError("Batch parameters must be a list of parameter dictionaries.")

        if whisper is None:
            raise ToolValidationError(
                "Whisper is required for ASR. Install it with `pip install -U openai-whisper`."
            )

        if np is None:
            raise ToolValidationError(
                "NumPy is required for ASR. Install it with `pip install numpy`."
            )

        model = cls._get_model()
        results: List[Dict[str, Any]] = []

        for parameters in tqdm(batch_parameters, desc="Processing batches"):
            if not isinstance(parameters, dict):
                raise ToolValidationError("Each batch item must be a parameter dictionary.")

            try:
                cls.validate_parameters(parameters)
                results.append(cls._execute_single(parameters, model))
            except ToolValidationError as exc:
                results.append(
                    {
                        "audio_path": parameters.get("audio_path"),
                        "audio_begin": parameters.get("audio_begin"),
                        "audio_end": parameters.get("audio_end"),
                        "language": parameters.get("language"),
                        "status": "failure",
                        "transcript": None,
                        "raw_result": None,
                        "message": str(exc),
                    }
                )

        return results

    @classmethod
    def _execute_single(cls, parameters: Dict[str, Any], model: Any) -> Dict[str, Any]:
        if whisper is None:
            raise ToolValidationError(
                "Whisper is required for ASR. Install it with `pip install -U openai-whisper`."
            )

        if np is None:
            raise ToolValidationError(
                "NumPy is required for ASR. Install it with `pip install numpy`."
            )

        audio_path = Path(parameters["audio_path"])
        audio_begin = parameters["audio_begin"]
        audio_end = parameters["audio_end"]
        language = parameters.get("language") or "auto-detect"

        if not audio_path.exists():
            raise ToolValidationError(f"Audio file not found: {audio_path}")

        begin_seconds = cls._parse_timestamp(audio_begin)
        end_seconds = cls._parse_timestamp(audio_end)
        if end_seconds <= begin_seconds:
            raise ToolValidationError("audio_end must be greater than audio_begin.")

        if audio_path.suffix.lower() != ".wav":
            raise ToolValidationError("ASR currently supports only WAV audio files in this implementation.")

        audio_segment, sample_rate = cls._load_wav_segment(audio_path, begin_seconds, end_seconds)
        if audio_segment.size == 0:
            raise ToolValidationError("Requested audio segment is empty.")

        if audio_segment.ndim > 1:
            audio_segment = np.mean(audio_segment, axis=1)

        target_rate = getattr(whisper.audio, "SAMPLE_RATE", 16000) if hasattr(whisper, "audio") else 16000
        if sample_rate != target_rate:
            audio_segment = cls._resample_audio(audio_segment, sample_rate, target_rate)

        audio_segment = whisper.pad_or_trim(audio_segment)

        mel = whisper.log_mel_spectrogram(audio_segment).to(model.device)

        detected_language = None
        decode_kwargs: Dict[str, Any] = {}
        if language != "auto-detect":
            decode_kwargs["language"] = language
        else:
            _, probs = model.detect_language(mel)
            detected_language = max(probs, key=probs.get)

        options = whisper.DecodingOptions(**decode_kwargs)
        result = whisper.decode(model, mel, options)
        transcript = result.text.strip()

        raw_result: Dict[str, Any] = {"text": transcript}
        if detected_language is not None:
            raw_result["detected_language"] = detected_language

        return {
            "audio_path": str(audio_path),
            "audio_begin": audio_begin,
            "audio_end": audio_end,
            "language": language if language != "auto-detect" else detected_language,
            "status": "success",
            "transcript": transcript,
            "raw_result": raw_result,
            "message": "ASR operation transcribed the requested WAV segment successfully.",
        }

    @classmethod
    def _get_model(cls) -> Any:
        if cls._model is not None:
            return cls._model

        device = "cpu"
        if torch is not None and torch.cuda.is_available():
            device = "cuda"
        else:
            raise ToolValidationError(
                "CUDA-compatible GPU is required for ASR with Whisper. "
                "Please run on a machine with a compatible GPU and CUDA drivers installed."
            )

        cls._model = whisper.load_model("small", device=device)
        return cls._model

    @classmethod
    def _load_wav_segment(cls, path: Path, begin_seconds: float, end_seconds: float) -> tuple[Any, int]:
        try:
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

            dtype = cls._dtype_from_width(sampwidth)
            audio = np.frombuffer(frames, dtype=dtype)
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels)

            audio = audio.astype(np.float32)
            if np.issubdtype(dtype, np.integer):
                max_val = float(np.iinfo(dtype).max)
                audio = audio / max_val

            return audio, sample_rate
        except (wave.Error, OSError) as exc:
            return cls._load_audio_segment_with_fallback(path, begin_seconds, end_seconds, exc)

    @classmethod
    def _load_audio_segment_with_fallback(
        cls, path: Path, begin_seconds: float, end_seconds: float, original_exc: Exception
    ) -> tuple[Any, int]:
        if sf is not None:
            try:
                audio, sample_rate = sf.read(str(path), dtype="float32")
                audio = np.asarray(audio, dtype=np.float32)
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                start_frame = int(begin_seconds * sample_rate)
                end_frame = int(end_seconds * sample_rate)
                if start_frame < 0 or end_frame > len(audio):
                    raise ToolValidationError(
                        f"Requested clip range [{begin_seconds:.3f}, {end_seconds:.3f}] is out of bounds for file duration {len(audio) / sample_rate:.3f}s."
                    )
                return audio[start_frame:end_frame], sample_rate
            except Exception:
                pass

        if librosa is not None:
            try:
                audio, sample_rate = librosa.load(str(path), sr=None, mono=False)
                audio = np.asarray(audio, dtype=np.float32)
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=0)
                start_frame = int(begin_seconds * sample_rate)
                end_frame = int(end_seconds * sample_rate)
                if start_frame < 0 or end_frame > len(audio):
                    raise ToolValidationError(
                        f"Requested clip range [{begin_seconds:.3f}, {end_seconds:.3f}] is out of bounds for file duration {len(audio) / sample_rate:.3f}s."
                    )
                return audio[start_frame:end_frame], sample_rate
            except Exception:
                pass

        raise ToolValidationError(
            f"Failed to open WAV file '{path}' ({original_exc}). "
            "Install soundfile or librosa, or convert the file to a standard PCM WAV format."
        )

    @classmethod
    def _resample_audio(cls, audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
        if original_sr == target_sr:
            return audio

        original_length = audio.shape[0]
        target_length = int(round(original_length * target_sr / original_sr))
        if target_length <= 0:
            raise ToolValidationError("Audio segment is too short to resample.")

        original_indices = np.linspace(0, original_length - 1, num=original_length)
        target_indices = np.linspace(0, original_length - 1, num=target_length)
        audio = np.interp(target_indices, original_indices, audio).astype(np.float32)
        return audio

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
