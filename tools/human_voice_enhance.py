from __future__ import annotations

from tqdm import tqdm
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .abstract_tool import Tool, ToolValidationError

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None

try:
    import torch
    from df.enhance import enhance, init_df
    from df.io import resample as df_resample
    _DEEPFILTER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DEEPFILTER_AVAILABLE = False

_DF_MODEL_CACHE = None


class HumanVoiceEnhanceTool(Tool):
    @classmethod
    def name(cls) -> str:
        return "human_voice_enhance"

    @classmethod
    def description(cls) -> str:
        return (
            "Enhance human voice in an audio file by reducing background noise using DeepFilterNet. "
            "Supports WAV, MP3, FLAC, OGG, AIFF, and other formats readable by soundfile or librosa."
        )

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"},
                "output_path": {"type": "string"},
            },
            "required": ["audio_path"],
        }

    @classmethod
    def execute(cls, parameters: Dict[str, Any]) -> Dict[str, Any]:
        cls.validate_parameters(parameters)

        if not _DEEPFILTER_AVAILABLE:
            raise ToolValidationError(
                "DeepFilterNet is not installed. Install it with: pip install deepfilternet"
            )

        return cls._enhance_single(parameters, cls._get_model())

    @classmethod
    def execute_batch(cls, batch_parameters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(batch_parameters, list):
            raise ToolValidationError("Batch parameters must be a list of parameter dictionaries.")

        if not _DEEPFILTER_AVAILABLE:
            raise ToolValidationError(
                "DeepFilterNet is not installed. Install it with: pip install deepfilternet"
            )

        model_tuple = cls._get_model()

        results: List[Dict[str, Any]] = []
        for parameters in tqdm(batch_parameters):
            if not isinstance(parameters, dict):
                raise ToolValidationError("Each batch item must be a parameter dictionary.")
            try:
                results.append(cls._enhance_single(parameters, model_tuple))
            except ToolValidationError as exc:
                results.append({
                    "audio_path": parameters.get("audio_path"),
                    "status": "failure",
                    "output_path": None,
                    "message": str(exc),
                })

        return results

    @classmethod
    def _enhance_single(cls, parameters: Dict[str, Any], model_tuple) -> Dict[str, Any]:
        audio_path = Path(parameters["audio_path"])
        atten_lim_db = parameters.get("atten_lim_db")
        output_path = parameters.get("output_path")

        if not audio_path.exists():
            raise ToolValidationError(f"Audio file not found: {audio_path}")

        audio, sample_rate = cls._load_audio(audio_path)
        if audio.size == 0:
            raise ToolValidationError("Requested audio segment is empty.")

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        model, df_state, _ = model_tuple
        df_sr = df_state.sr()

        audio_tensor = torch.as_tensor(audio, dtype=torch.float32).unsqueeze(0)
        if sample_rate != df_sr:
            audio_tensor = df_resample(audio_tensor, sample_rate, df_sr)

        enhanced = enhance(model, df_state, audio_tensor, pad=True, atten_lim_db=atten_lim_db)

        if sample_rate != df_sr:
            enhanced = df_resample(enhanced, df_sr, sample_rate)

        audio_out = np.clip(enhanced.squeeze(0).numpy(), -1.0, 1.0)

        if output_path is None:
            output_path = audio_path.parent / f"{audio_path.stem}_deepfilter.wav"
        else:
            output_path = Path(output_path)

        cls._save_wav(output_path, audio_out, sample_rate)

        return {
            "audio_path": str(audio_path),
            "status": "success",
            "output_path": str(output_path),
            "message": "Denoise operation completed using DeepFilterNet.",
        }

    @classmethod
    def _get_model(cls):
        global _DF_MODEL_CACHE
        if _DF_MODEL_CACHE is None:
            _DF_MODEL_CACHE = init_df(log_level="ERROR", log_file=None)
        return _DF_MODEL_CACHE

    @classmethod
    def _load_audio(cls, path: Path) -> tuple[np.ndarray, int]:
        # stdlib wave only handles PCM WAV — skip it for other formats
        if path.suffix.lower() == ".wav":
            try:
                return cls._load_via_wave(path)
            except ToolValidationError:
                raise
            except Exception:
                pass

        if sf is not None:
            try:
                return cls._load_via_soundfile(path)
            except ToolValidationError:
                raise
            except Exception:
                pass

        if librosa is not None:
            try:
                return cls._load_via_librosa(path)
            except ToolValidationError:
                raise
            except Exception:
                pass

        raise ToolValidationError(
            f"Failed to open audio file '{path}'. "
            "Install soundfile (for WAV/FLAC/OGG/AIFF) or librosa (for MP3 and more), "
            "or convert the file to standard PCM WAV format."
        )

    @classmethod
    def _load_via_wave(cls, path: Path) -> tuple[np.ndarray, int]:
        with wave.open(str(path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            frames = wav_file.readframes(n_frames)

        dtype = cls._dtype_from_width(sampwidth)
        audio = np.frombuffer(frames, dtype=dtype)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)

        audio = audio.astype(np.float32)
        if np.issubdtype(dtype, np.integer):
            audio = audio / float(np.iinfo(dtype).max)

        return audio, sample_rate

    @classmethod
    def _load_via_soundfile(cls, path: Path) -> tuple[np.ndarray, int]:
        info = sf.info(str(path))
        sample_rate = info.samplerate
        audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
        return np.asarray(audio, dtype=np.float32), sample_rate

    @classmethod
    def _load_via_librosa(cls, path: Path) -> tuple[np.ndarray, int]:
        audio, sample_rate = librosa.load(str(path), sr=None, mono=False)
        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim == 2:
            audio = audio.T

        if audio.shape[0] == 0:
            raise ToolValidationError(f"Audio file '{path}' appears to be empty.")

        return audio, sample_rate

    @staticmethod
    def _save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
        audio_int16 = np.int16(np.round(audio * np.iinfo(np.int16).max))
        with wave.open(str(path), "wb") as output_wav:
            output_wav.setnchannels(1)
            output_wav.setsampwidth(2)
            output_wav.setframerate(sample_rate)
            output_wav.writeframes(audio_int16.tobytes())

    @staticmethod
    def _dtype_from_width(width: int) -> Any:
        if width == 1:
            return np.uint8
        if width == 2:
            return np.int16
        if width == 4:
            return np.int32
        raise ToolValidationError(f"Unsupported WAV sample width: {width}")
