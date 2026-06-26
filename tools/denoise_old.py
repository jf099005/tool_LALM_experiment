from __future__ import annotations

from tqdm import tqdm
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

from .abstract_tool import Tool, ToolValidationError

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None


class DenoiseTool(Tool):
    @classmethod
    def name(cls) -> str:
        return "denoise"

    @classmethod
    def description(cls) -> str:
        return (
            "Apply noise reduction or echo cancellation to a WAV audio file using the chosen "
            "algorithm. Requires audio_path and algorithm; optional noise_factor, "
            "sensitivity, and output_path control processing and output destination."
        )

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"},
                "algorithm": {
                    "type": "string",
                    "enum": [
                        "spectral_subtraction",
                        "wiener",
                        "echo_cancellation",
                        "adaptive",
                    ],
                },
                "noise_factor": {
                    "type": "number",
                },
                "sensitivity": {
                    "type": "number",
                },
                "output_path": {"type": "string"},
            },
            "required": ["audio_path", "algorithm"],
        }

    @classmethod
    def execute(cls, parameters: Dict[str, Any]) -> Dict[str, Any]:
        cls.validate_parameters(parameters)

        audio_path = parameters["audio_path"]
        algorithm = parameters["algorithm"]
        noise_factor = float(parameters.get("noise_factor", 2.0))
        sensitivity = float(parameters.get("sensitivity", 0.5))
        output_path = parameters.get("output_path")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise ToolValidationError(f"Audio file not found: {audio_path}")

        if audio_path.suffix.lower() != ".wav":
            raise ToolValidationError("Denoise currently supports only WAV audio files.")

        audio, sample_rate = cls._load_wav(audio_path)
        if audio.size == 0:
            raise ToolValidationError("Requested audio segment is empty.")

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if algorithm == "spectral_subtraction":
            processed = cls._spectral_subtraction(audio, sample_rate, noise_factor=noise_factor)
        elif algorithm == "wiener":
            processed = cls._wiener_denoise(audio, sample_rate)
        elif algorithm == "echo_cancellation":
            processed = cls._echo_cancellation(audio, sample_rate)
        elif algorithm == "adaptive":
            processed = cls._adaptive_denoise(audio, sample_rate, sensitivity=sensitivity)
        else:
            raise ToolValidationError(f"Unsupported algorithm: {algorithm}")

        processed = np.clip(processed, -1.0, 1.0)

        if output_path is None:
            output_path = (
                audio_path.parent
                / f"{audio_path.stem}_{algorithm}.wav"
            )
        else:
            output_path = Path(output_path)

        cls._save_wav(output_path, processed, sample_rate)

        return {
            "audio_path": str(audio_path),
            "algorithm": algorithm,
            "status": "success",
            "output_path": str(output_path),
            "message": f"Denoise operation completed using algorithm '{algorithm}'.",
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
            except ToolValidationError as exc:
                results.append(
                    {
                        "audio_path": parameters.get("audio_path"),
                        "algorithm": parameters.get("algorithm"),
                        "status": "failure",
                        "output_path": None,
                        "message": str(exc),
                    }
                )

        return results

    @classmethod
    def _spectral_subtraction(
        cls,
        audio: np.ndarray,
        sr: int,
        noise_factor: float = 2.0,
    ) -> np.ndarray:
        n_fft = 2048
        hop_length = 512
        _, _, Zxx = signal.stft(
            audio,
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            boundary=None,
            padded=False,
        )

        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        noise_frames = max(1, int(0.5 * sr / hop_length))
        noise_estimate = np.median(magnitude[:, :noise_frames], axis=1, keepdims=True)

        magnitude_denoised = magnitude - noise_factor * noise_estimate
        magnitude_denoised = np.maximum(magnitude_denoised, 0.1 * magnitude)

        Zxx_denoised = magnitude_denoised * np.exp(1j * phase)
        _, audio_denoised = signal.istft(
            Zxx_denoised,
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            input_onesided=True,
        )
        return cls._match_length(audio_denoised, len(audio))

    @classmethod
    def _wiener_denoise(cls, audio: np.ndarray, sr: int) -> np.ndarray:
        n_fft = 2048
        hop_length = 512
        _, _, Zxx = signal.stft(
            audio,
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            boundary=None,
            padded=False,
        )

        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        noise_frames = max(1, int(0.5 * sr / hop_length))
        noise_power = np.mean(magnitude[:, :noise_frames] ** 2, axis=1, keepdims=True)

        signal_power = magnitude ** 2
        wiener_gain = np.maximum(signal_power - noise_power, 0.0) / (signal_power + 1e-10)
        magnitude_filtered = magnitude * wiener_gain

        Zxx_filtered = magnitude_filtered * np.exp(1j * phase)
        _, audio_filtered = signal.istft(
            Zxx_filtered,
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            input_onesided=True,
        )
        return cls._match_length(audio_filtered, len(audio))

    @classmethod
    def _echo_cancellation(cls, audio: np.ndarray, sr: int) -> np.ndarray:
        sos = signal.butter(4, 80, "hp", fs=sr, output="sos")
        filtered = signal.sosfilt(sos, audio)

        n_fft = 2048
        hop_length = 512
        _, _, Zxx = signal.stft(
            filtered,
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            boundary=None,
            padded=False,
        )

        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        envelope = median_filter(magnitude, size=(1, 21))
        magnitude_reduced = magnitude / (1.0 + 0.3 * envelope / (np.max(envelope) + 1e-10))

        Zxx_reduced = magnitude_reduced * np.exp(1j * phase)
        _, audio_reduced = signal.istft(
            Zxx_reduced,
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            input_onesided=True,
        )
        return cls._match_length(audio_reduced, len(audio))

    @classmethod
    def _adaptive_denoise(cls, audio: np.ndarray, sr: int, sensitivity: float = 0.5) -> np.ndarray:
        sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
        n_fft = 2048
        hop_length = 512
        _, _, Zxx = signal.stft(
            audio,
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            boundary=None,
            padded=False,
        )

        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        noise_estimate = np.zeros_like(magnitude)
        window_size = 20
        num_bins, num_frames = magnitude.shape
        for freq_idx in range(num_bins):
            for frame_idx in range(num_frames):
                start = max(0, frame_idx - window_size)
                end = min(num_frames, frame_idx + window_size)
                percentile = 10 + 40 * (1.0 - sensitivity)
                noise_estimate[freq_idx, frame_idx] = np.percentile(
                    magnitude[freq_idx, start:end], percentile
                )

        magnitude_denoised = magnitude - 1.5 * sensitivity * noise_estimate
        magnitude_denoised = np.maximum(magnitude_denoised, 0.05 * magnitude)

        Zxx_denoised = magnitude_denoised * np.exp(1j * phase)
        _, audio_denoised = signal.istft(
            Zxx_denoised,
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            input_onesided=True,
        )
        return cls._match_length(audio_denoised, len(audio))

    @staticmethod
    def _match_length(audio: np.ndarray, target_len: int) -> np.ndarray:
        if len(audio) > target_len:
            return audio[:target_len]
        if len(audio) < target_len:
            return np.pad(audio, (0, target_len - len(audio)))
        return audio

    @classmethod
    def _load_wav(cls, path: Path) -> tuple[np.ndarray, int]:
        # --- attempt 1: stdlib wave ---
        try:
            return cls._load_via_wave(path)
        except ToolValidationError:
            raise
        except Exception:
            pass

        # --- attempt 2: soundfile ---
        if sf is not None:
            try:
                return cls._load_via_soundfile(path)
            except ToolValidationError:
                raise
            except Exception:
                pass

        # --- attempt 3: librosa ---
        if librosa is not None:
            try:
                return cls._load_via_librosa(path)
            except ToolValidationError:
                raise
            except Exception:
                pass

        raise ToolValidationError(
            f"Failed to open audio file '{path}'. "
            "The file could not be read by the wave, soundfile, or librosa backends. "
            "Install soundfile or librosa, or convert the file to standard PCM WAV format."
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

        # librosa returns (channels, samples) for multi-channel; normalise to (samples,) or (samples, channels)
        if audio.ndim == 2:
            audio = audio.T  # -> (samples, channels)

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
