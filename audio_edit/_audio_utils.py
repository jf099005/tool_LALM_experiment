"""Shared audio I/O and signal-processing helpers used across `audio_edit/`.

Kept dependency-light (numpy + librosa/soundfile only) and importable both as
part of the `audio_edit` package and as a standalone module, since the sibling
scripts in this directory are also run directly (e.g. `python
audio_edit/audio_add_noise.py ...`).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None


class AudioEditError(ValueError):
    """Raised for invalid parameters or unusable audio in audio_edit tools."""


def parse_timestamp(value: str) -> float:
    """Parse an HH:MM:SS.mmm timestamp into seconds."""
    try:
        time_text, millis_text = value.split(".")
        hours, minutes, seconds = (int(part) for part in time_text.split(":"))
        millis = int(millis_text.ljust(3, "0")[:3])
    except (ValueError, AttributeError) as exc:
        raise AudioEditError(f"Invalid timestamp '{value}'. Expected HH:MM:SS.mmm") from exc
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm, flooring so re-parsing never overshoots."""
    total_millis = int((max(0.0, seconds) - 1e-6) * 1000)
    total_millis = max(0, total_millis)
    whole, millis = divmod(total_millis, 1000)
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def load_audio(path: Path, sr: Optional[int] = None, mono: bool = True) -> Tuple[np.ndarray, int]:
    """Load audio as float32 via librosa (resamples only if `sr` is given)."""
    if librosa is None:
        raise AudioEditError("librosa is required. Install it with `pip install librosa`.")

    path = Path(path)
    if not path.exists():
        raise AudioEditError(f"Audio file not found: {path}")

    try:
        audio, sample_rate = librosa.load(str(path), sr=sr, mono=mono)
    except Exception as exc:
        raise AudioEditError(f"Failed to open audio file '{path}': {exc}") from exc

    audio = np.asarray(audio, dtype=np.float32)
    if audio.shape[-1] == 0:
        raise AudioEditError(f"Audio file '{path}' appears to be empty.")

    return audio, sample_rate


def save_audio(path: Path, audio: np.ndarray, sample_rate: int, ceiling: float = 0.98) -> Path:
    """Write float audio to a 16-bit PCM WAV, peak-limiting only if it would clip."""
    if sf is None:
        raise AudioEditError("soundfile is required. Install it with `pip install soundfile`.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > ceiling:
        audio = audio / peak * ceiling

    sf.write(str(path), audio.astype(np.float32), sample_rate, subtype="PCM_16")
    return path


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def db_to_ratio(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def scale_to_snr(reference: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale `noise` so that reference_rms / scaled_noise_rms == 10**(snr_db/20)."""
    ref_rms = rms(reference)
    noise_rms = rms(noise)
    if ref_rms <= 0 or noise_rms <= 0:
        return noise

    target_noise_rms = ref_rms / db_to_ratio(snr_db)
    return noise * (target_noise_rms / noise_rms)


def generate_noise(n_samples: int, noise_type: str = "white", rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate unit-RMS noise. `noise_type` is "white" or "pink" (1/f, Fourier-filtered)."""
    rng = rng or np.random.default_rng()
    white = rng.standard_normal(n_samples).astype(np.float32)

    if noise_type == "white":
        noise = white
    elif noise_type == "pink":
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples)
        freqs[0] = freqs[1] if n_samples > 1 else 1.0  # avoid divide-by-zero at DC
        spectrum = spectrum / np.sqrt(freqs)
        noise = np.fft.irfft(spectrum, n=n_samples).astype(np.float32)
    else:
        raise AudioEditError(f"Unknown noise_type '{noise_type}'. Expected 'white' or 'pink'.")

    current_rms = rms(noise)
    return noise / current_rms if current_rms > 0 else noise


def fade_in_out(audio: np.ndarray, sample_rate: int, fade_ms: float) -> np.ndarray:
    """Apply a raised-cosine (Hann-half) fade in/out to avoid onset/offset clicks."""
    if fade_ms <= 0:
        return audio

    audio = audio.copy()
    fade_len = min(int(sample_rate * fade_ms / 1000.0), audio.shape[-1] // 2)
    if fade_len <= 0:
        return audio

    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_len, dtype=np.float32)))
    audio[..., :fade_len] *= ramp
    audio[..., -fade_len:] *= ramp[::-1]
    return audio


def fit_to_length(audio: np.ndarray, n_samples: int) -> np.ndarray:
    """Loop (tile) or trim `audio` so it is exactly `n_samples` long."""
    if audio.shape[-1] >= n_samples:
        return audio[..., :n_samples]

    repeats = int(np.ceil(n_samples / audio.shape[-1]))
    return np.tile(audio, repeats)[..., :n_samples]


def default_output_path(audio_path: Path, suffix: str) -> Path:
    audio_path = Path(audio_path)
    return audio_path.parent / f"{audio_path.stem}_{suffix}.wav"


def _validate_timestamp_format(value: str) -> None:
    try:
        datetime.strptime(value, "%H:%M:%S.%f")
    except ValueError as exc:
        raise AudioEditError(f"'{value}' must be in HH:MM:SS.mmm format: {exc}") from exc
