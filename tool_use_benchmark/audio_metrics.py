"""Audio-closeness metrics between a model-reconstructed audio and ground truth.

These are plain evaluation-time similarity metrics, not a training loss --
nothing here is differentiated or backpropagated through. `compare_audio`
loads both clips mono at 16kHz and reports:

- `log_mel_cosine` / `log_mel_l1`: cosine similarity and mean absolute
  difference between the two clips' log-mel spectrograms (64 mel bands,
  n_fft=1024, hop=256), i.e. how alike the two sounds are frame-by-frame in a
  perceptually-scaled frequency representation.
- `mfcc_cosine`: cosine similarity between 13-coefficient MFCCs derived from
  the same log-mel spectrograms -- a coarser, more timbre/phoneme-focused view
  than the full mel spectrogram.
- `rms_db_diff`: difference in overall loudness (20*log10(RMS amplitude)).
- `si_sdr_db`: scale-invariant signal-to-distortion ratio on raw waveforms,
  only computed when durations already match within 2% (see below).
- `closeness_score`: the single composite number `summary.mean_audio_closeness_score`
  in run_eval.py's aggregate report is built from, defined as
  `0.25*(log_mel_cosine + 1) + 0.25*(mfcc_cosine + 1)`, i.e. the average of the
  two cosine similarities rescaled from [-1, 1] to [0, 1].

Tools in this benchmark can change duration (clipping, trim_silence, pad_noise,
time_stretch, ...), so plain sample-aligned waveform metrics (e.g. SI-SDR) are
only meaningful when both clips happen to have matching length. The cosine/L1
spectral metrics stay meaningful across length changes because the *shorter*
clip's feature matrix is resampled (via `scipy.signal.resample`, i.e. FFT-based
interpolation) along the time axis onto the *longer* clip's frame count before
comparison -- so e.g. a clip that's been clipped to half its length is still
compared frame-for-frame against the corresponding half of the target, not
penalized purely for having fewer frames. SI-SDR is reported only as a bonus
when durations already line up closely, since it has no such length-robust
form here.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import librosa
import numpy as np
from scipy.signal import resample


def _load_mono(path: str, sr: int = 16000) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten()
    b = b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def _resize_time_axis(feat: np.ndarray, target_frames: int) -> np.ndarray:
    """Resample a (n_features, n_frames) matrix along the time axis to target_frames."""
    if feat.shape[1] == target_frames:
        return feat
    return resample(feat, target_frames, axis=1)


def _align_features(feat_a: np.ndarray, feat_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target = max(feat_a.shape[1], feat_b.shape[1])
    return _resize_time_axis(feat_a, target), _resize_time_axis(feat_b, target)


def _si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    n = min(len(estimate), len(reference))
    estimate, reference = estimate[:n], reference[:n]
    if n == 0 or np.allclose(reference, 0.0):
        return float("nan")
    scale = np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-8)
    projection = scale * reference
    noise = estimate - projection
    ratio = (np.sum(projection ** 2) + 1e-8) / (np.sum(noise ** 2) + 1e-8)
    return float(10 * np.log10(ratio))


def compare_audio(path_pred: str, path_target: str, sr: int = 16000) -> Dict[str, Any]:
    """Return a dict of closeness metrics between a predicted and target audio file."""
    y_pred = _load_mono(path_pred, sr=sr)
    y_tgt = _load_mono(path_target, sr=sr)

    duration_pred = len(y_pred) / sr
    duration_tgt = len(y_tgt) / sr
    duration_ratio = (duration_pred / duration_tgt) if duration_tgt > 0 else float("nan")

    n_fft, hop = 1024, 256
    mel_pred = librosa.feature.melspectrogram(y=y_pred, sr=sr, n_mels=64, n_fft=n_fft, hop_length=hop)
    mel_tgt = librosa.feature.melspectrogram(y=y_tgt, sr=sr, n_mels=64, n_fft=n_fft, hop_length=hop)
    log_mel_pred = librosa.power_to_db(mel_pred)
    log_mel_tgt = librosa.power_to_db(mel_tgt)
    log_mel_pred_a, log_mel_tgt_a = _align_features(log_mel_pred, log_mel_tgt)
    log_mel_cosine = _cosine(log_mel_pred_a, log_mel_tgt_a)
    log_mel_l1 = float(np.mean(np.abs(log_mel_pred_a - log_mel_tgt_a)))

    mfcc_pred = librosa.feature.mfcc(S=log_mel_pred, n_mfcc=13)
    mfcc_tgt = librosa.feature.mfcc(S=log_mel_tgt, n_mfcc=13)
    mfcc_pred_a, mfcc_tgt_a = _align_features(mfcc_pred, mfcc_tgt)
    mfcc_cosine = _cosine(mfcc_pred_a, mfcc_tgt_a)

    rms_pred_db = float(20 * np.log10(np.sqrt(np.mean(y_pred ** 2)) + 1e-8))
    rms_tgt_db = float(20 * np.log10(np.sqrt(np.mean(y_tgt ** 2)) + 1e-8))

    si_sdr: Optional[float] = None
    if duration_tgt > 0 and abs(duration_ratio - 1.0) < 0.02:
        si_sdr = _si_sdr(y_pred, y_tgt)

    closeness_score = float(np.clip(0.5 * (log_mel_cosine + 1) / 2 + 0.5 * (mfcc_cosine + 1) / 2, 0.0, 1.0))

    return {
        "duration_pred_sec": duration_pred,
        "duration_target_sec": duration_tgt,
        "duration_ratio": duration_ratio,
        "log_mel_cosine": log_mel_cosine,
        "log_mel_l1": log_mel_l1,
        "mfcc_cosine": mfcc_cosine,
        "rms_db_pred": rms_pred_db,
        "rms_db_target": rms_tgt_db,
        "rms_db_diff": rms_pred_db - rms_tgt_db,
        "si_sdr_db": si_sdr,
        "closeness_score": closeness_score,
    }
