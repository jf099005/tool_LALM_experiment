"""Registry of audio-editing tools usable to synthesize A -> B tool-call chains.

Each entry wraps an existing tool implementation from `tools/` (or `audio_edit/`)
behind one call shape:

    output_path, parameters = registry[name].apply(audio_path, output_path, rng, duration)

`parameters` is the exact argument dict that was passed to the underlying tool
(with `audio_path` set to the real working file), so it can be reused verbatim
as the ground-truth `tool_calls` entry for that step (after swapping in the
placeholder audio reference). Availability is probed once at import time so the
registry only exposes tools whose backend dependencies actually import in the
current interpreter -- run this under the project's `ms-swift` (or similar)
conda env to unlock librosa/soundfile-backed tools, and under `deepfilternet`,
`audiosr`, or `sam_audio` to unlock those specific heavy tools.
"""

from __future__ import annotations

import importlib
import shutil
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import random

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.abstract_tool import ToolValidationError  # noqa: E402
from tools.clipping import ClippingTool  # noqa: E402
from tools.denoise_old import DenoiseTool  # noqa: E402
from tools.normalize import (  # noqa: E402
    AmplitudeNormalizeTool,
    DCOffsetRemovalTool,
    LoudnessNormalizeTool,
    PreEmphasisTool,
    SpectralNormalizeTool,
    TrimSilenceTool,
)
from tools.pitch_time import (  # noqa: E402
    PITCH_SHIFT_STEPS,
    TIME_STRETCH_RATES,
    PitchShiftTool,
    TimeStretchTool,
)


def _probe(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


_HAS_LIBROSA = _probe("librosa") and hasattr(importlib.import_module("librosa"), "load")
_HAS_TORCH = _probe("torch")
_HAS_DEEPFILTER = _HAS_TORCH and _probe("df.enhance")
_HAS_AUDIOSR = _HAS_TORCH and _probe("audiosr")
_HAS_SAM_AUDIO = _HAS_TORCH and _probe("sam_audio")
_HAS_SOUNDFILE = _probe("soundfile")

if _HAS_DEEPFILTER:
    from tools.human_voice_enhance import HumanVoiceAmplifyTool, HumanVoiceEnhanceTool, VOICE_AMPLIFY_GAIN_DB  # noqa: E402
if _HAS_AUDIOSR:
    from tools.super_resolution import SuperResolutionTool  # noqa: E402
if _HAS_SAM_AUDIO:
    from tools.extract_remove_target import SEPARATION_LABELS, ExtractTargetTool, RemoveTargetTool  # noqa: E402

_HAS_AUDIO_EDIT = _HAS_SOUNDFILE and _HAS_LIBROSA
if _HAS_AUDIO_EDIT:
    try:
        from audio_edit.editor import add_noise as _ae_add_noise  # noqa: E402
        from audio_edit.editor import insert_background_event as _ae_insert_event  # noqa: E402
        from audio_edit.editor import pad_noise as _ae_pad_noise  # noqa: E402
    except ImportError:
        _HAS_AUDIO_EDIT = False


# ---------------------------------------------------------------------------
# Audio helpers (avoid a hard soundfile dependency where the stdlib suffices)
# ---------------------------------------------------------------------------

def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm, always rounding *down*.

    Flooring (with a tiny epsilon to absorb float repr error, e.g. a duration of
    10.0009375s naively rounds to "10.001", which is *past* the true sample
    count once re-parsed and re-multiplied by the sample rate) guarantees a
    formatted timestamp never exceeds the true position it was derived from --
    important since these timestamps get fed straight back into begin/end
    bounds checks against the actual frame count.
    """
    total_millis = int((max(0.0, seconds) - 1e-6) * 1000)
    total_millis = max(0, total_millis)
    whole, millis = divmod(total_millis, 1000)
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def get_duration_seconds(path: Path) -> float:
    path = Path(path)
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as wav_file:
            return wav_file.getnframes() / float(wav_file.getframerate())
    if not _HAS_LIBROSA:
        raise RuntimeError(f"librosa is required to read the duration of non-WAV file: {path}")
    librosa = importlib.import_module("librosa")
    return float(librosa.get_duration(path=str(path)))


def ensure_wav(path: Path, work_dir: Path) -> Path:
    """Return a WAV copy of `path`, converting (and resampling to mono) if needed."""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        return path

    if not _HAS_LIBROSA:
        raise RuntimeError(f"librosa is required to convert non-WAV source audio: {path}")

    librosa = importlib.import_module("librosa")
    import numpy as np

    audio, sr = librosa.load(str(path), sr=None, mono=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / f"{path.stem}.wav"

    audio = np.clip(audio, -1.0, 1.0)
    int_samples = (audio * np.iinfo(np.int16).max).astype(np.int16)
    with wave.open(str(out_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes(int_samples.tobytes())

    return out_path


def _finalize(produced_path: Path, desired_path: Path) -> Path:
    """Move a tool's auto-named output to the chain's desired file name."""
    produced_path = Path(produced_path)
    desired_path = Path(desired_path)
    if produced_path == desired_path:
        return desired_path
    desired_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced_path), str(desired_path))
    return desired_path


@dataclass
class ToolSpec:
    description: str
    apply: Callable[[Path, Path, "random.Random", float], Tuple[Dict[str, Any], Path]]


REGISTRY: Dict[str, ToolSpec] = {}


def register(name: str, description: str):
    def _decorator(fn: Callable[[Path, Path, random.Random, float], Tuple[Dict[str, Any], Path]]):
        REGISTRY[name] = ToolSpec(description=description, apply=fn)
        return fn
    return _decorator


# ---------------------------------------------------------------------------
# Lightweight, dependency-light tools (stdlib wave + numpy/scipy only)
# ---------------------------------------------------------------------------

@register("clipping", ClippingTool.description())
def _apply_clipping(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
    begin = 0.0
    end = max(min(duration, 0.5), duration * rng.uniform(0.5, 0.9))
    params = {
        "audio_path": str(audio_path),
        "audio_begin": format_timestamp(begin),
        "audio_end": format_timestamp(end),
    }
    result = ClippingTool.execute(params)
    final_path = _finalize(Path(result["clip_path"]), output_path)
    return params, final_path


@register("denoise", DenoiseTool.description())
def _apply_denoise(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
    algorithm = rng.choice(["spectral_subtraction", "wiener", "echo_cancellation", "adaptive"])
    params: Dict[str, Any] = {"audio_path": str(audio_path), "algorithm": algorithm}
    if algorithm == "spectral_subtraction":
        params["noise_factor"] = round(rng.uniform(1.0, 3.0), 2)
    elif algorithm == "adaptive":
        params["sensitivity"] = round(rng.uniform(0.2, 0.8), 2)
    params["output_path"] = str(output_path)
    result = DenoiseTool.execute(params)
    params.pop("output_path")
    return params, Path(result["output_path"])


@register("amplitude_normalize", AmplitudeNormalizeTool.description())
def _apply_amplitude_normalize(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
    params = {
        "audio_path": str(audio_path),
        "audio_begin": format_timestamp(0.0),
        "audio_end": format_timestamp(duration),
        "target_level": round(rng.uniform(0.6, 0.95), 2),
        "method": rng.choice(["peak", "rms"]),
        "output_path": str(output_path),
    }
    result = AmplitudeNormalizeTool.execute(params)
    params.pop("output_path")
    return params, Path(result["output_path"])


@register("loudness_normalize", LoudnessNormalizeTool.description())
def _apply_loudness_normalize(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
    params = {
        "audio_path": str(audio_path),
        "audio_begin": format_timestamp(0.0),
        "audio_end": format_timestamp(duration),
        "target_lufs": rng.choice([-23.0, -20.0, -18.0, -16.0, -14.0]),
        "output_path": str(output_path),
    }
    result = LoudnessNormalizeTool.execute(params)
    params.pop("output_path")
    return params, Path(result["output_path"])


@register("remove_dc_offset", DCOffsetRemovalTool.description())
def _apply_remove_dc_offset(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
    params = {
        "audio_path": str(audio_path),
        "audio_begin": format_timestamp(0.0),
        "audio_end": format_timestamp(duration),
        "output_path": str(output_path),
    }
    result = DCOffsetRemovalTool.execute(params)
    params.pop("output_path")
    return params, Path(result["output_path"])


@register("pre_emphasis", PreEmphasisTool.description())
def _apply_pre_emphasis(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
    params = {
        "audio_path": str(audio_path),
        "audio_begin": format_timestamp(0.0),
        "audio_end": format_timestamp(duration),
        "coef": rng.choice([0.9, 0.95, 0.97, 0.99]),
        "output_path": str(output_path),
    }
    result = PreEmphasisTool.execute(params)
    params.pop("output_path")
    return params, Path(result["output_path"])


if _HAS_LIBROSA:
    @register("spectral_normalize", SpectralNormalizeTool.description())
    def _apply_spectral_normalize(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
        params = {
            "audio_path": str(audio_path),
            "audio_begin": format_timestamp(0.0),
            "audio_end": format_timestamp(duration),
            "strength": round(rng.uniform(0.3, 0.8), 2),
            "output_path": str(output_path),
        }
        result = SpectralNormalizeTool.execute(params)
        params.pop("output_path")
        return params, Path(result["output_path"])

    @register("trim_silence", TrimSilenceTool.description())
    def _apply_trim_silence(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
        params = {
            "audio_path": str(audio_path),
            "audio_begin": format_timestamp(0.0),
            "audio_end": format_timestamp(duration),
            "threshold_db": rng.choice([25, 30, 35, 40, 45]),
            "output_path": str(output_path),
        }
        result = TrimSilenceTool.execute(params)
        params.pop("output_path")
        return params, Path(result["output_path"])


# ---------------------------------------------------------------------------
# Pitch / time-stretch: delegate to a subprocess in a dedicated conda env, so
# these stay available even though the main interpreter lacks `pedalboard`.
# ---------------------------------------------------------------------------

@register("pitch_shift", PitchShiftTool.description())
def _apply_pitch_shift(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
    params = {
        "audio_path": str(audio_path),
        "n_steps": rng.choice(PITCH_SHIFT_STEPS),
        "output_path": str(output_path),
    }
    result = PitchShiftTool.execute(params)
    params.pop("output_path")
    return params, Path(result["output_path"])


@register("time_stretch", TimeStretchTool.description())
def _apply_time_stretch(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
    params = {
        "audio_path": str(audio_path),
        "rate": rng.choice(TIME_STRETCH_RATES),
        "output_path": str(output_path),
    }
    result = TimeStretchTool.execute(params)
    params.pop("output_path")
    return params, Path(result["output_path"])


# ---------------------------------------------------------------------------
# Heavy ML-backed tools, only registered when their dependency actually imports.
# ---------------------------------------------------------------------------

if _HAS_DEEPFILTER:
    @register("human_voice_enhance", HumanVoiceEnhanceTool.description())
    def _apply_human_voice_enhance(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
        params = {"audio_path": str(audio_path), "output_path": str(output_path)}
        result = HumanVoiceEnhanceTool.execute(params)
        params.pop("output_path")
        return params, Path(result["output_path"])

    @register("human_voice_amplify", HumanVoiceAmplifyTool.description())
    def _apply_human_voice_amplify(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
        params = {
            "audio_path": str(audio_path),
            "gain_db": rng.choice(VOICE_AMPLIFY_GAIN_DB),
            "output_path": str(output_path),
        }
        result = HumanVoiceAmplifyTool.execute(params)
        params.pop("output_path")
        return params, Path(result["output_path"])

if _HAS_AUDIOSR:
    @register("super_resolution", SuperResolutionTool.description())
    def _apply_super_resolution(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
        params = {
            "audio_path": str(audio_path),
            "model_name": rng.choice(["basic", "speech"]),
            "output_path": str(output_path),
        }
        result = SuperResolutionTool.execute(params)
        params.pop("output_path")
        return params, Path(result["output_path"])

if _HAS_SAM_AUDIO:
    @register("extract_target", ExtractTargetTool.description())
    def _apply_extract_target(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
        params = {
            "audio_path": str(audio_path),
            "target_description": rng.choice(SEPARATION_LABELS),
        }
        result = ExtractTargetTool.execute(params)
        final_path = _finalize(Path(result["output_path"]), output_path)
        return params, final_path

    @register("remove_target", RemoveTargetTool.description())
    def _apply_remove_target(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
        params = {
            "audio_path": str(audio_path),
            "target_description": rng.choice(SEPARATION_LABELS),
        }
        result = RemoveTargetTool.execute(params)
        final_path = _finalize(Path(result["output_path"]), output_path)
        return params, final_path


# ---------------------------------------------------------------------------
# audio_edit ops: noise mixing / padding / background-event insertion.
# ---------------------------------------------------------------------------

if _HAS_AUDIO_EDIT:
    _INSERT_EVENT_LABELS = ["Dog", "Bird", "Wind", "Rain", "Siren", "Traffic noise", "Cat", "Vehicle"]

    @register("add_noise", "Mix Gaussian (or a real recording's) noise into the whole clip at a target SNR.")
    def _apply_add_noise(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
        params = {
            "audio_path": str(audio_path),
            "snr_db": round(rng.uniform(5.0, 20.0), 1),
            "output_path": str(output_path),
        }
        result = _ae_add_noise(**params)
        params.pop("output_path")
        return params, Path(result["output_path"])

    @register("pad_noise", "Extend the clip with noise padding at the start, end, or both.")
    def _apply_pad_noise(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
        params = {
            "audio_path": str(audio_path),
            "position": rng.choice(["start", "end", "both"]),
            "duration_sec": round(rng.uniform(0.3, 1.5), 2),
            "output_path": str(output_path),
        }
        result = _ae_pad_noise(**params)
        params.pop("output_path")
        return params, Path(result["output_path"])

    @register("insert_event", "Insert a labeled AudioSet background sound event at a timestamp.")
    def _apply_insert_event(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
        timestamp = round(rng.uniform(0.0, max(0.0, duration - 0.2)), 2)
        params = {
            "audio_path": str(audio_path),
            "label": rng.choice(_INSERT_EVENT_LABELS),
            "timestamp": format_timestamp(timestamp),
            "snr_db": round(rng.uniform(0.0, 10.0), 1),
            "output_path": str(output_path),
        }
        result = _ae_insert_event(**params)
        params.pop("output_path")
        return params, Path(result["output_path"])


def available_tool_names() -> list[str]:
    return sorted(REGISTRY.keys())


def describe_available_tools() -> str:
    lines = [f"- {name}: {spec.description}" for name, spec in sorted(REGISTRY.items())]
    return "\n".join(lines)
