"""Registry of audio-editing tools usable to synthesize A -> B tool-call chains.

Used by `tool_use_training/gen_1st_stage_data/build_dataset.py` (imported there as
`from tools import synthetic_registry as tool_registry`). Each entry wraps an
existing tool implementation from this package (or `audio_edit/`) behind one
call shape:

    output_path, parameters = registry[name].apply(audio_path, output_path, rng, duration)

`parameters` is the exact argument dict that was passed to the underlying tool
(with `audio_path` set to the real working file), so it can be reused verbatim
as the ground-truth `tool_calls` entry for that step (after swapping in the
placeholder audio reference). Availability is probed once at import time so the
registry only exposes tools whose backend dependencies actually import in the
current interpreter -- run this under the project's `ms-swift` (or similar)
conda env to unlock librosa/soundfile-backed tools.

Heavy ML tools (`human_voice_enhance`, `super_resolution`, `extract_target`,
`remove_target`) are never imported into this interpreter -- this registry is
meant to run in the ms-swift env, which deliberately doesn't carry torch,
DeepFilterNet, AudioSR, or sam_audio. Instead each call is dispatched
out-of-process via `tools/tool_batch_execute.py`, run under that tool's own
conda env (see `_CONDA_ENV_PYTHON`, mirroring `apply_tools.py`'s `TOOL_ENV`).
Availability is therefore probed by checking the target env's python exists
on disk, not by importing anything.
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import random

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .abstract_tool import ToolValidationError  # noqa: E402
from .clipping import ClippingTool  # noqa: E402
from .denoise import DenoiseTool  # noqa: E402
from .normalize import (  # noqa: E402
    AmplitudeNormalizeTool,
    DCOffsetRemovalTool,
    LoudnessNormalizeTool,
    PreEmphasisTool,
    SpectralNormalizeTool,
    TrimSilenceTool,
)
from .pitch_time import (  # noqa: E402
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
_HAS_SOUNDFILE = _probe("soundfile")

# Heavy ML tools (DeepFilterNet / AudioSR / sam_audio) don't get imported into
# this interpreter -- this registry runs under the ms-swift env, which
# intentionally doesn't carry torch or those packages. Instead they're
# dispatched out-of-process to their own dedicated conda envs via
# `tools/tool_batch_execute.py`, the same split apply_tools.py's TOOL_ENV
# uses. Availability is therefore probed by checking the env's python exists,
# not by importing anything here.
_CONDA_ENV_PYTHON: Dict[str, str] = {
    "human_voice_enhance": "/home/u1501463/miniconda3/envs/deepfilternet/bin/python",
    "super_resolution": "/home/u1501463/miniconda3/envs/audiosr/bin/python",
    "extract_target": "/home/u1501463/miniconda3/envs/sam_audio/bin/python",
    "remove_target": "/home/u1501463/miniconda3/envs/sam_audio/bin/python",
}


def _run_tool_subprocess(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Run one heavy ML tool call out-of-process, in the conda env that owns it."""
    python_executable = _CONDA_ENV_PYTHON[tool_name]
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = Path(tmp_dir) / "input.json"
        output_path = Path(tmp_dir) / "output.json"
        input_path.write_text(json.dumps([params]), encoding="utf-8")
        subprocess.run(
            [
                python_executable,
                str(REPO_ROOT / "tools" / "tool_batch_execute.py"),
                "--tool-name", tool_name,
                "--input-file", str(input_path),
                "--output-file", str(output_path),
            ],
            check=True,
        )
        results = json.loads(output_path.read_text(encoding="utf-8"))

    if not isinstance(results, list) or len(results) != 1:
        raise RuntimeError(f"Unexpected batch result for {tool_name}: {results!r}")
    result = results[0]
    if result.get("status") not in (None, "success"):
        raise RuntimeError(f"{tool_name} failed: {result.get('message') or result.get('error')}")
    return result


_HAS_DEEPFILTER = Path(_CONDA_ENV_PYTHON["human_voice_enhance"]).exists()
_HAS_AUDIOSR = Path(_CONDA_ENV_PYTHON["super_resolution"]).exists()
_HAS_SAM_AUDIO = Path(_CONDA_ENV_PYTHON["extract_target"]).exists()

_SEPARATION_LABELS = ["vocals", "background sound", "background music", "noise"]

# add_noise / pad_noise / insert_event (below) are all disabled, so nothing
# currently needs audio_edit.editor -- left as a flag/import stub in case
# any of them get re-enabled later.
_HAS_AUDIO_EDIT = _HAS_SOUNDFILE and _HAS_LIBROSA
# if _HAS_AUDIO_EDIT:
#     try:
#         from audio_edit.editor import insert_background_event as _ae_insert_event  # noqa: E402
#     except ImportError:
#         _HAS_AUDIO_EDIT = False


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
    result = DenoiseTool.execute(params, str(output_path))
    return params, Path(result["output_path"])


# @register("amplitude_normalize", AmplitudeNormalizeTool.description())
# def _apply_amplitude_normalize(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
#     params = {
#         "audio_path": str(audio_path),
#         "audio_begin": format_timestamp(0.0),
#         "audio_end": format_timestamp(duration),
#         "target_level": round(rng.uniform(0.6, 0.95), 2),
#         "method": rng.choice(["peak", "rms"]),
#     }
#     result = AmplitudeNormalizeTool.execute(params, str(output_path))
#     return params, Path(result["output_path"])


# @register("loudness_normalize", LoudnessNormalizeTool.description())
# def _apply_loudness_normalize(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
#     params = {
#         "audio_path": str(audio_path),
#         "audio_begin": format_timestamp(0.0),
#         "audio_end": format_timestamp(duration),
#         "target_lufs": rng.choice([-23.0, -20.0, -18.0, -16.0, -14.0]),
#     }
#     result = LoudnessNormalizeTool.execute(params, str(output_path))
#     return params, Path(result["output_path"])


# @register("remove_dc_offset", DCOffsetRemovalTool.description())
# def _apply_remove_dc_offset(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
#     params = {
#         "audio_path": str(audio_path),
#         "audio_begin": format_timestamp(0.0),
#         "audio_end": format_timestamp(duration),
#     }
#     result = DCOffsetRemovalTool.execute(params, str(output_path))
#     return params, Path(result["output_path"])


@register("pre_emphasis", PreEmphasisTool.description())
def _apply_pre_emphasis(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
    params = {
        "audio_path": str(audio_path),
        "audio_begin": format_timestamp(0.0),
        "audio_end": format_timestamp(duration),
        "coef": rng.choice([0.9, 0.95, 0.97, 0.99]),
    }
    result = PreEmphasisTool.execute(params, str(output_path))
    return params, Path(result["output_path"])


if _HAS_LIBROSA:
    # @register("spectral_normalize", SpectralNormalizeTool.description())
    # def _apply_spectral_normalize(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
    #     params = {
    #         "audio_path": str(audio_path),
    #         "audio_begin": format_timestamp(0.0),
    #         "audio_end": format_timestamp(duration),
    #         "strength": round(rng.uniform(0.3, 0.8), 2),
    #     }
    #     result = SpectralNormalizeTool.execute(params, str(output_path))
    #     return params, Path(result["output_path"])

    @register("trim_silence", TrimSilenceTool.description())
    def _apply_trim_silence(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
        params = {
            "audio_path": str(audio_path),
            "audio_begin": format_timestamp(0.0),
            "audio_end": format_timestamp(duration),
            "threshold_db": rng.choice([25, 30, 35, 40, 45]),
        }
        result = TrimSilenceTool.execute(params, str(output_path))
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
    }
    result = PitchShiftTool.execute(params, str(output_path))
    return params, Path(result["output_path"])


@register("time_stretch", TimeStretchTool.description())
def _apply_time_stretch(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
    params = {
        "audio_path": str(audio_path),
        "rate": rng.choice(TIME_STRETCH_RATES),
    }
    result = TimeStretchTool.execute(params, str(output_path))
    return params, Path(result["output_path"])


# ---------------------------------------------------------------------------
# Heavy ML-backed tools, dispatched via subprocess to their own conda env (see
# `_run_tool_subprocess` above), only registered when that env's python exists.
# ---------------------------------------------------------------------------

# if _HAS_DEEPFILTER:
#     @register(
#         "human_voice_enhance",
#         "Enhance human voice in an audio file by reducing background noise using DeepFilterNet.",
#     )
#     def _apply_human_voice_enhance(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
#         params = {"audio_path": str(audio_path)}
#         result = _run_tool_subprocess("human_voice_enhance", params)
#         final_path = _finalize(Path(result["output_path"]), output_path)
#         return params, final_path

# if _HAS_AUDIOSR:
#     @register(
#         "super_resolution",
#         "Upsample and restore high-frequency detail of an audio file using the AudioSR "
#         "latent diffusion model, producing a 48kHz output.",
#     )
#     def _apply_super_resolution(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
#         params = {
#             "audio_path": str(audio_path),
#             "model_name": rng.choice(["basic", "speech"]),
#         }
#         result = _run_tool_subprocess("super_resolution", params)
#         final_path = _finalize(Path(result["output_path"]), output_path)
#         return params, final_path

# if _HAS_SAM_AUDIO:
#     @register(
#         "extract_target",
#         "Extract a specific sound source from a WAV audio segment. "
#         f"The label must be one of the fixed supported values: {', '.join(_SEPARATION_LABELS)}.",
#     )
#     def _apply_extract_target(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
#         params = {
#             "audio_path": str(audio_path),
#             "target_description": rng.choice(_SEPARATION_LABELS),
#         }
#         result = _run_tool_subprocess("extract_target", params)
#         final_path = _finalize(Path(result["output_path"]), output_path)
#         return params, final_path

#     @register(
#         "remove_target",
#         "Remove a specific sound source from a WAV audio segment. "
#         f"The label must be one of the fixed supported values: {', '.join(_SEPARATION_LABELS)}.",
#     )
#     def _apply_remove_target(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
#         params = {
#             "audio_path": str(audio_path),
#             "target_description": rng.choice(_SEPARATION_LABELS),
#         }
#         result = _run_tool_subprocess("remove_target", params)
#         final_path = _finalize(Path(result["output_path"]), output_path)
#         return params, final_path


# ---------------------------------------------------------------------------
# audio_edit ops: noise mixing / padding / background-event insertion.
# ---------------------------------------------------------------------------

# Disabled: add_noise/pad_noise only ever mix in generic noise (busywork, not
# a meaningful edit); insert_event was disabled at the user's request too.
# Kept here (commented) rather than deleted in case they're wanted again later.
#
# if _HAS_AUDIO_EDIT:
#     _INSERT_EVENT_LABELS = ["Dog", "Bird", "Wind", "Rain", "Siren", "Traffic noise", "Cat", "Vehicle"]
#
#     @register("add_noise", "Mix Gaussian (or a real recording's) noise into the whole clip at a target SNR.")
#     def _apply_add_noise(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
#         params = {
#             "audio_path": str(audio_path),
#             "snr_db": round(rng.uniform(5.0, 20.0), 1),
#             "output_path": str(output_path),
#         }
#         result = _ae_add_noise(**params)
#         params.pop("output_path")
#         return params, Path(result["output_path"])
#
#     @register("pad_noise", "Extend the clip with noise padding at the start, end, or both.")
#     def _apply_pad_noise(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
#         params = {
#             "audio_path": str(audio_path),
#             "position": rng.choice(["start", "end", "both"]),
#             "duration_sec": round(rng.uniform(0.3, 1.5), 2),
#             "output_path": str(output_path),
#         }
#         result = _ae_pad_noise(**params)
#         params.pop("output_path")
#         return params, Path(result["output_path"])
#
#     @register("insert_event", "Insert a labeled AudioSet background sound event at a timestamp.")
#     def _apply_insert_event(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
#         timestamp = round(rng.uniform(0.0, max(0.0, duration - 0.2)), 2)
#         params = {
#             "audio_path": str(audio_path),
#             "label": rng.choice(_INSERT_EVENT_LABELS),
#             "timestamp": format_timestamp(timestamp),
#             "snr_db": round(rng.uniform(0.0, 10.0), 1),
#             "output_path": str(output_path),
#         }
#         result = _ae_insert_event(**params)
#         params.pop("output_path")
#         return params, Path(result["output_path"])


def available_tool_names() -> list[str]:
    return sorted(REGISTRY.keys())


def describe_available_tools() -> str:
    lines = [f"- {name}: {spec.description}" for name, spec in sorted(REGISTRY.items())]
    return "\n".join(lines)
