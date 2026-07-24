"""Config-driven appliers for the real `tools/`-package tools (not `audio_edit/` synthetic
disturbance ops -- see build_dataset.py's module docstring for how this differs from
gen_2nd_stage_data's disturb -> recover strategy).

Each tool gets one applier function of the shape used throughout this repo's dataset
generators (`tools/tools_registry.py`, `gen_2nd_stage_data/disturb_recover.py`):

    params, output_path = APPLIERS[name](tool_cfg, audio_path, output_path, rng, duration)

`tool_cfg` is that tool's sub-dict from tool_config.json (`{"enabled", "weight", "params",
...}`); `params` is the exact argument dict passed to the tool's `execute()`, so it doubles
as the ground-truth record of what was applied. Parameter *values* are drawn here from
`tool_cfg["params"]`'s ranges (see tool_config.json's top-level comment for the spec
format); audio_begin/audio_end for whole-clip tools and clipping's cut point are computed
from the real duration, not drawn from config, since they depend on the specific clip.

Light tools (clipping/denoise/pitch_shift/time_stretch/the normalize family) run in-process
-- this module needs the same librosa/soundfile/scipy env `gen_2nd_stage_data`'s scripts
require. Heavy ML tools (remove_target/extract_target/human_voice_enhance/super_resolution)
are dispatched one call at a time via `tools/tool_batch_execute.py` in that tool's own conda
env (`tool_cfg["env"]`), same split as `tools_registry._run_tool_subprocess` -- unlike
`build_by_disturb.py`'s recovery-chain execution, calls here are not batched across samples,
since this strategy applies one independent tool per sample rather than a shared multi-turn
chain (heavy tools are disabled by default in tool_config.json for exactly this reason: turn
them on only once the target conda env is confirmed present).
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.clipping import ClippingTool  # noqa: E402
from tools.denoise import DenoiseTool  # noqa: E402
from tools.normalize import (  # noqa: E402
    AmplitudeNormalizeTool,
    DCOffsetRemovalTool,
    LoudnessNormalizeTool,
    PreEmphasisTool,
    SpectralNormalizeTool,
    TrimSilenceTool,
)
from tools.pitch_time import PitchShiftTool, TimeStretchTool  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers (duplicated, rather than imported, from tools/tools_registry.py
# and gen_2nd_stage_data/disturb_recover.py -- kept local so this stage's directory
# stays self-contained, matching how those two already duplicate the same helpers
# rather than cross-importing).
# ---------------------------------------------------------------------------

def format_timestamp(seconds: float) -> str:
    total_millis = int((max(0.0, seconds) - 1e-6) * 1000)
    total_millis = max(0, total_millis)
    whole, millis = divmod(total_millis, 1000)
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def get_duration_seconds(path: Path) -> float:
    """Tolerant duration read -- see disturb_recover.get_duration_seconds for why
    stdlib wave alone isn't enough for real-world corpora."""
    path = Path(path)
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        pass

    import librosa

    return float(librosa.get_duration(path=str(path)))


def _is_plain_pcm_wav(path: Path) -> bool:
    """Whether stdlib `wave` can open this file. Real corpora (this repo has hit it with
    the 2025 DCASE AudioQA audio) contain '.wav' files in WAV_FORMAT_EXTENSIBLE or other
    variants `wave` can't parse even though the audio itself is perfectly valid --
    `normalize.py`/`clipping.py` use stdlib `wave` directly (unlike denoise.py's tolerant
    3-backend loader), so left unconverted these fail every tool call, not just once."""
    try:
        with wave.open(str(path), "rb"):
            return True
    except Exception:
        return False


def ensure_wav(path: Path, work_dir: Path) -> Path:
    """Return a plain PCM16 mono WAV copy of `path`, re-encoding whenever the source
    isn't already one -- either a non-WAV format, or a WAV variant stdlib `wave` can't
    parse (see `_is_plain_pcm_wav`). Most tools here only accept WAV input (see
    clipping.py/denoise.py's explicit checks)."""
    path = Path(path)
    if path.suffix.lower() == ".wav" and _is_plain_pcm_wav(path):
        return path

    import librosa
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
    """Move a tool's auto-named output to the chain's desired file name.

    `shutil.move` (not `Path.replace`/`os.rename`) because some tools -- clipping.py in
    particular -- always write next to their *input* file regardless of any requested
    output path; when that input is still the original source audio (chain step 1, before
    anything has been copied under --output-dir), the produced file can sit on a different
    filesystem than the destination, and a plain rename fails with EXDEV.
    """
    produced_path = Path(produced_path)
    desired_path = Path(desired_path)
    if produced_path == desired_path:
        return desired_path
    desired_path.parent.mkdir(parents=True, exist_ok=True)
    if desired_path.exists():
        desired_path.unlink()
    shutil.move(str(produced_path), str(desired_path))
    return desired_path


def _full_segment(duration: float) -> Tuple[str, str]:
    return format_timestamp(0.0), format_timestamp(duration)


# ---------------------------------------------------------------------------
# Parameter draw engine, driven entirely by tool_config.json's per-tool `params`.
# ---------------------------------------------------------------------------

def _draw_value(spec: Dict[str, Any], rng: random.Random) -> Any:
    kind = spec["kind"]
    if kind == "choice":
        return rng.choice(spec["values"])
    if kind == "uniform":
        value = rng.uniform(spec["min"], spec["max"])
        if "round" in spec:
            value = round(value, spec["round"])
        return value
    if kind == "randint":
        return rng.randint(spec["min"], spec["max"])
    if kind == "fixed":
        return spec["value"]
    raise ValueError(f"Unknown param spec kind: {kind!r}")


def draw_params(tool_cfg: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Draw one dict of {param_name: value} from a tool's tool_config.json `params`.

    Dict order matters: a param with `only_if` is skipped unless the param it
    references was *already* drawn (earlier in the same dict) with a matching value --
    e.g. denoise's `noise_factor`/`sensitivity` only fire for their matching `algorithm`.
    """
    drawn: Dict[str, Any] = {}
    for name, spec in tool_cfg.get("params", {}).items():
        only_if = spec.get("only_if")
        if only_if and drawn.get(only_if["param"]) not in only_if["values"]:
            continue
        drawn[name] = _draw_value(spec, rng)
    return drawn


# ---------------------------------------------------------------------------
# Heavy-tool subprocess dispatch (mirrors tools_registry._run_tool_subprocess).
# ---------------------------------------------------------------------------

def _run_tool_subprocess(tool_name: str, env_python: str, params: Dict[str, Any]) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = Path(tmp_dir) / "input.json"
        output_path = Path(tmp_dir) / "output.json"
        input_path.write_text(json.dumps([params]), encoding="utf-8")
        subprocess.run(
            [
                env_python,
                str(REPO_ROOT / "tools" / "tool_batch_execute.py"),
                "--tool-name", tool_name,
                "--input-file", str(input_path),
                "--output-file", str(output_path),
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )
        results = json.loads(output_path.read_text(encoding="utf-8"))

    if not isinstance(results, list) or len(results) != 1:
        raise RuntimeError(f"Unexpected batch result for {tool_name}: {results!r}")
    result = results[0]
    if result.get("status") not in (None, "success"):
        raise RuntimeError(f"{tool_name} failed: {result.get('message') or result.get('error')}")
    return result


# ---------------------------------------------------------------------------
# Light-tool appliers (in-process).
# ---------------------------------------------------------------------------

def _apply_clipping(tool_cfg, audio_path: Path, output_path: Path, rng: random.Random, duration: float):
    drawn = draw_params(tool_cfg, rng)
    fraction = drawn.get("clip_fraction", rng.uniform(0.4, 0.9))
    end = max(min(duration, 0.5), duration * fraction)
    params = {
        "audio_path": str(audio_path),
        "audio_begin": format_timestamp(0.0),
        "audio_end": format_timestamp(end),
    }
    result = ClippingTool.execute(params)
    final_path = _finalize(Path(result["clip_path"]), output_path)
    return params, final_path


def _apply_denoise(tool_cfg, audio_path: Path, output_path: Path, rng: random.Random, duration: float):
    drawn = draw_params(tool_cfg, rng)
    params: Dict[str, Any] = {"audio_path": str(audio_path), **drawn}
    result = DenoiseTool.execute(params, str(output_path))
    return params, Path(result["output_path"])


def _apply_pitch_shift(tool_cfg, audio_path: Path, output_path: Path, rng: random.Random, duration: float):
    drawn = draw_params(tool_cfg, rng)
    params = {"audio_path": str(audio_path), **drawn}
    result = PitchShiftTool.execute(params, str(output_path))
    return params, Path(result["output_path"])


def _apply_time_stretch(tool_cfg, audio_path: Path, output_path: Path, rng: random.Random, duration: float):
    drawn = draw_params(tool_cfg, rng)
    params = {"audio_path": str(audio_path), **drawn}
    result = TimeStretchTool.execute(params, str(output_path))
    return params, Path(result["output_path"])


def _make_full_segment_applier(tool_cls):
    def _apply(tool_cfg, audio_path: Path, output_path: Path, rng: random.Random, duration: float):
        drawn = draw_params(tool_cfg, rng)
        begin, end = _full_segment(duration)
        params = {
            "audio_path": str(audio_path),
            "audio_begin": begin,
            "audio_end": end,
            **drawn,
        }
        result = tool_cls.execute(params, str(output_path))
        return params, Path(result["output_path"])

    return _apply


# Heavy tools dispatched via `_run_tool_subprocess` where the real tool requires an
# explicit `output_path` (see tools/abstract_tool.Tool.requires_output_path) --
# `remove_target`/`extract_target` don't need one and are omitted here.
_HEAVY_TOOLS_REQUIRING_OUTPUT_PATH = {"human_voice_enhance", "super_resolution"}


def _make_heavy_applier(tool_name: str):
    def _apply(tool_cfg, audio_path: Path, output_path: Path, rng: random.Random, duration: float):
        drawn = draw_params(tool_cfg, rng)
        env_python = tool_cfg["env"]
        params = {"audio_path": str(audio_path), **drawn}
        if tool_name in _HEAVY_TOOLS_REQUIRING_OUTPUT_PATH:
            params["output_path"] = str(output_path)
        result = _run_tool_subprocess(tool_name, env_python, params)
        params.pop("output_path", None)
        final_path = _finalize(Path(result["output_path"]), output_path)
        return params, final_path

    return _apply


ApplierFn = Callable[[Dict[str, Any], Path, Path, random.Random, float], Tuple[Dict[str, Any], Path]]

APPLIERS: Dict[str, ApplierFn] = {
    "clipping": _apply_clipping,
    "denoise": _apply_denoise,
    "pitch_shift": _apply_pitch_shift,
    "time_stretch": _apply_time_stretch,
    "trim_silence": _make_full_segment_applier(TrimSilenceTool),
    "amplitude_normalize": _make_full_segment_applier(AmplitudeNormalizeTool),
    "loudness_normalize": _make_full_segment_applier(LoudnessNormalizeTool),
    "remove_dc_offset": _make_full_segment_applier(DCOffsetRemovalTool),
    "spectral_normalize": _make_full_segment_applier(SpectralNormalizeTool),
    "pre_emphasis": _make_full_segment_applier(PreEmphasisTool),
    "remove_target": _make_heavy_applier("remove_target"),
    "extract_target": _make_heavy_applier("extract_target"),
    "human_voice_enhance": _make_heavy_applier("human_voice_enhance"),
    "super_resolution": _make_heavy_applier("super_resolution"),
}


def load_tool_config(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def available_tool_names(tool_config: Dict[str, Any]) -> list[str]:
    """Enabled tools that both have an applier registered and (for heavy tools whose
    config carries an `env`) whose conda env python actually exists on this machine."""
    names = []
    for name, cfg in tool_config.get("tools", {}).items():
        if not cfg.get("enabled", False):
            continue
        if name not in APPLIERS:
            continue
        env_python = cfg.get("env")
        if env_python and not Path(env_python).exists():
            continue
        names.append(name)
    return sorted(names)


def tool_weights(tool_config: Dict[str, Any], tool_names: list[str]) -> list[float]:
    tools_cfg = tool_config.get("tools", {})
    return [float(tools_cfg[name].get("weight", 1.0)) for name in tool_names]
