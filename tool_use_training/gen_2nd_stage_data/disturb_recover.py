"""Disturbance <-> recovery mapping table for stage-2 tool-use training data.

Stage 1 (`gen_1st_stage_data/`) teaches the model to *reproduce* an arbitrary target
audio B from a source A by calling tools. Stage 2 teaches something narrower and more
benchmark-relevant: given a QA pair whose audio has been degraded by a data-augmentation
op (noise, padding, a stray background event, a speed/pitch change), call the *right*
tool(s) to clean it back up so the question is answerable again -- the same move a model
should make at inference time against MMAU/DCASE-style benchmarks when the audio is
noisy or otherwise hard to parse directly.

This module owns exactly one thing: the *table* pairing each disturbance op (from
`audio_edit/`) with one or more recovery strategies (tools from `tools/`), plus the
functions that generate disturbance parameters and, from those, the matching recovery
parameters. `build_by_disturb.py` drives the actual dataset construction; see
`DISTURB_RECOVER_MAPPING.md` (same directory) for the human-readable table + rationale.

Design invariant relied on by the recovery-chain builder: every recovery op here either
exactly restores the pre-disturbance duration (clipping undoes pad_noise's length
change; time_stretch's inverse rate undoes change_speed's length change) or leaves
duration untouched entirely (denoise, remove_target, pitch_shift). So when a chain of
several disturbance ops is undone in reverse, each recovery step sees the exact
timeline (durations, onsets) its matching forward step originally produced -- content
fidelity is best-effort (these are lossy augmentations), but timing bookkeeping never
drifts.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audio_edit.editor import edit_audio  # noqa: E402


# ---------------------------------------------------------------------------
# Small local helpers.
# ---------------------------------------------------------------------------

def get_duration_seconds(path: Path) -> float:
    """Read a clip's duration, tolerant of real-world WAV/audio quirks.

    Real corpora (this repo has hit it with the 2025 DCASE AudioQA audio) contain
    "WAV" files the stdlib `wave` module can't parse -- WAVE_FORMAT_EXTENSIBLE,
    float/24-bit PCM, odd chunk ordering, or a `.wav` extension on a different
    container/codec entirely -- even though the audio is perfectly valid. `wave`
    only understands plain PCM RIFF/WAVE and raises `Error: file does not start
    with RIFF id` (or similar) on anything else. `soundfile` (libsndfile) reads
    the header for many more WAV variants without decoding the whole file;
    `librosa` (ffmpeg/audioread-backed) falls back further for other
    containers/codecs a `.wav` file might actually be. Only a genuinely
    unreadable file (e.g. an HTML error page saved with a `.wav` extension --
    also seen in this corpus) fails all three, and propagates so the caller's
    retry/give-up logic can skip that sample.
    """
    path = Path(path)
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        pass

    import librosa

    return float(librosa.get_duration(path=str(path)))


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm, flooring so re-parsing never overshoots."""
    total_millis = int((max(0.0, seconds) - 1e-6) * 1000)
    total_millis = max(0, total_millis)
    whole, millis = divmod(total_millis, 1000)
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


# ---------------------------------------------------------------------------
# Which conda env each recovery tool needs when *this script* executes it
# in-process via subprocess dispatch (`tools/tool_batch_execute.py`). This is
# independent of `apply_tools.py`'s own `TOOL_ENV` (used only if someone
# re-runs the emitted schedule JSON through that script instead) -- both just
# need to point at an env with the right deps; they don't need to agree.
# ---------------------------------------------------------------------------
TOOL_ENV: Dict[str, str] = {
    "clipping": "/home/u1501463/miniconda3/envs/Whisper/bin/python",
    "denoise": "/home/u1501463/miniconda3/envs/Whisper/bin/python",
    "pitch_shift": "/home/u1501463/miniconda3/envs/Whisper/bin/python",
    "time_stretch": "/home/u1501463/miniconda3/envs/Whisper/bin/python",
    "trim_silence": "/home/u1501463/miniconda3/envs/Whisper/bin/python",
    "remove_target": "/home/u1501463/miniconda3/envs/sam_audio/bin/python",
}


# ---------------------------------------------------------------------------
# Disturbance registry: name -> (description, apply-fn). Every apply-fn has
# the same shape as `tools/synthetic_registry.py`'s:
#   params, output_path = fn(audio_path, output_path, rng, duration)
# `params` is the exact kwargs passed to `audio_edit.editor.edit_audio`, so it
# doubles as the ground-truth record of what was done (and everything the
# matching recovery strategy needs to invert it).
# ---------------------------------------------------------------------------

@dataclass
class DisturbSpec:
    description: str
    apply: Callable[[Path, Path, random.Random, float], Tuple[Dict[str, Any], Path]]


DISTURB_REGISTRY: Dict[str, DisturbSpec] = {}


def register_disturb(name: str, description: str):
    def _decorator(fn):
        DISTURB_REGISTRY[name] = DisturbSpec(description=description, apply=fn)
        return fn
    return _decorator


@register_disturb("add_noise", "Mix white/pink noise across the whole clip at a target SNR.")
def _disturb_add_noise(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
    params = {
        "noise_type": rng.choice(["white", "pink"]),
        "snr_db": round(rng.uniform(3.0, 15.0), 1),
    }
    result = edit_audio("add_noise", str(audio_path), str(output_path), **params)
    return params, Path(result["output_path"])


@register_disturb("pad_noise", "Extend the clip with noise padding at the start, end, or both.")
def _disturb_pad_noise(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
    params = {
        "position": rng.choice(["start", "end", "both"]),
        "duration_sec": round(rng.uniform(0.5, 2.0), 2),
        "noise_type": rng.choice(["white", "pink"]),
        "snr_db": round(rng.uniform(10.0, 25.0), 1),
    }
    result = edit_audio("pad_noise", str(audio_path), str(output_path), **params)
    return params, Path(result["output_path"])


_INSERT_EVENT_LABELS = ["Dog", "Bird", "Wind", "Rain", "Siren", "Traffic noise", "Cat", "Vehicle"]


@register_disturb("insert_background_event", "Insert a labeled AudioSet background sound event at a timestamp.")
def _disturb_insert_event(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
    label = rng.choice(_INSERT_EVENT_LABELS)
    timestamp_sec = round(rng.uniform(0.0, max(0.0, duration - 0.2)), 2)
    params = {
        "label": label,
        "timestamp": format_timestamp(timestamp_sec),
        "snr_db": round(rng.uniform(0.0, 8.0), 1),
    }
    result = edit_audio("insert_background_event", str(audio_path), str(output_path), **params)
    # Carry the resolved onset/offset (seconds) forward -- needed by the
    # edge-clip recovery strategy below -- without disturbing `params`, which
    # stays exactly what was passed to `edit_audio` (i.e. reusable verbatim as
    # a `insert_background_event` tool call).
    extra = {"onset": result["onset"], "offset": result["offset"]}
    return {**params, **extra}, Path(result["output_path"])


@register_disturb("change_speed", "Time-stretch the clip (speed up or slow down) without changing pitch.")
def _disturb_change_speed(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
    rate = rng.choice([0.8, 0.9, 1.1, 1.25])
    params = {"rate": rate}
    result = edit_audio("change_speed", str(audio_path), str(output_path), **params)
    return params, Path(result["output_path"])


@register_disturb("change_pitch", "Pitch-shift the clip without changing its duration.")
def _disturb_change_pitch(audio_path: Path, output_path: Path, rng: random.Random, duration: float):
    n_steps = rng.choice([-5, -3, 3, 5])
    params = {"n_steps": n_steps}
    result = edit_audio("change_pitch", str(audio_path), str(output_path), **params)
    return params, Path(result["output_path"])


def available_disturb_names() -> List[str]:
    return sorted(DISTURB_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Recovery registry: disturb-op name -> list of (strategy_name, applicable?,
# builder). `builder(disturb_params, rng) -> (tool_name, parameters)` returns
# the tool call that undoes (or best-effort cleans up) that disturbance;
# `parameters` excludes audio_path/output_path, which the caller fills in per
# step. `applicable(disturb_params)` lets a strategy opt out (e.g. the
# edge-clip recovery for an inserted event only makes sense if the event
# actually landed near a boundary).
# ---------------------------------------------------------------------------

RecoveryBuilder = Callable[[Dict[str, Any], random.Random], Tuple[str, Dict[str, Any]]]


@dataclass
class RecoveryStrategy:
    name: str
    tool_name: str
    build: RecoveryBuilder
    applicable: Callable[[Dict[str, Any]], bool] = field(default=lambda params: True)


RECOVERY_STRATEGIES: Dict[str, List[RecoveryStrategy]] = {}


def register_recovery(disturb_op: str, strategy: RecoveryStrategy) -> None:
    RECOVERY_STRATEGIES.setdefault(disturb_op, []).append(strategy)


# -- add_noise: broadband noise mixed across the whole clip -> denoise (a few
#    algorithm choices; none is an exact inverse since the noise was additive
#    and not separately recorded, so several plausible cleanups exist). ------

def _recover_add_noise_spectral(disturb_params: Dict[str, Any], rng: random.Random):
    return "denoise", {
        "algorithm": "spectral_subtraction",
        "noise_factor": round(rng.uniform(1.5, 2.5), 2),
    }


def _recover_add_noise_wiener(disturb_params: Dict[str, Any], rng: random.Random):
    return "denoise", {"algorithm": "wiener"}


def _recover_add_noise_adaptive(disturb_params: Dict[str, Any], rng: random.Random):
    return "denoise", {
        "algorithm": "adaptive",
        "sensitivity": round(rng.uniform(0.3, 0.7), 2),
    }


register_recovery("add_noise", RecoveryStrategy("denoise_spectral_subtraction", "denoise", _recover_add_noise_spectral))
register_recovery("add_noise", RecoveryStrategy("denoise_wiener", "denoise", _recover_add_noise_wiener))
register_recovery("add_noise", RecoveryStrategy("denoise_adaptive", "denoise", _recover_add_noise_adaptive))


# -- pad_noise: noise appended before/after the real audio -> clipping cuts
#    the pad off exactly (position/duration_sec fully determine the boundary,
#    so this is an exact inverse up to the fade region). trim_silence is an
#    approximate alternative that doesn't need the exact pad length, since the
#    pad is quiet relative to the clip by construction (see pad_noise's
#    default 20 dB SNR). ------------------------------------------------------

def _recover_pad_noise_clip(disturb_params: Dict[str, Any], rng: random.Random):
    position = disturb_params["position"]
    duration_sec = disturb_params["duration_sec"]
    input_duration = disturb_params["_input_duration"]
    pad_start = duration_sec if position in ("start", "both") else 0.0
    begin = format_timestamp(pad_start)
    end = format_timestamp(pad_start + input_duration)
    return "clipping", {"audio_begin": begin, "audio_end": end}


def _recover_pad_noise_trim_silence(disturb_params: Dict[str, Any], rng: random.Random):
    output_duration = disturb_params["_output_duration"]
    return "trim_silence", {
        "audio_begin": format_timestamp(0.0),
        "audio_end": format_timestamp(output_duration),
        "threshold_db": rng.choice([20.0, 25.0, 30.0, 35.0]),
    }


register_recovery("pad_noise", RecoveryStrategy("clip_off_padding", "clipping", _recover_pad_noise_clip))
register_recovery("pad_noise", RecoveryStrategy("trim_silence_heuristic", "trim_silence", _recover_pad_noise_trim_silence))


# -- insert_background_event: a labeled real-world event mixed in at an onset
#    -> remove_target (source separation) always applies; clipping out the
#    contaminated edge is a cheaper alternative but only valid when the event
#    landed close enough to a boundary that clipping it off doesn't also
#    discard the bulk of the original content. ------------------------------

# AudioSet event labels the disturbance draws from -> the closest label in
# tools/extract_remove_target.py's fixed SEPARATION_LABELS vocabulary. There
# is no exact match (the separation model's labels are broad source classes,
# not AudioSet ontology classes), so this is a best-effort lookup, defaulting
# to "background sound" for anything unmapped.
EVENT_LABEL_TO_SEPARATION_TARGET: Dict[str, str] = {
    "Dog": "background sound",
    "Bird": "background sound",
    "Cat": "background sound",
    "Wind": "noise",
    "Rain": "noise",
    "Traffic noise": "noise",
    "Siren": "background sound",
    "Vehicle": "background sound",
}

# The edge-clip recovery keeps whichever side of the event (before onset, or
# after offset) is longer, and only applies at all if that side still holds on
# to most of the clip -- otherwise clipping "recovers" a sliver and throws away
# the bulk of the original content, which is worse than just leaving the event
# in. Note this is about the *event's* span relative to the clip, not just
# which edge it touches: an event that starts near the middle and runs to the
# very end is "near the end boundary" by onset/offset alone but still spans
# most of the clip, so it must be rejected too.
_MIN_PRESERVED_FRACTION = 0.6


def _recover_insert_event_remove_target(disturb_params: Dict[str, Any], rng: random.Random):
    target = EVENT_LABEL_TO_SEPARATION_TARGET.get(disturb_params["label"], "background sound")
    return "remove_target", {"target_description": target}


def _edge_clip_bounds(disturb_params: Dict[str, Any]) -> Tuple[float, float, float, float]:
    duration = disturb_params["_output_duration"]
    onset, offset = disturb_params["onset"], disturb_params["offset"]
    keep_before, keep_after = onset, duration - offset
    if keep_before >= keep_after:
        begin, end = 0.0, onset
    else:
        begin, end = offset, duration
    return begin, end, max(keep_before, keep_after), duration


def _insert_event_edge_clip_applicable(disturb_params: Dict[str, Any]) -> bool:
    duration = disturb_params["_output_duration"]
    if duration <= 0:
        return False
    _, _, kept, duration = _edge_clip_bounds(disturb_params)
    return kept >= _MIN_PRESERVED_FRACTION * duration


def _recover_insert_event_edge_clip(disturb_params: Dict[str, Any], rng: random.Random):
    begin, end, _, _ = _edge_clip_bounds(disturb_params)
    return "clipping", {"audio_begin": format_timestamp(begin), "audio_end": format_timestamp(end)}


register_recovery(
    "insert_background_event",
    RecoveryStrategy("remove_target_by_label", "remove_target", _recover_insert_event_remove_target),
)
register_recovery(
    "insert_background_event",
    RecoveryStrategy("edge_clip", "clipping", _recover_insert_event_edge_clip, applicable=_insert_event_edge_clip_applicable),
)


# -- change_speed / change_pitch: parametric, invertible transforms -> the
#    exact inverse rate / semitone count. ------------------------------------

def _recover_change_speed(disturb_params: Dict[str, Any], rng: random.Random):
    rate = disturb_params["rate"]
    return "time_stretch", {"rate": round(1.0 / rate, 6)}


def _recover_change_pitch(disturb_params: Dict[str, Any], rng: random.Random):
    n_steps = disturb_params["n_steps"]
    return "pitch_shift", {"n_steps": -n_steps}


register_recovery("change_speed", RecoveryStrategy("inverse_time_stretch", "time_stretch", _recover_change_speed))
register_recovery("change_pitch", RecoveryStrategy("inverse_pitch_shift", "pitch_shift", _recover_change_pitch))


def build_recovery_step(
    disturb_op: str,
    disturb_params: Dict[str, Any],
    rng: random.Random,
) -> Tuple[str, str, Dict[str, Any]]:
    """Pick one applicable recovery strategy for `disturb_op` and build its call.

    Returns (strategy_name, tool_name, parameters); parameters excludes
    audio_path/output_path, which the caller (build_by_disturb.py) fills in
    per step since they depend on where that step sits in the chain.
    """
    strategies = [s for s in RECOVERY_STRATEGIES.get(disturb_op, []) if s.applicable(disturb_params)]
    if not strategies:
        raise ValueError(f"No applicable recovery strategy registered for disturb op '{disturb_op}'")
    strategy = rng.choice(strategies)
    tool_name, parameters = strategy.build(disturb_params, rng)
    return strategy.name, tool_name, parameters


def describe_mapping() -> str:
    """Human-readable disturb-op -> recovery-strategy table (used by --describe)."""
    lines: List[str] = []
    for op_name in sorted(DISTURB_REGISTRY):
        spec = DISTURB_REGISTRY[op_name]
        lines.append(f"{op_name}: {spec.description}")
        strategies = RECOVERY_STRATEGIES.get(op_name, [])
        if not strategies:
            lines.append("  (no recovery strategy registered)")
        for strategy in strategies:
            lines.append(f"  -> {strategy.tool_name} [{strategy.name}]")
        lines.append("")
    return "\n".join(lines).strip()
