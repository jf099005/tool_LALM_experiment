"""Chain-of-thought sentences for the ground-truth answer, one per tool call.

Each tool call in a chain gets one short reasoning sentence, in the spirit of
"compare the current audio to the target, notice X, therefore apply tool T
with parameters P" -- e.g. for `time_stretch`:

    Comparing the current audio to the target, the target sounds faster and
    its duration is shorter. To match the target's tempo, I need to apply
    time_stretch with rate=1.2.
    <tool_call>
    {"name": "time_stretch", "arguments": {"rate": 1.2, "audio_id": "audio_0"}}
    </tool_call>

`render_cot` picks one of several templates per tool at random (so the
dataset doesn't collapse onto one repeated phrasing, same rationale as
`question_templates.py`) and fills it in from that step's actual ground-truth
parameters -- there's no real "listening" happening, the sentence is derived
straight from the known parameter values (e.g. `rate > 1` => "faster").

Only covers tools this project currently trains on (see
`tools.tools_registry.available_tool_names()`); an unknown tool name falls
back to a generic template so a newly `@register`-ed tool doesn't crash
generation, just produces a less specific sentence until someone adds
tool-specific templates for it here.

`build_dataset.py::build_one_sample` calls this once per step and stores the
result as that step's `cot` field; `to_swift_sample`'s `use_cot` flag decides
whether it actually gets rendered into the assistant turn.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List


def _time_stretch_context(parameters: Dict[str, Any]) -> Dict[str, Any]:
    rate = float(parameters["rate"])
    faster = rate > 1.0
    return {
        "rate": rate,
        "speed_word": "faster" if faster else "slower",
        "duration_word": "shorter" if faster else "longer",
        "verb": "speed up" if faster else "slow down",
    }


def _pitch_shift_context(parameters: Dict[str, Any]) -> Dict[str, Any]:
    n_steps = parameters["n_steps"]
    return {
        "n_steps": n_steps,
        "pitch_word": "higher" if n_steps > 0 else "lower",
        "magnitude": abs(n_steps),
    }


def _clipping_context(parameters: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "audio_begin": parameters["audio_begin"],
        "audio_end": parameters["audio_end"],
    }


_DENOISE_ALGORITHM_LABELS = {
    "spectral_subtraction": "spectral subtraction",
    "wiener": "Wiener filtering",
    "echo_cancellation": "echo cancellation",
    "adaptive": "adaptive noise gating",
}


def _denoise_context(parameters: Dict[str, Any]) -> Dict[str, Any]:
    algorithm = parameters["algorithm"]
    return {
        "algorithm": algorithm,
        "algorithm_label": _DENOISE_ALGORITHM_LABELS.get(algorithm, algorithm),
    }


def _pre_emphasis_context(parameters: Dict[str, Any]) -> Dict[str, Any]:
    return {"coef": parameters.get("coef", 0.97)}


def _trim_silence_context(parameters: Dict[str, Any]) -> Dict[str, Any]:
    return {"threshold_db": parameters.get("threshold_db", 40)}


_CONTEXT_BUILDERS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "time_stretch": _time_stretch_context,
    "pitch_shift": _pitch_shift_context,
    "clipping": _clipping_context,
    "denoise": _denoise_context,
    "pre_emphasis": _pre_emphasis_context,
    "trim_silence": _trim_silence_context,
}


COT_TEMPLATES: Dict[str, List[str]] = {
    "time_stretch": [
        "Comparing the current audio to the target, the target sounds {speed_word} and its "
        "duration is {duration_word}. To match the target's tempo, I need to apply time_stretch "
        "with rate={rate}.",

        "Listening to both clips, the pacing differs: the target plays back {speed_word} than "
        "the current audio. Time-stretching by a factor of {rate} should {verb} the current "
        "audio to align them.",

        "The main difference between the current audio and the target is speed, not pitch or "
        "content -- the target is {speed_word}. Apply time_stretch(rate={rate}) to close that "
        "gap.",
    ],
    "pitch_shift": [
        "Comparing the current audio to the target, the target's pitch sounds {pitch_word}, "
        "roughly by {magnitude} semitone(s). Apply pitch_shift with n_steps={n_steps} to "
        "correct it.",

        "Listening closely, the tonal register of the target is {pitch_word} relative to the "
        "current audio, while the tempo and content are unchanged. Shifting the pitch by "
        "n_steps={n_steps} should align them.",

        "The current audio and the target differ mainly in pitch -- the target sounds "
        "{pitch_word}. pitch_shift(n_steps={n_steps}) should correct that difference.",
    ],
    "clipping": [
        "The target audio is a shorter segment of the current audio, running only from "
        "{audio_begin} to {audio_end}. I should clip the current audio to that window.",

        "Comparing durations, the target is missing the parts of the current audio outside "
        "{audio_begin}-{audio_end}. Apply clipping with audio_begin={audio_begin} and "
        "audio_end={audio_end}.",

        "The content in the target matches only a portion of the current audio, between "
        "{audio_begin} and {audio_end}. I need to extract that segment with the clipping tool.",
    ],
    "denoise": [
        "The current audio has noticeably more background noise than the target, which sounds "
        "cleaner. Apply denoise using the {algorithm_label} algorithm (algorithm={algorithm}) "
        "to reduce it.",

        "Comparing the two clips, the target is much quieter in the background -- the noise "
        "floor present in the current audio has been suppressed. I should call denoise with "
        "algorithm={algorithm}.",

        "The target audio sounds clearer and less noisy than the current audio. Denoising with "
        "the {algorithm_label} method should remove that residual noise.",
    ],
    "pre_emphasis": [
        "The target audio has more emphasized high-frequency content than the current audio, "
        "sounding crisper. Apply pre_emphasis with coef={coef} to boost the highs.",

        "Comparing the spectral balance, the target's high frequencies are more pronounced than "
        "the current audio's. A pre_emphasis filter with coef={coef} should match that.",

        "The current audio sounds duller than the target. Applying pre_emphasis(coef={coef}) "
        "should sharpen the high end to match.",
    ],
    "trim_silence": [
        "The target audio has the silent portions at the start/end removed compared to the "
        "current audio. Apply trim_silence with threshold_db={threshold_db} to cut them out.",

        "Comparing the two clips, the current audio has extra silence at its edges that the "
        "target doesn't have. Trimming with threshold_db={threshold_db} should remove it.",

        "The target begins and ends right where the current audio's actual content starts and "
        "stops, with no leading/trailing silence. Apply trim_silence(threshold_db={threshold_db}) "
        "to match it.",
    ],
}


_GENERIC_TEMPLATES: List[str] = [
    "Comparing the current audio to the target, applying {tool_name} with these parameters "
    "should account for the remaining difference.",
]


def render_cot(tool_name: str, parameters: Dict[str, Any], rng: random.Random) -> str:
    """Pick and fill in one random chain-of-thought sentence for one tool call.

    `parameters` is the same ground-truth argument dict recorded for that
    step (before `audio_id`/`output_audio_id` bookkeeping keys matter -- only
    the tool's own parameters are read here).
    """
    templates = COT_TEMPLATES.get(tool_name)
    if not templates:
        return rng.choice(_GENERIC_TEMPLATES).format(tool_name=tool_name)

    context = _CONTEXT_BUILDERS[tool_name](parameters)
    return rng.choice(templates).format(**context)
