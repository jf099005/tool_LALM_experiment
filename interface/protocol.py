"""Prompt/message protocol for the tool-calling interface.

Mirrors the multi-turn convention `tool_use_training/gen_1st_stage_data/build_dataset.py`
teaches the model during SFT (see `to_swift_sample`): one assistant turn per
tool call, rendered as a single JSON object
(`tool_name` / `parameters` / `output_audio_id`), followed by a "tool" turn
that shows the real output tagged with the id the call itself declared, and a
final `{"done": true}` turn to close the chain. `testing_tool_use_benchmark/run_eval.py`
imports `parse_turn`/`AUDIO_TOKEN` from here directly and drives the eval-time
sibling of this same convention for its fixed source/target benchmark (its
own system prompt and per-turn scoring, since it has a ground-truth chain to
compare against); this module (together with `agent.py`) generalizes the
convention to an arbitrary natural-language instruction over one or more
input audios, and describes the *real* tools in `tools/` (via
`tools.generate_tool_descriptions`) rather than the reduced, randomly
parameterized set `tools/synthetic_registry.py` exposes for synthesizing
training data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import TOOL_NAME_TO_CLASS, generate_tool_descriptions  # noqa: E402

AUDIO_TOKEN = "<audio>"


def audio_tag(audio_id: str) -> str:
    return f"<{audio_id}>{AUDIO_TOKEN}"


def audio_to_audio_tool_names() -> List[str]:
    """Every tool in `tools.TOOL_NAME_TO_CLASS` whose result is a new audio.

    The agent's default tool set: audio-to-text tools (currently only `asr`) and
    any other non-audio-output tool are excluded, since the agent only ever needs
    to hand the model's next turn a new `<audio_id><audio>` tag to keep chaining
    calls on. Single source of truth for this restriction -- both
    `build_system_prompt`'s default catalogue and `agent.ToolCallingAgent`'s
    call-time allowlist derive from this, so a tool can't be advertised without
    also being callable (or vice versa).
    """
    return [name for name, cls in TOOL_NAME_TO_CLASS.items() if cls.produces_audio()]


def build_system_prompt(tool_names: Optional[List[str]] = None) -> str:
    """Render the tool-calling protocol + tool catalogue as a system prompt.

    `tool_names` restricts the catalogue (and, implicitly, what the model is
    told is available) to a subset -- e.g. only what's actually importable in
    the current environment. Defaults to `audio_to_audio_tool_names()` (every
    tool except text-only ones like `asr`), not literally every registered tool.
    """
    if tool_names is None:
        tool_names = audio_to_audio_tool_names()
    classes = [TOOL_NAME_TO_CLASS[name] for name in tool_names]
    tools_block = generate_tool_descriptions(classes)

    return (
        "You are an audio-editing assistant with access to a fixed set of tools. "
        "You are given one or more input audios (tagged <audio_0>, <audio_1>, ...) and a "
        "task instruction. Infer the chain of tool calls needed to complete the task.\n\n"
        "Respond with exactly one JSON object per turn and nothing else:\n"
        '  {"tool_name": "<tool name>", "parameters": {"audio_id": "<id of the audio to read>", '
        '<other tool arguments>}, "output_audio_id": "<a new id for this call\'s output>"}\n\n'
        "`audio_id` must refer to an audio you have already seen -- one of the input audios or "
        "an `output_audio_id` you declared in an earlier turn. If omitted, it defaults to "
        "whichever audio your previous turn most recently produced (or the sole input audio, "
        "on your first turn). After each tool call you will be shown its real result -- a new "
        "audio -- tagged with the id you gave it, before your next turn. Always give each call "
        "a fresh output id, distinct from every id used so far. Once the task is fully "
        'complete, respond with {"done": true} instead of another tool call.\n\n'
        f"Available tools:\n{tools_block}"
    )


def render_user_prompt(instruction: str, audio_ids: List[str]) -> str:
    tags = "\n".join(f"{audio_id}: {audio_tag(audio_id)}" for audio_id in audio_ids)
    return f"{instruction}\n\nInput audio:\n{tags}"


def render_tool_call_json(tool_name: str, parameters: Dict[str, Any], output_audio_id: str) -> str:
    """Render one assistant turn's tool-call JSON -- the exact shape `parse_turn`
    expects back. Used by ground-truth authoring (`gen_1st_stage_data/build_dataset.py`),
    not by the live agent (there, this text comes from the model itself)."""
    return json.dumps(
        {"tool_name": tool_name, "parameters": parameters, "output_audio_id": output_audio_id},
        ensure_ascii=False,
    )


def render_done_json() -> str:
    """Render the closing `{"done": true}` assistant turn."""
    return json.dumps({"done": True})


def render_tool_result_message(
    step_index: int, tool_name: str, audio_tags: List[str], text: Optional[str] = None
) -> str:
    """Render the "tool" turn shown after executing one call.

    Single source of truth for this wording -- `gen_1st_stage_data/build_dataset.py`
    bakes it verbatim into SFT training data, so `agent.py` (live inference) and
    `testing_tool_use_benchmark/run_eval.py` (eval) must render it identically or
    a trained model sees an out-of-distribution tool turn at inference time.

    Takes already-rendered tag strings (not raw audio ids) so callers stay free to
    use their own token spelling -- `build_dataset.py`'s `to_swift_sample` has a
    configurable `audio_token` override that a hardcoded call to `audio_tag()` here
    would silently ignore. `audio_tags` empty means a text-only result (e.g. a
    transcript); non-empty means one tag per produced audio, in order.
    """
    if audio_tags:
        return f"Output of step {step_index}: " + " ".join(audio_tags)
    return f"Output of step {step_index} ({tool_name}): {text}"


def parse_turn(raw_text: str) -> Optional[Dict[str, Any]]:
    """Parse one model turn as a JSON object.

    Returns the parsed dict as-is -- callers distinguish a tool call
    (`tool_name` present) from an explicit stop signal (`done` present) from
    garbage (neither key present). Returns None if the turn isn't even valid
    JSON, tolerating a ```json ... ``` fence some models wrap their output in.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None
