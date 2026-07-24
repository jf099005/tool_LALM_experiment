"""Prompt/message protocol for the tool-calling interface.

Mirrors the multi-turn convention `tool_use_training/gen_1st_stage_data/build_dataset.py`
teaches the model during SFT (see `to_swift_sample`): one assistant turn per
tool call, followed by a "tool" turn that shows the real output tagged with
the id the call itself declared, and a final turn to close the chain. The
wire convention for those turns (how a call/result/stop is spelled out as
text) is pluggable -- see `tool_call_formats.py` -- and defaults to Qwen's own
official Hermes-style `<tool_call>`/`<tools>` convention.
`testing_tool_use_benchmark/run_eval.py` imports `parse_turn`/`AUDIO_TOKEN`
from here directly and drives the eval-time sibling of this same convention
for its fixed source/target benchmark (its own system prompt and per-turn
scoring, since it has a ground-truth chain to compare against); this module
(together with `agent.py`) generalizes the convention to an arbitrary
natural-language instruction over one or more input audios. The tools it
describes/allows are `tools.tools_registry`'s curated, project-wide toolset
(the same one `tool_use_training/gen_1st_stage_data/build_dataset.py` trains
on) -- not the full universe of every `Tool` subclass under `tools/` -- so
the live agent never advertises a tool the model was never trained to call.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import TOOL_NAME_TO_CLASS, tool_function_schemas  # noqa: E402
from tools import tools_registry  # noqa: E402
from tool_call_formats import get_tool_call_format  # noqa: E402

AUDIO_TOKEN = "<audio>"
DEFAULT_TOOL_CALL_FORMAT = "qwen"


def audio_tag(audio_id: str) -> str:
    return f"<{audio_id}>{AUDIO_TOKEN}"


def audio_to_audio_tool_names() -> List[str]:
    """Every tool in `tools_registry`'s curated toolset whose result is a new audio.

    The agent's default tool set: audio-to-text tools (currently none are
    registered, but the filter stays as a safety net) and any other
    non-audio-output tool are excluded, since the agent only ever needs to
    hand the model's next turn a new `<audio_id><audio>` tag to keep
    chaining calls on. Single source of truth for this restriction -- both
    `build_system_prompt`'s default catalogue and `agent.ToolCallingAgent`'s
    call-time allowlist derive from this, so a tool can't be advertised
    without also being callable (or vice versa). Restricted to
    `tools_registry.available_tool_names()` (not `tools.TOOL_NAME_TO_CLASS`,
    which lists every implemented `Tool` subclass regardless of whether it's
    part of this project's curated, trained-on toolset) so the live agent
    stays in lockstep with what training data generation actually covers.
    """
    return [
        name for name in tools_registry.available_tool_names()
        if TOOL_NAME_TO_CLASS[name].produces_audio()
    ]


def build_system_prompt(
    tool_names: Optional[List[str]] = None,
    tool_call_format: str = DEFAULT_TOOL_CALL_FORMAT,
) -> str:
    """Render the tool-calling protocol + tool catalogue as a system prompt.

    `tool_names` restricts the catalogue (and, implicitly, what the model is
    told is available) to a subset -- e.g. only what's actually importable in
    the current environment. Defaults to `audio_to_audio_tool_names()` (every
    tool except text-only ones like `asr`), not literally every registered tool.

    `tool_call_format` selects the wire convention (see `tool_call_formats.py`)
    for the `<tool_call>`-shaped section of this prompt -- everything else
    here (the `audio_id`/`output_audio_id` chaining convention) is this
    project's own, not part of any model's official convention, so it stays
    fixed regardless of format.
    """
    if tool_names is None:
        tool_names = audio_to_audio_tool_names()
    classes = [TOOL_NAME_TO_CLASS[name] for name in tool_names]
    tools_preamble = get_tool_call_format(tool_call_format).render_tools_preamble(
        tool_function_schemas(classes)
    )

    protocol_explanation = (
        "You are an audio-editing assistant with access to a fixed set of tools. "
        "You are given one or more input audios (tagged <audio_0>, <audio_1>, ...) and a "
        "task instruction. Infer the chain of tool calls needed to complete the task.\n\n"
        "Every call's `audio_id` argument names which already-seen audio to read -- one of "
        "the input audios, or an `output_audio_id` you declared in an earlier turn. If "
        "omitted, it defaults to whichever audio your previous turn most recently produced "
        "(or the sole input audio, on your first turn). Every audio-producing tool also takes "
        "an `output_audio_id` argument: a fresh id you choose, distinct from every id used so "
        "far, to name that call's result. After each call you will be shown its real output "
        "tagged with the id you gave it, before your next turn. Once the task is fully "
        "complete, stop calling tools and reply normally instead."
    )
    return f"{protocol_explanation}\n\n{tools_preamble}"


def render_user_prompt(instruction: str, audio_ids: List[str]) -> str:
    tags = "\n".join(f"{audio_id}: {audio_tag(audio_id)}" for audio_id in audio_ids)
    return f"{instruction}\n\nInput audio:\n{tags}"


def render_tool_call(
    tool_name: str,
    parameters: Dict[str, Any],
    output_audio_id: Optional[str] = None,
    tool_call_format: str = DEFAULT_TOOL_CALL_FORMAT,
) -> str:
    """Render one assistant turn's tool-call content in `tool_call_format`'s
    wire convention -- the exact shape `parse_turn` expects back. Used by
    ground-truth authoring (`gen_1st_stage_data/build_dataset.py`), not by the
    live agent (there, this text comes from the model itself).

    `output_audio_id` folds into the call's arguments (there's no separate
    channel for it in a function-calling schema) if given -- omit it for
    tools that don't produce audio (e.g. `asr`).
    """
    arguments = dict(parameters)
    if output_audio_id is not None:
        arguments["output_audio_id"] = output_audio_id
    return get_tool_call_format(tool_call_format).render_tool_call(tool_name, arguments)


def render_done(tool_call_format: str = DEFAULT_TOOL_CALL_FORMAT) -> str:
    """Render the closing assistant turn once the task is done."""
    return get_tool_call_format(tool_call_format).render_done()


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

    Deliberately returns this raw, un-wrapped text regardless of
    `tool_call_format`: for training data consumed by ms-swift (the default
    `template_backend='swift'`) and for `interface.engine.SwiftEngine`
    inference, ms-swift's own hermes agent_template already wraps a
    `role: "tool"` message's content in `<tool_response>...</tool_response>`
    at tokenize time -- wrapping it again here would double-wrap. See
    `tool_call_formats.py`'s module docstring for the one path
    (`interface.engine.VLLMEngine`) that does need to wrap it itself.
    """
    if audio_tags:
        return f"Output of step {step_index}: " + " ".join(audio_tags)
    return f"Output of step {step_index} ({tool_name}): {text}"


def parse_turn(raw_text: str, tool_call_format: str = DEFAULT_TOOL_CALL_FORMAT) -> Optional[Dict[str, Any]]:
    """Parse one model turn using `tool_call_format`'s wire convention.

    Returns `{"tool_name": ..., "parameters": {...}}` for a call
    (`output_audio_id`, if any, folded into `parameters`), `{"done": True}`
    if the turn signals completion, or `None` if it's unparseable garbage.
    """
    return get_tool_call_format(tool_call_format).parse_turn(raw_text)
