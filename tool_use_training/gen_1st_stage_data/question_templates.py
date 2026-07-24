"""Linguistic variations for the A -> B tool-chain inference question.

Each template takes the source audio reference and the target audio
reference and renders a full question. A new template is picked at random
per sample so the dataset doesn't collapse onto one repeated phrasing.

The tool catalogue is NOT rendered here -- it lives in the system prompt
instead (see `official_system_prompt.py`), not repeated inline in every user
turn.
"""

from __future__ import annotations

import random
from typing import List

QUESTION_TEMPLATES: List[str] = [
    "Transform the source audio into the target audio using the available tools.\n\n"
    "Source audio: {source}\nTarget audio: {target}",

    "Determine which audio editing operations should be applied so that the source "
    "audio becomes the target audio.\n\nSource audio (A): {source}\nTarget audio (B): {target}",

    "Identify the correct sequence of tool invocations required to reproduce the target "
    "audio from the source audio.\n\nSource: {source}\nTarget: {target}",

    "Given the original and transformed audio below, infer the editing pipeline that "
    "connects them.\n\nOriginal audio: {source}\nTransformed audio: {target}",

    "Find a sequence of tool calls that maps the source recording to the target "
    "recording.\n\nSource recording: {source}\nTarget recording: {target}",

    "Listening to audio A and audio B, work out which operations turn A into B and in "
    "what order.\n\nAudio A: {source}\nAudio B: {target}",

    "You are given two audio clips, a source and its edited version. Reconstruct the "
    "tool-call chain that produces the edited clip from the source.\n\n"
    "Source clip: {source}\nEdited clip: {target}",

    "Recover the editing pipeline: starting from the source audio, which tool calls, "
    "applied in which order, yield the target audio?\n\nSource: {source}\nTarget: {target}",

    "Audio A must become audio B. Using only the tools listed in the system prompt, output "
    "the call sequence that performs this transformation.\n\nAudio A: {source}\nAudio B: {target}",

    "Compare the before/after audio pair and infer the underlying processing chain.\n\n"
    "Before: {source}\nAfter: {target}",
]


def render_question(source: str, target: str, rng: random.Random) -> str:
    template = rng.choice(QUESTION_TEMPLATES)
    return template.format(source=source, target=target)
