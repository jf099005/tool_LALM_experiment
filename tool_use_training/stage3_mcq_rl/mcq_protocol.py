"""Stage-3 MCQ-with-tools prompt/answer convention -- the one place it is defined.

`build_mcq_dataset.py` (renders the prompts into parquet), `audio_tool_agent_loop.py`
(the live rollout) and `reward.py` (scores the final turn) all import from here, so the
wording the policy sees and the answer format the reward expects can't drift apart.

Everything about *tool calling* (system-turn tool catalogue, `<tool_call>` wire format,
`Output of step k: <audio_k><audio>` tool turns, harness-assigned audio ids) is reused
verbatim from stage-1 (`tools.tools_registry`, `official_system_prompt`,
`interface.protocol`) -- see AGENT_TOOL_DESIGN.md. Stage-3 only adds the MCQ framing in
the user turn and the `<answer>X</answer>` closing turn.
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from interface import protocol as agent_protocol  # noqa: E402
from official_system_prompt import compose_system_prompt  # noqa: E402

LETTERS = "ABCDEFGH"
INPUT_AUDIO_ID = "audio_0"

_ANSWER_RE = re.compile(r"<answer>\s*\(?([A-H])\)?[.):]?\s*</answer>", re.IGNORECASE)

TASK_INSTRUCTION = (
    "Answer the multiple-choice question about the input audio. You may call the audio "
    "tools first if processing the audio (e.g. isolating a source, removing noise, zooming "
    "into a segment, changing speed/pitch) would help you hear the relevant details; each "
    "tool output is shown to you before your next turn. When you are confident, stop calling "
    "tools and reply with only the letter of the correct option inside <answer></answer>, "
    "e.g. <answer>B</answer>."
)


def build_system_prompt(model_type: str = "qwen3_omni", tool_call_format: str = "qwen") -> str:
    """Stage-1's exact system turn: official persona (empty for Qwen3-Omni) + tool catalogue."""
    from tools import tools_registry  # heavy-ish import, keep it lazy

    return compose_system_prompt(
        tools_block=tools_registry.describe_available_tools(tool_call_format=tool_call_format),
        model_type=model_type,
    )


def shuffle_choices(choices: Sequence[str], answer: str, rng: random.Random) -> Tuple[List[str], str]:
    """Shuffle options and return (shuffled, correct_letter). Removes source position bias."""
    choices = list(choices)
    if answer not in choices:
        raise ValueError(f"answer {answer!r} not among choices {choices!r}")
    rng.shuffle(choices)
    return choices, LETTERS[choices.index(answer)]


def render_user_prompt(question: str, choices: Sequence[str]) -> str:
    """User turn. `<audio_0><audio>` follows stage-1's id-tagging convention; verl's
    RLHFDataset turns the literal `<audio>` into the audio content block."""
    options = "\n".join(f"({LETTERS[i]}) {c}" for i, c in enumerate(choices))
    return (
        f"{TASK_INSTRUCTION}\n\n"
        f"Input audio: {agent_protocol.audio_tag(INPUT_AUDIO_ID)}\n\n"
        f"Question: {question.strip()}\n"
        f"Options:\n{options}"
    )


def extract_answer_letter(text: str) -> Optional[str]:
    """Letter inside the *last* <answer>...</answer>, or None."""
    matches = _ANSWER_RE.findall(text or "")
    return matches[-1].upper() if matches else None
