"""Convert a `gen_1st_stage_data/build_dataset.py`-style raw dataset JSON into
a single-turn ms-swift GRPO dataset.

`gen_1st_stage_data.to_swift_sample` renders the full ground-truth tool-call
chain as an SFT-style multi-turn dialogue (one `assistant` turn per tool call,
each followed by a `tool` turn carrying that step's real output audio). Fed
straight to GRPO -- which, absent a `multi_turn_scheduler`/`gym_env`, only
generates a completion for the trailing message and treats everything before
it as fixed prompt -- that shape hands the model the entire correct answer as
context and only asks it to generate the closing `{"done": true}` sentinel:
there is no tool-call generation happening for `reward.py` to score.

This script instead emits one turn per sample: the question (source/target
audio + tool catalogue, same as the SFT prompt) plus an explicit instruction
to answer with a single `{"tool_calls": [...]}` JSON object, and carries the
full ground truth forward as a `solution` column (`entry["answer"]`, already
`{"tool_calls": [...]}`) for `reward.py`'s `ToolCallAccuracyReward` /
`ToolNameF1Reward` / `ToolUseReward` to score against.

This is a stopgap for single-shot GRPO, not real multi-turn tool-use RL: the
model still never sees a real tool execution result mid-episode. Wiring an
actual `multi_turn_scheduler` around `interface/agent.py`'s real tool
executor is the follow-up.

Usage:
    python build_rl_dataset.py \\
        --input /work/u1501463/stage1_RL/tool_usage_qa_audioset.json \\
        --output /work/u1501463/stage1_RL/train_rl_singleturn.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

OUTPUT_INSTRUCTION = (
    "\n\nRespond with exactly one JSON object of the form "
    '{"tool_calls": [{"tool_name": "...", "parameters": {...}}, ...]} '
    "describing the full ordered chain of tool calls needed to reproduce the "
    "edited clip from the source. Do not execute the calls one at a time and "
    "do not include any text outside the JSON object."
)


def to_swift_rl_sample(
    entry: Dict[str, Any],
    audio_token: str = "<audio>",
    system_prompt: str | None = None,
) -> Dict[str, Any]:
    """Convert one raw dataset entry into a single-turn ms-swift GRPO row.

    Only `audio_0` (source) and `audio_1` (target) are tagged -- unlike
    `to_swift_sample`, there are no intermediate per-step audios since the
    model must emit the whole chain in one shot, without seeing any real
    tool output mid-episode.
    """
    audio_id_map = entry["audio_id_map"]
    audios: List[str] = []

    def tag(audio_id: str) -> str:
        audios.append(audio_id_map[audio_id])
        return f"<{audio_id}>{audio_token}"

    question_text = entry["question"].replace(entry["source_audio"], tag("audio_0"), 1)
    question_text = question_text.replace(entry["target_audio"], tag("audio_1"), 1)
    question_text += OUTPUT_INSTRUCTION

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question_text})

    return {
        "messages": messages,
        "audios": audios,
        "solution": entry["answer"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Raw dataset JSON (list of entries).")
    parser.add_argument("--output", type=Path, required=True, help="Output ms-swift GRPO jsonl path.")
    parser.add_argument("--audio-token", type=str, default="<audio>")
    parser.add_argument("--system-prompt", type=str, default=None)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as handle:
        entries = json.load(handle)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for entry in entries:
            row = to_swift_rl_sample(entry, audio_token=args.audio_token, system_prompt=args.system_prompt)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(entries)} single-turn GRPO rows to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
