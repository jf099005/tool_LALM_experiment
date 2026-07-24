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
audio, same framing as the SFT prompt) plus an explicit instruction to answer
with a single `{"tool_calls": [...]}` JSON object, and carries the full
ground truth forward as a `solution` column (`entry["answer"]`, already
`{"tool_calls": [...]}`) for `reward.py`'s `ToolCallAccuracyReward` /
`ToolNameF1Reward` / `ToolUseReward` to score against. The tool catalogue
itself lives in the system turn (see `official_system_prompt.py`), same as
`gen_1st_stage_data.to_swift_sample` -- pass the same `--system-prompt-model-dir`
used for stage-1 so the policy sees the identical system prompt across both
stages.

This is a stopgap for single-shot GRPO, not real multi-turn tool-use RL: the
model still never sees a real tool execution result mid-episode. Wiring an
actual `multi_turn_scheduler` around `interface/agent.py`'s real tool
executor is the follow-up.

Usage:
    python build_rl_dataset.py \\
        --input /work/u1501463/stage1_RL/tool_usage_qa_audioset.json \\
        --output /work/u1501463/stage1_RL/train_rl_singleturn.jsonl \\
        --system-prompt-model-dir /work/u1501463/model_qwen_25/ \\
        --system-prompt-model-type qwen2_5_omni
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from official_system_prompt import compose_system_prompt  # noqa: E402

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
    base_system_prompt: str | None = None,
    system_prompt_model_dir: str | None = None,
    system_prompt_model_type: str | None = None,
) -> Dict[str, Any]:
    """Convert one raw dataset entry into a single-turn ms-swift GRPO row.

    Only `audio_0` (source) and `audio_1` (target) are tagged -- unlike
    `to_swift_sample`, there are no intermediate per-step audios since the
    model must emit the whole chain in one shot, without seeing any real
    tool output mid-episode.

    Like `to_swift_sample`, the tool catalogue lives in the system turn (the
    target model's own official default + the tool catalogue -- see
    `official_system_prompt.compose_system_prompt`) rather than inline in the
    user question, so the same system prompt convention carries over from
    stage-1 SFT into stage-2 GRPO.
    """
    if "tools_block" not in entry:
        raise KeyError(
            "entry has no 'tools_block' -- regenerate the raw dataset JSON with the current "
            "gen_1st_stage_data/build_dataset.py (older raw JSONs predate moving the tool "
            "catalogue into the system prompt)."
        )

    audio_id_map = entry["audio_id_map"]
    audios: List[str] = []

    def tag(audio_id: str) -> str:
        audios.append(audio_id_map[audio_id])
        return f"<{audio_id}>{audio_token}"

    question_text = entry["question"].replace(entry["source_audio"], tag("audio_0"), 1)
    question_text = question_text.replace(entry["target_audio"], tag("audio_1"), 1)
    question_text += OUTPUT_INSTRUCTION

    system_prompt = compose_system_prompt(
        tools_block=entry["tools_block"],
        base_system_prompt=base_system_prompt,
        model_dir=system_prompt_model_dir,
        model_type=system_prompt_model_type,
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question_text}]

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
    parser.add_argument(
        "--base-system-prompt",
        type=str,
        default=None,
        help=(
            "Explicit override for the system prompt's base greeting (the part before the "
            "tool catalogue). If omitted, it's auto-detected from --system-prompt-model-dir "
            "(or --system-prompt-model-type as a fallback) -- see official_system_prompt.py. "
            "Should normally match whatever stage-1 SFT used, since this is a continuation of "
            "the same policy."
        ),
    )
    parser.add_argument(
        "--system-prompt-model-dir",
        type=str,
        default=None,
        help=(
            "Path to the target model's own directory, used to auto-detect its official "
            "default system message. Matches the flag of the same name in "
            "gen_1st_stage_data/build_dataset.py -- pass the same model dir here as stage-1 "
            "used, then swap both together when training a different model."
        ),
    )
    parser.add_argument(
        "--system-prompt-model-type",
        type=str,
        default="qwen2_5_omni",
        help=(
            "Fallback key into official_system_prompt.KNOWN_DEFAULT_SYSTEM_PROMPTS, used only "
            "if --system-prompt-model-dir is omitted or its chat_template can't be parsed."
        ),
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as handle:
        entries = json.load(handle)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for entry in entries:
            row = to_swift_rl_sample(
                entry,
                audio_token=args.audio_token,
                base_system_prompt=args.base_system_prompt,
                system_prompt_model_dir=args.system_prompt_model_dir,
                system_prompt_model_type=args.system_prompt_model_type,
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(entries)} single-turn GRPO rows to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
