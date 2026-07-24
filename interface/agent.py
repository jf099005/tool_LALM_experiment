"""Drives the multi-turn tool-calling loop for a LALM.

The model emits one tool call per turn; we execute it for real against the
audio it referenced, and feed the real result back as the next turn's input
-- the same turn shape `tool_use_training/gen_1st_stage_data/build_dataset.py`
teaches during SFT (`to_swift_sample`) and `testing_tool_use_benchmark/run_eval.py`
drives at eval time. This module generalizes that loop to an arbitrary
natural-language instruction over one or more input audios, rather than only
the fixed source/target reconstruction task the benchmark measures, and has
no notion of a ground-truth chain to score against.

`ToolCallingAgent.run()` also accepts a fully custom `messages`/`audio_id_map`
seed (bypassing the instruction/audio_paths convenience path) and an
arbitrary `max_steps`, so callers with their own prompt framing or stepping
budget -- e.g. `testing_tool_use_benchmark/run_eval.py`'s source/target
reconstruction task -- can drive the same loop/scoring hooks without being
forced into the single-instruction convention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import protocol
from .executor import ToolExecutionError, UnknownToolError, extract_audio_outputs, run_tool_call


@dataclass
class Step:
    index: int
    raw_text: str
    tool_name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    success: bool = False
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    produced_audio_ids: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "raw_text": self.raw_text,
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "success": self.success,
            "error": self.error,
            "produced_audio_ids": self.produced_audio_ids,
        }


@dataclass
class AgentResult:
    instruction: Optional[str]
    audio_id_map: Dict[str, str]
    steps: List[Step]
    stop_reason: str
    final_audio_id: Optional[str]
    final_audio_path: Optional[str]
    messages: List[Dict[str, str]]

    def to_json(self) -> Dict[str, Any]:
        return {
            "instruction": self.instruction,
            "audio_id_map": self.audio_id_map,
            "stop_reason": self.stop_reason,
            "final_audio_id": self.final_audio_id,
            "final_audio_path": self.final_audio_path,
            "steps": [step.to_json() for step in self.steps],
        }


class ToolCallingAgent:
    """Wraps a model backend (`engine.generate_turn(messages, audios) -> str`,
    see `engine.SwiftEngine` / `engine.VLLMEngine`) and drives it through a
    bounded tool-calling loop."""

    def __init__(
        self,
        engine: Any,
        system_prompt: Optional[str] = None,
        tool_names: Optional[List[str]] = None,
    ):
        self.engine = engine
        # Resolved once and enforced at call time (see `run()`) regardless of
        # `system_prompt` -- so a tool that isn't supposed to be available (e.g.
        # asr, or anything else excluded via `tool_names`) stays disabled even if
        # a custom `system_prompt` happens to still mention it.
        self.tool_names = protocol.audio_to_audio_tool_names() if tool_names is None else list(tool_names)
        # An explicit `system_prompt=""` (falsy but not None) means "no system
        # turn at all" -- e.g. to match a fine-tuned checkpoint whose SFT data
        # never had one. Only a bare default (None) triggers auto-building one.
        self.system_prompt = protocol.build_system_prompt(self.tool_names) if system_prompt is None else system_prompt

    def run(
        self,
        *,
        instruction: Optional[str] = None,
        audio_paths: Optional[List[str]] = None,
        work_dir: Path,
        max_steps: int = 8,
        messages: Optional[List[Dict[str, str]]] = None,
        audio_id_map: Optional[Dict[str, str]] = None,
        last_audio_id: Optional[str] = None,
    ) -> AgentResult:
        """Drive one bounded tool-calling run.

        Two ways to seed it:
          - `instruction` + `audio_paths` (the default): audios are
            auto-tagged audio_0, audio_1, ... in order, and the initial
            messages are built from `self.system_prompt` +
            `protocol.render_user_prompt`.
          - `messages` + `audio_id_map`: full control over the starting
            conversation -- pass your own initial message list (any framing,
            e.g. one that shows a target/reference audio alongside the
            inputs) and the audio-id -> path mapping those messages' tags
            refer to. `self.system_prompt` is ignored in this mode; include
            a system turn in `messages` yourself if you want one. Pass
            `last_audio_id` to say which id a tool call's `audio_id` should
            default to when omitted (defaults to the last entry in
            `audio_id_map`).

        `max_steps` is always caller-controlled -- e.g. pass a budget derived
        from a ground-truth chain length rather than a fixed constant.
        """
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        if messages is not None or audio_id_map is not None:
            if messages is None or audio_id_map is None:
                raise ValueError("`messages` and `audio_id_map` must be provided together")
            messages = list(messages)
            audio_id_map = dict(audio_id_map)
            if last_audio_id is None and audio_id_map:
                last_audio_id = next(reversed(audio_id_map))
        else:
            if instruction is None:
                raise ValueError("`instruction` is required unless `messages`/`audio_id_map` are given")
            audio_paths = audio_paths or []
            audio_id_map = {f"audio_{i}": str(p) for i, p in enumerate(audio_paths)}
            last_audio_id = f"audio_{len(audio_paths) - 1}" if audio_paths else None

            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({
                "role": "user",
                "content": protocol.render_user_prompt(instruction, list(audio_id_map.keys())),
            })

        audios: List[str] = list(audio_id_map.values())
        next_fresh_id = len(audio_id_map)

        steps: List[Step] = []
        seen_calls = set()
        stop_reason = "max_steps_reached"

        for step_index in range(1, max_steps + 1):
            raw_text = self.engine.generate_turn(messages, audios)
            messages.append({"role": "assistant", "content": raw_text})
            step = Step(index=step_index, raw_text=raw_text)

            call = protocol.parse_turn(raw_text)
            if call is None:
                step.error = "unparseable_output"
                steps.append(step)
                stop_reason = "unparseable_output"
                break
            if call.get("done"):
                steps.append(step)
                stop_reason = "model_signaled_done"
                break

            tool_name = call.get("tool_name")
            parameters = call.get("parameters") if isinstance(call.get("parameters"), dict) else {}
            step.tool_name = tool_name
            step.parameters = parameters

            if not tool_name:
                step.error = "missing_tool_name"
                steps.append(step)
                stop_reason = "unparseable_output"
                break

            if tool_name not in self.tool_names:
                step.error = f"tool '{tool_name}' is not available"
                steps.append(step)
                stop_reason = "disallowed_tool"
                break

            call_signature = (tool_name, json.dumps(parameters, sort_keys=True))
            if call_signature in seen_calls:
                step.error = "repeated_call"
                steps.append(step)
                stop_reason = "repeated_call"
                break
            seen_calls.add(call_signature)

            input_audio_id = parameters.get("audio_id") or last_audio_id
            input_audio_path = audio_id_map.get(input_audio_id) if input_audio_id else None
            if input_audio_path is None:
                step.error = f"unknown audio_id '{input_audio_id}'"
                steps.append(step)
                stop_reason = "unknown_audio_id"
                break

            try:
                result = run_tool_call(tool_name, parameters, input_audio_path, work_dir, step_index)
            except (UnknownToolError, ToolExecutionError) as exc:
                step.error = str(exc)
                steps.append(step)
                stop_reason = "tool_execution_failed"
                break

            step.success = True
            step.result = result

            if call.get("output_audio_id"):
                output_audio_id = call["output_audio_id"]
            else:
                while f"audio_{next_fresh_id}" in audio_id_map:
                    next_fresh_id += 1
                output_audio_id = f"audio_{next_fresh_id}"
                next_fresh_id += 1
            audio_outputs = extract_audio_outputs(result)

            if not audio_outputs:
                text = result.get("transcript") or result.get("message") or json.dumps(result)
                tool_message = protocol.render_tool_result_message(step_index, tool_name, [], text=text)
            else:
                for stem, path in audio_outputs.items():
                    audio_id = output_audio_id if stem in ("", "target") else f"{output_audio_id}_{stem}"
                    audio_id_map[audio_id] = str(path)
                    audios.append(str(path))
                    step.produced_audio_ids.append(audio_id)
                last_audio_id = step.produced_audio_ids[0]
                tags = [protocol.audio_tag(audio_id) for audio_id in step.produced_audio_ids]
                tool_message = protocol.render_tool_result_message(step_index, tool_name, tags)

            steps.append(step)
            messages.append({"role": "tool", "content": tool_message})
        else:
            stop_reason = "max_steps_reached"

        final_audio_path = audio_id_map.get(last_audio_id) if last_audio_id else None

        return AgentResult(
            instruction=instruction,
            audio_id_map=audio_id_map,
            steps=steps,
            stop_reason=stop_reason,
            final_audio_id=last_audio_id,
            final_audio_path=final_audio_path,
            messages=messages,
        )
