"""verl agent loop: multi-turn audio-tool use for MCQ answering (stage-3 RL rollout).

Why a custom loop instead of verl's `ToolAgentLoop`:
  * our tools return *audio*, and `ToolAgentLoop` only knows how to feed images back;
  * the turn protocol must be stage-1's exactly (see AGENT_TOOL_DESIGN.md): the model's
    `<tool_call>` is parsed with `interface.protocol.parse_turn`, executed with
    `interface.executor.run_tool_call` (the project's single execution entry point),
    output ids are assigned by the harness (`audio_1`, `audio_2`, ... like
    `interface/agent.py`), and the tool turn reads `Output of step k: <audio_k><audio>`
    (`interface.protocol.render_tool_result_message`) with the real output audio in place
    of `<audio>`.

Per episode: system(tool catalogue) + user(MCQ + audio_0) -> [assistant tool call -> tool
turn with new audio]* -> assistant `<answer>X</answer>`. Tool-turn tokens are masked out of
the loss (response_mask=0); every new audio is appended to the episode's audio list in the
order its placeholder appears, so vLLM rollout and the FSDP actor's recompute see the same
multimodal sequence.

Episode statistics go to `extra_fields`, which verl forwards to `reward.compute_score`
via `extra_info` (and therefore into the logged reward metrics).

Registered through `agent_loop.yaml` (`actor_rollout_ref.rollout.agent.agent_loop_config_path`).
Env knobs:
  STAGE3_TOOL_WORKDIR        scratch dir for tool outputs (default /work/u1501463/stage3_mcq_rl/tool_work/default)
  STAGE3_KEEP_TOOL_OUTPUTS   1 = don't delete an episode's tool outputs (debugging)
  STAGE3_TOOL_TIMEOUT        seconds per tool call (default 600)
  STAGE3_MAX_TOOL_AUDIO_SEC  longer tool outputs are rejected with an error turn (default 60)
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.profiler import simple_timer
from verl.utils.tokenizer.continuous_token_wiring import (
    ContinuousTokenModelFamily,
    create_continuous_token_builder,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from interface import protocol as agent_protocol  # noqa: E402
from interface.executor import ToolExecutionError, UnknownToolError, extract_audio_outputs, run_tool_call  # noqa: E402
from tools import heavy_tool_pool, tools_registry  # noqa: E402

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

TOOL_WORKDIR = Path(os.environ.get("STAGE3_TOOL_WORKDIR", "/work/u1501463/stage3_mcq_rl/tool_work/default"))
KEEP_TOOL_OUTPUTS = os.environ.get("STAGE3_KEEP_TOOL_OUTPUTS", "0") == "1"
TOOL_TIMEOUT = float(os.environ.get("STAGE3_TOOL_TIMEOUT", "600"))
MAX_TOOL_AUDIO_SEC = float(os.environ.get("STAGE3_MAX_TOOL_AUDIO_SEC", "60"))
SAMPLING_RATE = 16000  # Qwen3-Omni feature extractor rate; QwenOmniRLHFDataset loads at this rate

_AUDIO_SPLIT_RE = re.compile(f"({re.escape(agent_protocol.AUDIO_TOKEN)})")


def _require_heavy_tool_pool() -> None:
    """Heavy tools without a running pool would reload SAM-Audio/AudioSR per call. Fail loudly."""
    if tools_registry.heavy_tool_env_pythons() and not os.environ.get(heavy_tool_pool.POOL_DIR_ENV):
        raise RuntimeError(
            f"{heavy_tool_pool.POOL_DIR_ENV} is unset but heavy tools are registered "
            f"({tools_registry.heavy_tool_env_pythons()}). Launch training through "
            "stage3_mcq_rl/launch_with_tool_pool.py so the persistent tool servers are up."
        )


class AudioToolMCQAgentLoop(AgentLoopBase):
    def __init__(self, *args, tools: Any = None, **kwargs):
        # `tools` (verl's own tool objects) is unused: our toolset is tools_registry's.
        super().__init__(*args, **kwargs)
        _require_heavy_tool_pool()
        # qwen3_omni_moe has no registered Continuous Token family and falls back to the
        # generic VL builder; use the Qwen VL one so the `<|im_end|>\n` turn boundary after a
        # generated tool call matches the ChatML text stage-1 SFT was trained on.
        self.continuous_token_builder = create_continuous_token_builder(
            self.tokenizer,
            model_family=ContinuousTokenModelFamily.QWEN_VL,
            chat_template_kwargs=self.apply_chat_template_kwargs,
            mm_processor_kwargs=self.mm_processor_kwargs,
            processor=self.processor,
        )
        multi_turn = self.rollout_config.multi_turn
        self.max_assistant_turns = multi_turn.max_assistant_turns or 4
        self.response_length = self.rollout_config.response_length
        self.tool_names = set(agent_protocol.audio_to_audio_tool_names())

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _input_audio_path(messages: List[Dict[str, Any]]) -> Optional[str]:
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "audio" and isinstance(block.get("audio"), str):
                        return block["audio"]
        return None

    async def _load_audios(self, paths: List[str]) -> List[Any]:
        """Load through the dataset class (16 kHz + hop padding) exactly like the input audio."""
        probe = [{"role": "user", "content": [{"type": "audio", "audio": p} for p in paths]}]
        mm = await self.process_multi_modal_info(probe)
        return list(mm.get("audios") or [])

    def _execute(self, tool_name: str, parameters: Dict[str, Any], input_path: str, work_dir: Path, step: int):
        # Dataset audio may be mp3/flac (sometimes under a .wav name); the wave-based tools need
        # real PCM WAV. Tool outputs are already WAV, so this only converts the original input.
        wav_path = tools_registry.ensure_wav(Path(input_path), work_dir / "_inputs")
        return run_tool_call(tool_name, parameters, str(wav_path), work_dir, step)

    # ------------------------------------------------------------------ main loop

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages: List[Dict[str, Any]] = list(kwargs["raw_prompt"])
        request_id = uuid4().hex
        work_dir = TOOL_WORKDIR / request_id
        metrics: Dict[str, Any] = {}

        mm = await self.process_multi_modal_info(messages)
        audios: List[Any] = list(mm.get("audios") or [])
        self._assert_mm_supported(bool(audios))
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        input_path = self._input_audio_path(messages)
        # Continuous Token re-renders the full history through the processor (e.g. for the
        # generation-prompt delta), and the audio feature extractor needs arrays, not paths:
        # swap each audio block for the array loaded above (same order as `audios`).
        audio_iter = iter(audios)
        messages = [
            {**m, "content": [
                {**b, "audio": next(audio_iter)} if isinstance(b, dict) and b.get("type") == "audio" else b
                for b in m["content"]
            ]} if isinstance(m.get("content"), list) else m
            for m in messages
        ]
        audio_id_map: Dict[str, str] = {"audio_0": input_path} if input_path else {}
        last_audio_id: Optional[str] = "audio_0" if input_path else None
        next_fresh_id = len(audio_id_map)
        seen_calls: set = set()

        prompt_ids = await self.ct_build_initial_tokens(messages, audios=audios or None)
        response_mask: List[int] = []
        response_logprobs: List[float] = []
        extra_fields: Dict[str, Any] = {}
        routed_experts = None

        stats = {"num_tool_calls": 0, "num_tool_success": 0, "num_tool_errors": 0, "num_parse_errors": 0}
        tools_used: List[str] = []
        stop_reason = "max_turns"
        final_text = ""
        assistant_turns = user_turns = 0

        try:
            while True:
                with simple_timer("generate_sequences", metrics):
                    output = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                        audio_data=audios or None,
                        mm_processor_kwargs=mm_processor_kwargs,
                    )
                metrics["num_preempted"] = metrics.get("num_preempted", 0) + (output.num_preempted or 0)
                if not extra_fields:
                    extra_fields.update(output.extra_fields or {})
                if output.routed_experts is not None:
                    routed_experts = output.routed_experts
                assistant_turns += 1

                merge, response_mask, merged_logprobs = await self.ct_merge_assistant_token(
                    prompt_ids,
                    output.token_ids,
                    response_mask,
                    response_logprobs if (response_logprobs or output.log_probs) else None,
                    assistant_logprobs=output.log_probs or None,
                )
                prompt_ids = merge.token_ids
                if merged_logprobs is not None:
                    response_logprobs = merged_logprobs

                text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
                final_text = text
                messages.append({"role": "assistant", "content": text})

                if len(response_mask) >= self.response_length:
                    stop_reason = "response_length"
                    break
                call = agent_protocol.parse_turn(text)
                if call is None:
                    stats["num_parse_errors"] += 1
                    stop_reason = "unparseable_tool_call"
                    break
                if call.get("done"):
                    stop_reason = "answered"
                    break
                if assistant_turns >= self.max_assistant_turns:
                    stop_reason = "max_turns"
                    break

                # ---- execute one tool call -------------------------------------------
                step = user_turns + 1
                stats["num_tool_calls"] += 1
                tool_name = call.get("tool_name")
                parameters = call.get("parameters") or {}
                tools_used.append(str(tool_name))
                error: Optional[str] = None
                new_paths: List[str] = []
                produced_ids: List[str] = []

                signature = (tool_name, json.dumps(parameters, sort_keys=True))
                input_id = parameters.get("audio_id") or last_audio_id
                if tool_name not in self.tool_names:
                    error = f"tool '{tool_name}' is not available"
                elif signature in seen_calls:
                    error = "this exact call was already made; its output is above"
                elif input_id not in audio_id_map:
                    error = f"unknown audio_id '{input_id}'"
                else:
                    seen_calls.add(signature)
                    try:
                        with simple_timer("tool_calls", metrics):
                            result = await asyncio.wait_for(
                                self.loop.run_in_executor(
                                    None,
                                    functools.partial(
                                        self._execute, tool_name, parameters, audio_id_map[input_id], work_dir, step
                                    ),
                                ),
                                timeout=TOOL_TIMEOUT,
                            )
                        outputs = extract_audio_outputs(result)
                        if not outputs:
                            error = "the tool produced no audio"
                        else:
                            while f"audio_{next_fresh_id}" in audio_id_map:
                                next_fresh_id += 1
                            base_id = f"audio_{next_fresh_id}"
                            next_fresh_id += 1
                            for stem, path in outputs.items():
                                produced_ids.append(base_id if stem in ("", "target") else f"{base_id}_{stem}")
                                new_paths.append(str(path))
                    except (UnknownToolError, ToolExecutionError) as exc:
                        error = str(exc)
                    except asyncio.TimeoutError:
                        error = f"{tool_name} timed out after {TOOL_TIMEOUT:.0f}s"

                new_audios: List[Any] = []
                if error is None:
                    new_audios = await self._load_audios(new_paths)
                    too_long = [len(a) / SAMPLING_RATE for a in new_audios if len(a) / SAMPLING_RATE > MAX_TOOL_AUDIO_SEC]
                    if too_long:
                        error = f"output audio too long ({max(too_long):.1f}s > {MAX_TOOL_AUDIO_SEC:.0f}s)"
                        new_audios = []

                if error is None:
                    stats["num_tool_success"] += 1
                    for audio_id, path in zip(produced_ids, new_paths):
                        audio_id_map[audio_id] = path
                    last_audio_id = produced_ids[0]
                    tags = [agent_protocol.audio_tag(a) for a in produced_ids]
                    tool_text = agent_protocol.render_tool_result_message(step, tool_name, tags)
                    content: List[Dict[str, Any]] = []
                    audio_iter = iter(new_audios)
                    for segment in filter(None, _AUDIO_SPLIT_RE.split(tool_text)):
                        if segment == agent_protocol.AUDIO_TOKEN:
                            content.append({"type": "audio", "audio": next(audio_iter)})
                        else:
                            content.append({"type": "text", "text": segment})
                    tool_message: Dict[str, Any] = {"role": "tool", "content": content}
                else:
                    stats["num_tool_errors"] += 1
                    tool_message = {"role": "tool", "content": f"Error in step {step} ({tool_name}): {error}"}

                previous_messages = list(messages)
                messages.append(tool_message)
                merge, new_mask, new_logprobs = await self.ct_merge_non_assistant_msg(
                    previous_messages,
                    messages,
                    prompt_ids,
                    response_mask,
                    response_logprobs or None,
                )
                if len(new_mask) >= self.response_length:
                    # Never commit a partially-fitting tool turn: truncating inside an audio
                    # placeholder run would misalign audio features.
                    stop_reason = "response_length"
                    break
                prompt_ids, response_mask = merge.token_ids, new_mask
                if response_logprobs:
                    response_logprobs = new_logprobs or []
                audios.extend(new_audios)
                user_turns += 1
        finally:
            if not KEEP_TOOL_OUTPUTS:
                shutil.rmtree(work_dir, ignore_errors=True)

        response_ids = prompt_ids[-len(response_mask):] if response_mask else []
        prompt_only = prompt_ids[: len(prompt_ids) - len(response_mask)]
        extra_fields.update({
            **stats,
            "tools_used": ",".join(tools_used),
            "stop_reason": stop_reason,
            "final_text": final_text,
            "assistant_turns": assistant_turns,
        })
        return AgentLoopOutput(
            prompt_ids=prompt_only,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data={"audios": audios} if audios else {},
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=assistant_turns + user_turns + 1,
            metrics=metrics,
            routed_experts=(
                routed_experts[: len(prompt_only) + self.response_length] if routed_experts is not None else None
            ),
            extra_fields=extra_fields,
        )
