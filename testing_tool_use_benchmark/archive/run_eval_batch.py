"""Batched version of `run_eval.py`.

`run_eval.py` drives the multi-turn tool-calling loop one sample at a time --
every `engine.generate_turn` call decodes a single conversation, which on a
batch-capable accelerator like an H100 leaves most of the SMs idle between
tiny generate() calls. Samples are independent and only loosely
synchronized (each advances one turn per round, but finishes after a
different number of turns), so the natural way to raise utilization is to
batch across *samples* at each turn: keep a fixed-size pool of in-flight
samples, generate one turn for every sample in the pool in a single
`engine.generate_turns` call, execute whichever tool calls came back
(cheap/CPU, not batched), retire whichever samples just finished, and
immediately backfill the pool from the queue so the batch never idles down
to zero before the tail of the benchmark starts. This exactly reproduces
`run_sample`'s per-sample state machine and metrics (see `step_sample`/
`finalize_state` below, which are `run_sample`'s body split into an
init/step/finalize triplet operating on a plain dict instead of locals) --
only the scheduling around it changes.

Must run under the same env as run_eval.py. Usage:
    ~/miniconda3/envs/swift_qwen3/bin/python run_eval_batch.py \
        --benchmark-file benchmark.json \
        --model /work/u1501463/Qwen3-Omni-30B-A3B-Instruct/ \
        --adapter-dir /work/u1501463/stage1_training/v0-.../checkpoint-100/ \
        --batch-size 16 \
        --output-file results_qwen3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_metrics import compare_audio  # noqa: E402
from interface.executor import ToolExecutionError, UnknownToolError, extract_audio_outputs, run_tool_call  # noqa: E402
from interface.protocol import (  # noqa: E402
    AUDIO_TOKEN,
    audio_to_audio_tool_names,
    diagnose_tool_call_tags,
    parse_turn,
    render_tool_result_message,
)

# Reuse run_eval.py's pure helpers verbatim so the two scripts never drift on
# what a "correct" answer/system prompt/aggregate metric looks like -- only
# the scheduling loop below is new.
from run_eval import (  # noqa: E402
    aggregate,
    render_initial_prompt,
    resolve_system_prompt,
)


def make_state(
    entry: Dict[str, Any],
    orig_index: int,
    work_dir: Path,
    max_extra_steps: int,
    hard_step_cap: int,
    system_prompt: Optional[str],
) -> Dict[str, Any]:
    """One sample's mutable evaluation state -- the batched equivalent of
    `run_sample`'s local variables, bundled into a dict so a pool slot can
    hold (and swap out) one of these at a time."""
    sample_id = entry["id"]
    sample_dir = work_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    audio_a = entry["source_audio"]
    audio_b = entry["target_audio"]
    gt_calls = entry["answer"]["tool_calls"]
    max_steps = min(len(gt_calls) + max_extra_steps, hard_step_cap)

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": render_initial_prompt(entry)})

    return {
        "entry": entry,
        "orig_index": orig_index,
        "sample_id": sample_id,
        "sample_dir": sample_dir,
        "audio_a": audio_a,
        "audio_b": audio_b,
        "gt_calls": gt_calls,
        "max_steps": max_steps,
        "system_prompt": system_prompt,
        "messages": messages,
        "audios": [audio_a, audio_b],
        # Mirrors `interface.agent.ToolCallingAgent.run`'s audio_id bookkeeping --
        # see `run_sample` in run_eval.py for the full rationale.
        "audio_id_map": {"audio_0": audio_a, "audio_1": audio_b},
        "last_audio_id": "audio_0",
        "current_audio": Path(audio_a),
        "predicted_steps": [],
        "seen_calls": set(),
        "stop_reason": "max_steps_reached",
        "accumulated_text": "",
        "decode_warnings": [],
        "num_turns_generated": 0,
        "tag_malformed_valid_content_steps": [],
        "step_idx": 0,
        "finished": max_steps <= 0,
    }


def step_sample(
    state: Dict[str, Any],
    raw_text: str,
    step_decode_warnings: List[Dict[str, int]],
    tool_call_format: str,
    known_tool_names: set,
) -> None:
    """Advance `state` by exactly one generated turn, in place. Identical
    step-by-step logic to the body of `run_sample`'s for-loop in
    run_eval.py; sets `state["finished"] = True` (with `stop_reason`) on
    whatever condition would have `break`-ed that loop, including hitting
    `max_steps` after a successful step (that loop's `for...else`)."""
    step_idx = state["step_idx"] + 1
    state["step_idx"] = step_idx
    state["num_turns_generated"] += 1
    for warning in step_decode_warnings:
        state["decode_warnings"].append({"step": step_idx, **warning})
    state["accumulated_text"] += raw_text + "\n"
    state["messages"].append({"role": "assistant", "content": raw_text})

    call = parse_turn(raw_text, tool_call_format=tool_call_format)
    if call is None or call.get("done"):
        diag = diagnose_tool_call_tags(raw_text, tool_call_format=tool_call_format)
        content = diag["content"]
        if not diag["tag_ok"] and content and content["tool_name"] in known_tool_names:
            state["tag_malformed_valid_content_steps"].append({
                "step": step_idx,
                "tool_name": content["tool_name"],
                "parameters": content["parameters"],
            })
    if call is None:
        state["stop_reason"] = "unparseable_output"
        state["finished"] = True
        return
    if call.get("done"):
        state["stop_reason"] = "model_signaled_done"
        state["finished"] = True
        return

    tool_name = call.get("tool_name")
    if not tool_name:
        state["stop_reason"] = "unparseable_output"
        state["finished"] = True
        return
    if tool_name not in known_tool_names:
        state["stop_reason"] = "disallowed_tool"
        state["finished"] = True
        return
    parameters = call.get("parameters", {}) if isinstance(call.get("parameters"), dict) else {}
    call_signature = (tool_name, json.dumps(parameters, sort_keys=True))
    if call_signature in state["seen_calls"]:
        state["stop_reason"] = "repeated_call"
        state["finished"] = True
        return
    state["seen_calls"].add(call_signature)

    input_audio_id = parameters.get("audio_id") or state["last_audio_id"]
    input_audio_path = state["audio_id_map"].get(input_audio_id) if input_audio_id else None
    if input_audio_path is None:
        state["predicted_steps"].append({
            "step": step_idx,
            "tool_name": tool_name,
            "parameters": parameters,
            "success": False,
            "error": f"unknown audio_id '{input_audio_id}'",
        })
        state["stop_reason"] = "unknown_audio_id"
        state["finished"] = True
        return

    exec_parameters = dict(parameters)
    step_record: Dict[str, Any] = {"step": step_idx, "tool_name": tool_name, "parameters": parameters}
    try:
        result = run_tool_call(tool_name, exec_parameters, Path(input_audio_path), state["sample_dir"], step_idx)
    except (UnknownToolError, ToolExecutionError) as exc:
        step_record["success"] = False
        step_record["error"] = str(exc)
        state["predicted_steps"].append(step_record)
        state["stop_reason"] = "tool_execution_failed"
        state["finished"] = True
        return

    step_record["success"] = True
    state["predicted_steps"].append(step_record)

    output_audio_id = parameters.get("output_audio_id") or f"audio_{step_idx + 1}"
    audio_outputs = extract_audio_outputs(result)
    if not audio_outputs:
        text = result.get("transcript") or result.get("message") or json.dumps(result)
        tool_message = render_tool_result_message(step_idx, tool_name, [], text=text)
    else:
        for stem, path in audio_outputs.items():
            audio_id = output_audio_id if stem in ("", "target") else f"{output_audio_id}_{stem}"
            state["audio_id_map"][audio_id] = str(path)
            state["audios"].append(str(path))
        state["current_audio"] = Path(
            audio_outputs.get("target") or audio_outputs.get("") or next(iter(audio_outputs.values()))
        )
        state["last_audio_id"] = output_audio_id
        tool_message = render_tool_result_message(step_idx, tool_name, [f"<{output_audio_id}>{AUDIO_TOKEN}"])
    state["messages"].append({"role": "tool", "content": tool_message})

    if step_idx >= state["max_steps"]:
        state["stop_reason"] = "max_steps_reached"
        state["finished"] = True


def finalize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the same per-sample metrics block `run_sample` returns, from
    a finished `state` dict. Kept byte-for-byte identical to run_eval.py's
    version so results/results_summary.json stay comparable across scripts."""
    from collections import Counter

    gt_calls = state["gt_calls"]
    predicted_steps = state["predicted_steps"]
    audio_a, audio_b = state["audio_a"], state["audio_b"]
    current_audio = state["current_audio"]

    gt_tool_names = [c["tool_name"] for c in gt_calls]
    pred_attempted_names = [s["tool_name"] for s in predicted_steps]
    pred_successful_names = [s["tool_name"] for s in predicted_steps if s["success"]]

    num_predicted = len(predicted_steps)
    num_successful = sum(1 for s in predicted_steps if s["success"])

    gt_counter, pred_counter = Counter(gt_tool_names), Counter(pred_successful_names)
    overlap = sum((gt_counter & pred_counter).values())
    precision = overlap / num_successful if num_successful else 0.0
    recall = overlap / len(gt_tool_names) if gt_tool_names else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    gt_tool_set = set(gt_tool_names)
    pred_successful_set = set(pred_successful_names)
    gt_tools_used = sorted(gt_tool_set & pred_successful_set)
    gt_tools_missed = sorted(gt_tool_set - pred_successful_set)
    gt_tool_coverage = (len(gt_tools_used) / len(gt_tool_set)) if gt_tool_set else 1.0

    audio_metrics = compare_audio(str(current_audio), audio_b)
    baseline_metrics = compare_audio(audio_a, audio_b)

    return {
        "id": state["sample_id"],
        "num_gt_steps": len(gt_calls),
        "gt_tool_sequence": gt_tool_names,
        "prompt": state["messages"][0]["content"] if state["system_prompt"] else None,
        "predicted_steps": predicted_steps,
        "predicted_attempted_sequence": pred_attempted_names,
        "predicted_successful_sequence": pred_successful_names,
        "stop_reason": state["stop_reason"],
        "num_predicted_calls": num_predicted,
        "num_successful_calls": num_successful,
        "tool_call_success_rate": (num_successful / num_predicted) if num_predicted else 0.0,
        "exact_sequence_match": pred_successful_names == gt_tool_names,
        "tool_name_precision": precision,
        "tool_name_recall": recall,
        "tool_name_f1": f1,
        "gt_tools_used": gt_tools_used,
        "gt_tools_missed": gt_tools_missed,
        "gt_tool_coverage": gt_tool_coverage,
        "used_any_gt_tool": bool(gt_tools_used) if gt_tool_set else None,
        "used_all_gt_tools": (len(gt_tools_missed) == 0) if gt_tool_set else None,
        "final_audio_path": str(current_audio),
        "audio_metrics": audio_metrics,
        "baseline_metrics": baseline_metrics,
        "LLM_output": state["accumulated_text"],
        "decode_warnings": state["decode_warnings"],
        "num_turns_generated": state["num_turns_generated"],
        "num_tag_malformed_valid_content": len(state["tag_malformed_valid_content_steps"]),
        "tag_malformed_valid_content_steps": state["tag_malformed_valid_content_steps"],
    }


def run_batch(
    engine,
    benchmark: List[Dict[str, Any]],
    work_dir: Path,
    max_extra_steps: int,
    hard_step_cap: int,
    args: argparse.Namespace,
    batch_size: int,
    output_file: Path,
) -> List[Dict[str, Any]]:
    known_tool_names = set(audio_to_audio_tool_names())
    total = len(benchmark)
    queue = deque(enumerate(benchmark))
    slots: List[Optional[Dict[str, Any]]] = [None] * min(batch_size, total)
    results: List[Optional[Dict[str, Any]]] = [None] * total
    num_completed = 0

    def fill_slot(i: int) -> None:
        while queue:
            idx, entry = queue.popleft()
            system_prompt = resolve_system_prompt(entry, args)
            state = make_state(entry, idx, work_dir, max_extra_steps, hard_step_cap, system_prompt)
            if state["finished"]:
                # max_steps <= 0 -- degenerate entry, never generates a turn
                # (matches run_sample's `range(1, max_steps + 1)` being empty).
                results[idx] = finalize_state(state)
                nonlocal num_completed
                num_completed += 1
                continue
            slots[i] = state
            return
        slots[i] = None

    for i in range(len(slots)):
        fill_slot(i)

    round_num = 0
    while any(s is not None for s in slots):
        round_num += 1
        active_positions = [i for i, s in enumerate(slots) if s is not None]
        batch_inputs = [{"messages": slots[i]["messages"], "audios": slots[i]["audios"]} for i in active_positions]
        raw_texts = engine.generate_turns(batch_inputs)
        decode_warnings_batch = getattr(engine, "last_decode_warnings_batch", None) or [[] for _ in active_positions]

        print(
            f"[round {round_num}] {len(active_positions)} in flight, "
            f"{num_completed}/{total} done", file=sys.stderr,
        )
        for pos, slot_idx in enumerate(active_positions):
            state = slots[slot_idx]
            step_sample(state, raw_texts[pos], decode_warnings_batch[pos], args.tool_call_format, known_tool_names)
            if state["finished"]:
                results[state["orig_index"]] = finalize_state(state)
                num_completed += 1
                print(
                    f"  [{num_completed}/{total}] {state['sample_id']} "
                    f"stop_reason={state['stop_reason']}", file=sys.stderr,
                )
                fill_slot(slot_idx)

        done_so_far = [r for r in results if r is not None]
        with output_file.open("w", encoding="utf-8") as f:
            json.dump({"results": done_so_far, "summary": aggregate(done_so_far)}, f, indent=2)

    return results  # every slot filled by construction


def main() -> None:
    parser = argparse.ArgumentParser(description="Batched evaluation of a trained LALM's tool-use ability.")
    parser.add_argument("--benchmark-file", type=Path, default=Path(__file__).resolve().parent / "benchmark.json")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Omni-7B", help="Base model id/path.")
    parser.add_argument("--adapter-dir", type=str, default=None, help="LoRA checkpoint dir from ms-swift SFT.")
    parser.add_argument("--work-dir", type=Path, default=Path("/work/u1501463/tool_use_benchmark_predictions"))
    parser.add_argument("--output-file", type=Path, default=Path(__file__).resolve().parent / "results.json")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-extra-steps", type=int, default=2, help="Budget beyond the ground-truth chain length.")
    parser.add_argument("--hard-step-cap", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N samples (debugging).")
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Number of samples kept in flight (and decoded together each turn) at once. Bigger uses more GPU "
        "memory (KV cache for --batch-size concurrent conversations, each carrying several audios) but keeps an "
        "H100 busier; drop it if you hit an OOM. The pool is refilled from the queue as samples finish, so "
        "utilization stays high even once the tail of the benchmark has fewer than --batch-size samples left.",
    )
    parser.add_argument(
        "--backend", choices=["swift", "vllm"], default="swift",
        help="'swift' = ms-swift TransformersEngine (required for --adapter-dir; also runs any raw official "
        "model ms-swift recognizes). 'vllm' = plain vLLM, fastest path for an unmodified official checkpoint, "
        "no adapter support.",
    )
    parser.add_argument(
        "--model-type", type=str, default=None,
        help="Swift backend only: explicit model_type if it can't be auto-detected from --model "
        "(e.g. running another official model family).",
    )
    parser.add_argument("--max-model-len", type=int, default=20000, help="vLLM backend only.")
    parser.add_argument("--max-num-seqs", type=int, default=8, help="vLLM backend only.")
    parser.add_argument(
        "--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16", "auto"],
        help="Model weight/activation precision. Defaults to float32, which roughly doubles memory footprint "
        "versus the half-precision defaults these backends otherwise use -- pair with --device-map (swift) or "
        "--tensor-parallel-size (vllm), both defaulted below, to shard a 7B-class model across both GPUs.",
    )
    parser.add_argument(
        "--device-map", type=str, default="auto",
        help="Swift backend only: passed straight to ms-swift's TransformersEngine (accelerate device_map). "
        "'auto' shards the model across all visible GPUs -- needed at --dtype float32. Pass a single device "
        "string (e.g. 'cuda:0') to force single-GPU placement instead.",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=2,
        help="vLLM backend only: number of GPUs to shard the model across. Needed at --dtype float32, where a "
        "7B-class model no longer fits on one GPU.",
    )
    parser.add_argument(
        "--base-system-prompt", type=str, default=None,
        help="Override the system prompt's base greeting. See run_eval.py's resolve_system_prompt for defaults.",
    )
    parser.add_argument(
        "--system-prompt-model-dir", type=str, default=None,
        help="Path to the model directory to auto-detect the official default system message from. "
        "Defaults to --model.",
    )
    parser.add_argument(
        "--system-prompt-model-type", type=str, default=None,
        help="Fallback key into official_system_prompt.KNOWN_DEFAULT_SYSTEM_PROMPTS. Defaults to --model-type, "
        "then 'qwen2_5_omni'.",
    )
    parser.add_argument(
        "--no-system-prompt", action="store_true",
        help="Force no system prompt regardless of benchmark format or --adapter-dir.",
    )
    parser.add_argument(
        "--tool-call-format", type=str, default="qwen", choices=["qwen", "legacy"],
        help="Wire convention to parse the model's tool-call turns with (see tool_call_formats.py). "
        "Must match whatever the benchmark's 'tools_block' was generated with.",
    )
    args = parser.parse_args()
    args.system_prompt_model_type = args.system_prompt_model_type or args.model_type or "qwen2_5_omni"

    output_file = Path(args.output_file) if isinstance(args.output_file, str) else args.output_file
    output_summary_file = output_file.parent / (output_file.stem + "_summary.json")

    if args.backend == "vllm" and args.adapter_dir:
        raise SystemExit("--adapter-dir requires --backend swift; the vLLM backend only runs raw official weights.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1.")

    with args.benchmark_file.open("r", encoding="utf-8") as f:
        benchmark = json.load(f)
    if args.limit:
        benchmark = benchmark[: args.limit]

    args.work_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "swift":
        from interface.engine import SwiftEngine  # imported here so --help works without swift installed

        torch_dtype = None
        if args.dtype != "auto":
            import torch  # noqa: E402

            torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

        engine = SwiftEngine(
            model=args.model,
            adapter_dir=args.adapter_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            model_type=args.model_type,
            torch_dtype=torch_dtype,
            device_map=args.device_map,
            # Keep ms-swift's real forward-pass batch size in lockstep with the
            # in-flight pool size this script manages -- otherwise TransformersEngine
            # would re-chunk every generate_turns() call back down internally (see
            # SwiftEngine's docstring) and the pool would never actually decode together.
            max_batch_size=args.batch_size,
        )
    else:
        from interface.engine import VLLMEngine  # imported here so --help works without vllm installed

        engine = VLLMEngine(
            model=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_model_len=args.max_model_len,
            max_num_seqs=max(args.max_num_seqs, args.batch_size),
            max_audios_per_prompt=args.hard_step_cap + 2,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
        )

    results = run_batch(
        engine, benchmark, args.work_dir, args.max_extra_steps, args.hard_step_cap, args,
        batch_size=args.batch_size, output_file=output_file,
    )

    summary = aggregate(results)
    print(json.dumps(summary, indent=2))
    with output_file.open("w", encoding="utf-8") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)

    with output_summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
