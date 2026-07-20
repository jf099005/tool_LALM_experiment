"""Evaluate a trained LALM's tool-use ability on the held-out benchmark.

For each (audio A, audio B, tool toolset) item, feed the model A and B (same
framing as training: "infer the tool chain that turns A into B"), let it
generate tool calls turn by turn, actually execute each predicted call against
the real audio, and feed the real result back as the next turn's audio -- this
mirrors the multi-turn structure `build_dataset.to_swift_sample` used for SFT.

Two questions get answered per sample:
  1. Tool-calling ability: how many of the model's emitted tool calls parsed
     and executed successfully, and how well does the predicted tool sequence
     match the ground-truth chain (exact-match / precision / recall / F1)?
  2. Audio fidelity: how close is the audio the model's tool chain actually
     produced to the ground-truth target B (log-mel / MFCC cosine, etc, from
     audio_metrics.compare_audio)?

Must run under the `ms-swift` conda env. Usage:
    ~/miniconda3/envs/ms-swift/bin/python run_eval.py \
        --benchmark-file benchmark.json \
        --model Qwen/Qwen2.5-Omni-7B \
        --adapter-dir ../output/v10-20260625-215142/checkpoint-500 \
        --work-dir /work/u1501463/tool_use_benchmark_predictions \
        --output-file results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_metrics import compare_audio  # noqa: E402
from tools.predicted_executor import ToolExecutionError, UnknownToolError, execute_predicted_tool_call  # noqa: E402

AUDIO_TOKEN = "<audio>"

# The SFT training data ends every chain with an explicit {"done": true} turn
# (see build_dataset.to_swift_sample), so a fine-tuned model is run with no
# system prompt, same as training -- it already learned the stop signal. A
# zero-shot/official model has never seen that JSON convention, so it needs to
# be told the protocol explicitly, including the same stop signal.
DEFAULT_PROTOCOL_SYSTEM_PROMPT = (
    "You are given a source audio (audio_0) and a target audio (audio_1), plus a list "
    "of audio-editing tools. Infer the chain of tool calls that transforms audio_0 into "
    "audio_1.\n\n"
    "Respond with exactly one JSON object per turn and nothing else:\n"
    '  {"tool_name": "<tool name>", "parameters": {"audio_id": "<id of the audio to read>", '
    '<other tool arguments>}, "output_audio_id": "<a new id for this call\'s output>"}\n\n'
    "`audio_id` must refer to audio_0, audio_1, or an output_audio_id you declared in an "
    "earlier turn. After each tool call you will be shown the real audio result, tagged "
    "with the output_audio_id you gave it, before your next turn -- always give each call's "
    'output a fresh id (never reuse audio_1, even for the call you believe finishes the '
    'chain). Once the chain is complete, respond with {"done": true} instead of another '
    "tool call."
)


def render_initial_prompt(entry: Dict[str, Any]) -> str:
    """Same substitution as build_dataset.to_swift_sample: literal A/B paths -> tagged audio_token."""
    text = entry["question"].replace(entry["source_audio"], f"<audio_0>{AUDIO_TOKEN}", 1)
    text = text.replace(entry["target_audio"], f"<audio_1>{AUDIO_TOKEN}", 1)
    return text


def parse_tool_call(raw_text: str) -> Optional[Dict[str, Any]]:
    """Parse one predicted assistant turn as a JSON object.

    Returns the parsed dict as-is -- callers distinguish a tool call
    (`tool_name` present) from an explicit stop signal (`done` present) from
    garbage (neither key present). Returns None if the turn isn't even valid
    JSON.
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
    if not isinstance(obj, dict):
        return None
    return obj


def run_sample(
    engine,
    entry: Dict[str, Any],
    work_dir: Path,
    max_extra_steps: int,
    hard_step_cap: int,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
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
    audios: List[str] = [audio_a, audio_b]

    current_audio = Path(audio_a)
    predicted_steps: List[Dict[str, Any]] = []
    seen_calls = set()
    stop_reason = "max_steps_reached"

    accumulated_text = ''

    for step_idx in range(1, max_steps + 1):
        raw_text = engine.generate_turn(messages, audios)
        accumulated_text += raw_text + '\n'
        messages.append({"role": "assistant", "content": raw_text})

        call = parse_tool_call(raw_text)
        if call is None:
            stop_reason = "unparseable_output"
            break
        if call.get("done"):
            stop_reason = "model_signaled_done"
            break

        tool_name = call.get("tool_name")
        if not tool_name:
            stop_reason = "unparseable_output"
            break
        parameters = call.get("parameters", {}) if isinstance(call.get("parameters"), dict) else {}
        call_signature = (tool_name, json.dumps(parameters, sort_keys=True))
        if call_signature in seen_calls:
            stop_reason = "repeated_call"
            break
        seen_calls.add(call_signature)

        out_path = sample_dir / f"pred_step{step_idx}_{tool_name}.wav"
        step_record: Dict[str, Any] = {"step": step_idx, "tool_name": tool_name, "parameters": parameters}
        try:
            new_audio = execute_predicted_tool_call(tool_name, parameters, current_audio, out_path)
            step_record["success"] = True
            current_audio = new_audio
        except (UnknownToolError, ToolExecutionError) as exc:
            step_record["success"] = False
            step_record["error"] = str(exc)
            predicted_steps.append(step_record)
            stop_reason = "tool_execution_failed"
            break

        predicted_steps.append(step_record)
        output_audio_id = call.get("output_audio_id") or f"audio_{step_idx + 1}"
        messages.append({"role": "tool", "content": f"Output of step {step_idx}: <{output_audio_id}>{AUDIO_TOKEN}"})
        audios.append(str(current_audio))
    else:
        stop_reason = "max_steps_reached"

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

    # Distinct-tool coverage: did the model actually invoke (successfully) each
    # unique tool that appears in the ground-truth chain, regardless of order/count.
    gt_tool_set = set(gt_tool_names)
    pred_successful_set = set(pred_successful_names)
    gt_tools_used = sorted(gt_tool_set & pred_successful_set)
    gt_tools_missed = sorted(gt_tool_set - pred_successful_set)
    gt_tool_coverage = (len(gt_tools_used) / len(gt_tool_set)) if gt_tool_set else 1.0

    audio_metrics = compare_audio(str(current_audio), audio_b)

    return {
        "id": sample_id,
        "num_gt_steps": len(gt_calls),
        "gt_tool_sequence": gt_tool_names,
        "predicted_steps": predicted_steps,
        "predicted_attempted_sequence": pred_attempted_names,
        "predicted_successful_sequence": pred_successful_names,
        "stop_reason": stop_reason,
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
        "LLM_output": accumulated_text,
    }


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {}

    total_predicted = sum(r["num_predicted_calls"] for r in results)
    total_successful = sum(r["num_successful_calls"] for r in results)

    def avg(key_path):
        vals = [key_path(r) for r in results]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and v != v)]
        return sum(vals) / len(vals) if vals else None

    by_num_steps: Dict[int, Dict[str, Any]] = {}
    for r in results:
        bucket = by_num_steps.setdefault(r["num_gt_steps"], {"n": 0, "f1_sum": 0.0, "closeness_sum": 0.0})
        bucket["n"] += 1
        bucket["f1_sum"] += r["tool_name_f1"]
        bucket["closeness_sum"] += r["audio_metrics"]["closeness_score"]
    breakdown = {
        k: {
            "n": v["n"],
            "mean_tool_name_f1": v["f1_sum"] / v["n"],
            "mean_audio_closeness": v["closeness_sum"] / v["n"],
        }
        for k, v in sorted(by_num_steps.items())
    }

    return {
        "num_samples": n,
        "total_predicted_tool_calls": total_predicted,
        "total_successful_tool_calls": total_successful,
        "overall_tool_call_success_rate": (total_successful / total_predicted) if total_predicted else 0.0,
        "mean_exact_sequence_match": avg(lambda r: float(r["exact_sequence_match"])),
        "mean_tool_name_precision": avg(lambda r: r["tool_name_precision"]),
        "mean_tool_name_recall": avg(lambda r: r["tool_name_recall"]),
        "mean_tool_name_f1": avg(lambda r: r["tool_name_f1"]),
        "mean_gt_tool_coverage": avg(lambda r: r["gt_tool_coverage"]),
        "mean_used_any_gt_tool": avg(lambda r: float(r["used_any_gt_tool"]) if r["used_any_gt_tool"] is not None else None),
        "mean_used_all_gt_tools": avg(lambda r: float(r["used_all_gt_tools"]) if r["used_all_gt_tools"] is not None else None),
        "mean_audio_log_mel_cosine": avg(lambda r: r["audio_metrics"]["log_mel_cosine"]),
        "mean_audio_mfcc_cosine": avg(lambda r: r["audio_metrics"]["mfcc_cosine"]),
        "mean_audio_closeness_score": avg(lambda r: r["audio_metrics"]["closeness_score"]),
        "mean_duration_ratio": avg(lambda r: r["audio_metrics"]["duration_ratio"]),
        "stop_reason_counts": dict(Counter(r["stop_reason"] for r in results)),
        "breakdown_by_num_gt_steps": breakdown,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained LALM's tool-use ability.")
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
        "--system-prompt", type=str, default=None,
        help="Override the system prompt. Default: none when --adapter-dir is set (matches the SFT data, which "
        "had no system turn); the built-in zero-shot tool-calling protocol prompt otherwise.",
    )
    parser.add_argument(
        "--no-system-prompt", action="store_true",
        help="Force no system prompt even for a zero-shot/official-model run.",
    )
    args = parser.parse_args()

    output_file = args.output_file
    output_file = Path(output_file) if isinstance(output_file, str) else output_file

    output_summary_file = output_file.parent / (output_file.stem + "_summary.json")
    
    if args.backend == "vllm" and args.adapter_dir:
        raise SystemExit("--adapter-dir requires --backend swift; the vLLM backend only runs raw official weights.")

    if args.system_prompt is not None:
        system_prompt = args.system_prompt
    elif args.no_system_prompt or args.adapter_dir:
        system_prompt = None
    else:
        system_prompt = DEFAULT_PROTOCOL_SYSTEM_PROMPT

    with args.benchmark_file.open("r", encoding="utf-8") as f:
        benchmark = json.load(f)
    if args.limit:
        benchmark = benchmark[: args.limit]

    args.work_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "swift":
        from model_engine import SwiftEngine  # imported here so --help works without swift installed

        engine = SwiftEngine(
            model=args.model,
            adapter_dir=args.adapter_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            model_type=args.model_type,
        )
    else:
        from model_engine import VLLMEngine  # imported here so --help works without vllm installed

        engine = VLLMEngine(
            model=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_audios_per_prompt=args.hard_step_cap + 2,
        )

    results = []
    for idx, entry in enumerate(benchmark, start=1):
        print(f"[{idx}/{len(benchmark)}] {entry['id']}", file=sys.stderr)
        result = run_sample(engine, entry, args.work_dir, args.max_extra_steps, args.hard_step_cap, system_prompt)
        results.append(result)
        with args.output_file.open("w", encoding="utf-8") as f:
            json.dump({"results": results, "summary": aggregate(results)}, f, indent=2)

    summary = aggregate(results)
    print(json.dumps(summary, indent=2))
    with args.output_file.open("w", encoding="utf-8") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)

    with output_summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
