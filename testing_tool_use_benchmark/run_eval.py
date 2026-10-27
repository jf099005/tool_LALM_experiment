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
from interface.executor import ToolExecutionError, UnknownToolError, extract_audio_outputs, run_tool_call  # noqa: E402
from interface.protocol import (  # noqa: E402
    AUDIO_TOKEN,
    audio_to_audio_tool_names,
    diagnose_tool_call_tags,
    parse_turn,
    render_tool_result_message,
)
from official_system_prompt import compose_system_prompt, load_official_system_prompt  # noqa: E402
from tool_analysis import analyze_per_tool, format_tool_analysis  # noqa: E402

# The SFT training data ends every chain with an explicit stop turn (see
# build_dataset.to_swift_sample), so it already learned the stop signal
# without needing the protocol spelled out in English. A fine-tuned model is
# therefore run with the same system prompt training used: the target model's
# own official default greeting + the tool catalogue (see
# `official_system_prompt.py`) -- NOT this protocol prompt. A zero-shot/
# official model has never seen this project's `audio_id`/`output_audio_id`
# chaining convention (it's specific to this task, not part of any model's
# own tool-calling training), so it needs that spelled out explicitly (this
# prompt), with the tool catalogue appended the same way -- the `<tool_call>`
# wire format itself needs no explanation for a Qwen model, since that's
# Qwen's own official convention (see `tool_call_formats.py`).
#
# Benchmark entries built before the tool catalogue moved into the system
# prompt have no "tools_block" field and already carry the catalogue inline
# in `entry["question"]` -- `resolve_system_prompt` below detects that and
# falls back to the old behavior (no system prompt for a fine-tuned model,
# this prompt alone with nothing appended for zero-shot) so old
# benchmark.json / old checkpoints keep evaluating exactly as before.
DEFAULT_PROTOCOL_SYSTEM_PROMPT = (
    "You are given a source audio (audio_0) and a target audio (audio_1), plus a list "
    "of audio-editing tools. Infer the chain of tool calls that transforms audio_0 into "
    "audio_1.\n\n"
    "Every call's `audio_id` argument must refer to audio_0, audio_1, or an "
    "`output_audio_id` you declared in an earlier turn. Every audio-producing tool also "
    "takes an `output_audio_id` argument: a fresh id you choose (never reuse audio_1, even "
    "for the call you believe finishes the chain). After each tool call you will be shown "
    "the real audio result, tagged with the output_audio_id you gave it, before your next "
    "turn. Once the chain is complete, stop calling tools and reply normally instead."
)


def render_initial_prompt(entry: Dict[str, Any]) -> str:
    """`entry["question"]` already carries the `<audio_0><audio>`/`<audio_1><audio>`
    tags inline (rendered up front by `build_dataset.build_one_sample` via
    `interface.protocol.audio_tag`), so this is just a passthrough -- no
    literal-path substitution needed here."""
    return entry["question"]


def resolve_system_prompt(entry: Dict[str, Any], args: argparse.Namespace) -> Optional[str]:
    """Reconstruct the exact system prompt training used for this entry.

    Keyed off whether `entry` has a `tools_block` (new-format raw dataset,
    tool catalogue lives in the system turn) or not (old-format, catalogue
    already inline in `entry["question"]`) so old benchmark files keep
    evaluating exactly as they did before this moved into the system prompt.
    """
    if args.no_system_prompt:
        return None
    tools_block = entry.get("tools_block")
    if args.base_system_prompt is not None:
        return compose_system_prompt(tools_block, base_system_prompt=args.base_system_prompt) if tools_block \
            else args.base_system_prompt
    if tools_block:
        if args.adapter_dir:
            # Fine-tuned: match the SFT system prompt -- the base model's own
            # official default + tools, same as gen_1st_stage_data/build_dataset.py.
            base = load_official_system_prompt(args.system_prompt_model_dir or args.model, args.system_prompt_model_type)
        else:
            base = DEFAULT_PROTOCOL_SYSTEM_PROMPT
        return compose_system_prompt(tools_block, base_system_prompt=base)
    # Old-format entry: preserve pre-existing behavior exactly.
    return None if args.adapter_dir else DEFAULT_PROTOCOL_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Per-tool parameter-accuracy metrics.
#
# Complements the existing name-only tool_name_precision/recall/f1: even when
# the predicted tool_name matches ground truth, this scores how close the
# *arguments* were (timestamp offsets for clipping/pre_emphasis/trim_silence,
# semitone offset for pitch_shift, rate offset for time_stretch, algorithm
# match + numeric offsets for denoise).
#
# ALIGNMENT ASSUMPTION: matches gt_calls[i] against predicted_steps[i] by
# position. This benchmark is currently single-step only (num_gt_steps == 1
# everywhere in benchmark.json), so position 0 vs position 0 is unambiguous.
# If the benchmark grows multi-step chains, positional alignment can silently
# mismatch a reordered-but-otherwise-correct chain -- revisit with a proper
# sequence-alignment (e.g. LCS on tool_name) before trusting this on
# multi-step entries.
#
# Shared verbatim by run_eval_parallel.py (imported, not re-implemented) so
# both scripts score parameters identically -- run_eval_batch.py is deprecated
# and intentionally NOT updated to call this.
# ---------------------------------------------------------------------------

def _parse_timestamp(ts: Any) -> Optional[float]:
    """'HH:MM:SS.mmm' -> seconds. Returns None for anything unparseable
    (including a missing/None/non-string value) rather than raising, since
    this is called on model-generated parameters that may be malformed."""
    if not isinstance(ts, str):
        return None
    parts = ts.split(":")
    if len(parts) != 3:
        return None
    try:
        hh, mm, ss = parts
        return int(hh) * 3600 + int(mm) * 60 + float(ss)
    except (ValueError, TypeError):
        return None


def _numeric_diff(pred_val: Any, gt_val: Any) -> Optional[float]:
    """abs(pred - gt) as float, or None if either side is missing/non-numeric."""
    if pred_val is None or gt_val is None:
        return None
    try:
        return abs(float(pred_val) - float(gt_val))
    except (TypeError, ValueError):
        return None


def compute_param_metrics(
    tool_name: str, gt_params: Dict[str, Any], pred_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Parameter-level accuracy for one (gt_call, predicted_step) pair whose
    tool_name already matches. Only ever set to a value when both sides
    actually provided the parameter (or, for pitch_shift's bins_per_octave,
    when the tool schema documents an explicit default) -- absent parameters
    stay None rather than being scored as a match or a miss.
    """
    metrics: Dict[str, Any] = {}

    if tool_name in ("clipping", "pre_emphasis", "trim_silence"):
        gt_begin = _parse_timestamp(gt_params.get("audio_begin"))
        pred_begin = _parse_timestamp(pred_params.get("audio_begin"))
        gt_end = _parse_timestamp(gt_params.get("audio_end"))
        pred_end = _parse_timestamp(pred_params.get("audio_end"))
        metrics["audio_begin_offset_sec"] = (
            abs(pred_begin - gt_begin) if gt_begin is not None and pred_begin is not None else None
        )
        metrics["audio_end_offset_sec"] = (
            abs(pred_end - gt_end) if gt_end is not None and pred_end is not None else None
        )

    if tool_name == "pre_emphasis":
        metrics["coef_diff"] = _numeric_diff(pred_params.get("coef"), gt_params.get("coef"))

    if tool_name == "trim_silence":
        metrics["threshold_db_diff"] = _numeric_diff(pred_params.get("threshold_db"), gt_params.get("threshold_db"))
        metrics["frame_length_diff"] = _numeric_diff(pred_params.get("frame_length"), gt_params.get("frame_length"))
        metrics["hop_length_diff"] = _numeric_diff(pred_params.get("hop_length"), gt_params.get("hop_length"))

    if tool_name == "pitch_shift":
        metrics["n_steps_diff"] = _numeric_diff(pred_params.get("n_steps"), gt_params.get("n_steps"))
        # Schema documents bins_per_octave's default as 12 -- treat an omitted
        # value as that default rather than "missing", on both sides.
        gt_bpo = gt_params.get("bins_per_octave", 12)
        pred_bpo = pred_params.get("bins_per_octave", 12)
        metrics["bins_per_octave_diff"] = _numeric_diff(pred_bpo, gt_bpo)

    if tool_name == "time_stretch":
        metrics["rate_diff"] = _numeric_diff(pred_params.get("rate"), gt_params.get("rate"))

    if tool_name == "denoise":
        gt_algo = gt_params.get("algorithm")
        pred_algo = pred_params.get("algorithm")
        metrics["algorithm_match"] = (
            (gt_algo == pred_algo) if gt_algo is not None and pred_algo is not None else None
        )
        # noise_factor only applies to spectral_subtraction, sensitivity only
        # to adaptive -- only score them when the GT call actually used them.
        if gt_params.get("noise_factor") is not None:
            metrics["noise_factor_diff"] = _numeric_diff(pred_params.get("noise_factor"), gt_params.get("noise_factor"))
        if gt_params.get("sensitivity") is not None:
            metrics["sensitivity_diff"] = _numeric_diff(pred_params.get("sensitivity"), gt_params.get("sensitivity"))

    gt_audio_id = gt_params.get("audio_id")
    pred_audio_id = pred_params.get("audio_id")
    metrics["audio_id_match"] = (
        (gt_audio_id == pred_audio_id) if gt_audio_id is not None and pred_audio_id is not None else None
    )

    return metrics


def compute_step_param_metrics(
    gt_calls: List[Dict[str, Any]], predicted_steps: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Positionally align gt_calls with predicted_steps (see module-level
    ALIGNMENT ASSUMPTION note above) and compute per-tool parameter metrics
    wherever the aligned predicted step succeeded and named the right tool."""
    out: List[Dict[str, Any]] = []
    for i in range(min(len(gt_calls), len(predicted_steps))):
        gt_call = gt_calls[i]
        pred_step = predicted_steps[i]
        if not pred_step.get("success"):
            continue
        if pred_step.get("tool_name") != gt_call.get("tool_name"):
            continue
        gt_params = gt_call.get("parameters", {}) or {}
        pred_params = pred_step.get("parameters", {}) or {}
        out.append({
            "step_index": i,
            "tool_name": gt_call["tool_name"],
            "metrics": compute_param_metrics(gt_call["tool_name"], gt_params, pred_params),
        })
    return out


def run_sample(
    engine,
    entry: Dict[str, Any],
    work_dir: Path,
    max_extra_steps: int,
    hard_step_cap: int,
    system_prompt: Optional[str] = None,
    tool_call_format: str = "qwen",
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

    # Mirrors `interface.agent.ToolCallingAgent.run`'s audio_id bookkeeping:
    # every audio the model can address by id (including audio_1, the
    # pre-revealed target) is tracked here, and each call's `audio_id` is
    # actually resolved against it -- not silently ignored in favor of
    # whatever the previous step happened to produce.
    audio_id_map: Dict[str, str] = {"audio_0": audio_a, "audio_1": audio_b}
    last_audio_id = "audio_0"
    current_audio = Path(audio_a)
    predicted_steps: List[Dict[str, Any]] = []
    seen_calls = set()
    stop_reason = "max_steps_reached"
    known_tool_names = set(audio_to_audio_tool_names())

    accumulated_text = ''
    decode_warnings: List[Dict[str, Any]] = []
    num_turns_generated = 0
    tag_malformed_valid_content_steps: List[Dict[str, Any]] = []

    for step_idx in range(1, max_steps + 1):
        raw_text = engine.generate_turn(messages, audios)
        num_turns_generated += 1
        for warning in getattr(engine, "last_decode_warnings", None) or []:
            decode_warnings.append({"step": step_idx, **warning})
        accumulated_text += raw_text + '\n'
        messages.append({"role": "assistant", "content": raw_text})

        call = parse_turn(raw_text, tool_call_format=tool_call_format)
        if call is None or call.get("done"):
            # Strict parsing found no (valid) tool call -- check whether that's
            # because there really wasn't one, or because the JSON payload was
            # fine but the `<tool_call>`/`</tool_call>` wrapper tokens were
            # malformed/missing (see `ToolCallFormat.diagnose`), which strict
            # parsing can't tell apart from "no tool call" since a broken
            # wrapper means it never finds the JSON at all.
            diag = diagnose_tool_call_tags(raw_text, tool_call_format=tool_call_format)
            content = diag["content"]
            if not diag["tag_ok"] and content and content["tool_name"] in known_tool_names:
                tag_malformed_valid_content_steps.append({
                    "step": step_idx,
                    "tool_name": content["tool_name"],
                    "parameters": content["parameters"],
                })
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
        if tool_name not in known_tool_names:
            stop_reason = "disallowed_tool"
            break
        parameters = call.get("parameters", {}) if isinstance(call.get("parameters"), dict) else {}
        call_signature = (tool_name, json.dumps(parameters, sort_keys=True))
        if call_signature in seen_calls:
            stop_reason = "repeated_call"
            break
        seen_calls.add(call_signature)

        input_audio_id = parameters.get("audio_id") or last_audio_id
        input_audio_path = audio_id_map.get(input_audio_id) if input_audio_id else None
        if input_audio_path is None:
            predicted_steps.append({
                "step": step_idx,
                "tool_name": tool_name,
                "parameters": parameters,
                "success": False,
                "error": f"unknown audio_id '{input_audio_id}'",
            })
            stop_reason = "unknown_audio_id"
            break

        exec_parameters = dict(parameters)
        step_record: Dict[str, Any] = {"step": step_idx, "tool_name": tool_name, "parameters": parameters}
        try:
            result = run_tool_call(tool_name, exec_parameters, Path(input_audio_path), sample_dir, step_idx)
        except (UnknownToolError, ToolExecutionError) as exc:
            step_record["success"] = False
            step_record["error"] = str(exc)
            predicted_steps.append(step_record)
            stop_reason = "tool_execution_failed"
            break

        step_record["success"] = True
        predicted_steps.append(step_record)

        output_audio_id = parameters.get("output_audio_id") or f"audio_{step_idx + 1}"
        audio_outputs = extract_audio_outputs(result)
        if not audio_outputs:
            # Disallowed above, but a tool could still legitimately return no audio
            # for other reasons -- feed back whatever text it gave and leave
            # current_audio/last_audio_id (and the audio list) alone.
            text = result.get("transcript") or result.get("message") or json.dumps(result)
            tool_message = render_tool_result_message(step_idx, tool_name, [], text=text)
        else:
            # Register every stem the tool produced under its own id (mirrors
            # `agent.py`: primary output under `output_audio_id`, any residual
            # under `{output_audio_id}_{stem}`) so a later call can address
            # either by id -- not only the primary one.
            for stem, path in audio_outputs.items():
                audio_id = output_audio_id if stem in ("", "target") else f"{output_audio_id}_{stem}"
                audio_id_map[audio_id] = str(path)
                audios.append(str(path))
            current_audio = Path(
                audio_outputs.get("target") or audio_outputs.get("") or next(iter(audio_outputs.values()))
            )
            last_audio_id = output_audio_id
            tool_message = render_tool_result_message(step_idx, tool_name, [f"<{output_audio_id}>{AUDIO_TOKEN}"])
        messages.append({"role": "tool", "content": tool_message})
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

    baseline_metrics = compare_audio(audio_a, audio_b)

    step_param_metrics = compute_step_param_metrics(gt_calls, predicted_steps)

    return {
        "id": sample_id,
        "num_gt_steps": len(gt_calls),
        "gt_tool_sequence": gt_tool_names,
        "gt_calls": gt_calls,
        "prompt": messages[0]["content"] if system_prompt else None,
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
        "step_param_metrics": step_param_metrics,
        "final_audio_path": str(current_audio),
        "audio_metrics": audio_metrics,
        "baseline_metrics": baseline_metrics,
        "LLM_output": accumulated_text,
        "decode_warnings": decode_warnings,
        "num_turns_generated": num_turns_generated,
        "num_tag_malformed_valid_content": len(tag_malformed_valid_content_steps),
        "tag_malformed_valid_content_steps": tag_malformed_valid_content_steps,
    }


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {}

    total_predicted = sum(r["num_predicted_calls"] for r in results)
    total_successful = sum(r["num_successful_calls"] for r in results)
    total_turns_generated = sum(r["num_turns_generated"] for r in results)
    total_tag_malformed_valid_content = sum(r["num_tag_malformed_valid_content"] for r in results)

    def avg(key_path):
        vals = [key_path(r) for r in results]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and v != v)]
        return sum(vals) / len(vals) if vals else None

    by_num_steps: Dict[int, Dict[str, Any]] = {}
    for r in results:
        bucket = by_num_steps.setdefault(r["num_gt_steps"], {"n": 0, "f1_sum": 0.0, "closeness_sum": 0.0, "closeness_diff": 0.0})
        bucket["n"] += 1
        bucket["f1_sum"] += r["tool_name_f1"]
        bucket["closeness_sum"] += r["audio_metrics"]["closeness_score"]
        bucket['closeness_diff'] += r["audio_metrics"]["closeness_score"] - r["baseline_metrics"]["closeness_score"]

    breakdown = {
        k: {
            "n": v["n"],
            "mean_tool_name_f1": v["f1_sum"] / v["n"],
            "mean_audio_closeness": v["closeness_sum"] / v["n"],
        }
        for k, v in sorted(by_num_steps.items())
    }

    # Per-tool parameter-accuracy aggregation: gather every step-level
    # metrics dict (already restricted to "predicted the right tool at the
    # aligned position" -- see compute_step_param_metrics) by tool_name, then
    # average each numeric/boolean field independently across whatever
    # entries actually had a non-None value for it. A field's absence from
    # a given step (e.g. noise_factor_diff for a wiener denoise call) is
    # simply excluded from that field's average, not treated as 0 or as a
    # miss.
    by_tool_fields: Dict[str, Dict[str, List[float]]] = {}
    by_tool_n: Dict[str, int] = {}
    for r in results:
        for entry in r.get("step_param_metrics", []):
            tool_name = entry["tool_name"]
            by_tool_n[tool_name] = by_tool_n.get(tool_name, 0) + 1
            field_bucket = by_tool_fields.setdefault(tool_name, {})
            for field_name, value in entry["metrics"].items():
                if value is None:
                    continue
                field_bucket.setdefault(field_name, []).append(float(value))

    param_metrics_by_tool = {
        tool_name: {
            "n": by_tool_n[tool_name],
            **{
                f"mean_{field_name}": (sum(vals) / len(vals))
                for field_name, vals in fields.items()
            },
        }
        for tool_name, fields in sorted(by_tool_fields.items())
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

        "mean_audio_log_mel_cosine_diff": avg(lambda r: r["audio_metrics"]["log_mel_cosine"] - r["baseline_metrics"]["log_mel_cosine"]),
        "mean_audio_mfcc_cosine_diff": avg(lambda r: r["audio_metrics"]["mfcc_cosine"] - r["baseline_metrics"]["mfcc_cosine"]),
        "mean_audio_closeness_diff": avg(lambda r: r["audio_metrics"]["closeness_score"] - r["baseline_metrics"]["closeness_score"]),

        "mean_duration_ratio": avg(lambda r: r["audio_metrics"]["duration_ratio"]),
        "stop_reason_counts": dict(Counter(r["stop_reason"] for r in results)),

        # Turns where the model's JSON payload named a real tool with valid
        # arguments, but the `<tool_call>`/`</tool_call>` wrapper tokens
        # around it were missing/malformed -- so `parse_turn` never saw it
        # and (in the common case) treated the turn as "done". Distinguishes
        # "the model can't produce a correct tool call" from "it can, but
        # botches the wrapper tokens", which the other tool-call metrics
        # above conflate (both look like a parse/execution failure to them).
        "total_tag_malformed_valid_content": total_tag_malformed_valid_content,
        "tag_malformed_valid_content_rate_over_turns": (
            total_tag_malformed_valid_content / total_turns_generated
        ) if total_turns_generated else 0.0,
        "num_samples_with_tag_malformed_valid_content": sum(
            1 for r in results if r["num_tag_malformed_valid_content"] > 0
        ),
        "breakdown_by_num_gt_steps": breakdown,
        "param_metrics_by_tool": param_metrics_by_tool,
        # Full per-GT-tool breakdown: selection accuracy/P/R/F1, confusion
        # matrix vs other tools, execution/audio stats, and a direct
        # parameter-by-parameter comparison against GT (see tool_analysis.py).
        "per_tool_analysis": analyze_per_tool(results),
    }


def write_outputs(output_file: Path, results: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    """Final outputs shared by run_eval.py and run_eval_parallel.py:
      <out>.json                 per-sample results + summary
      <out>_summary.json         summary only
      <out>_tool_analysis.json   per-tool analysis incl. every raw (gt, pred) parameter pair
      <out>_tool_analysis.txt    human-readable per-tool report (also printed to stderr)
    """
    with output_file.open("w", encoding="utf-8") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)
    with (output_file.parent / (output_file.stem + "_summary.json")).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    detailed = analyze_per_tool(results, include_pairs=True)
    with (output_file.parent / (output_file.stem + "_tool_analysis.json")).open("w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2)
    report = format_tool_analysis(detailed)
    with (output_file.parent / (output_file.stem + "_tool_analysis.txt")).open("w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report, file=sys.stderr)


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
        help="Override the system prompt's base greeting. The tool catalogue is appended after it when the "
        "benchmark entry has a 'tools_block' (new-format dataset); for an old-format entry (catalogue already "
        "inline in entry['question']) this is used verbatim with nothing appended. Default: auto-detected from "
        "--system-prompt-model-dir when --adapter-dir is set (matches SFT training); the built-in zero-shot "
        "tool-calling protocol prompt otherwise -- see resolve_system_prompt.",
    )
    parser.add_argument(
        "--system-prompt-model-dir", type=str, default=None,
        help="Path to the model directory to auto-detect the official default system message from (only used "
        "for --adapter-dir runs against new-format benchmark entries). Defaults to --model.",
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

    output_file = args.output_file
    output_file = Path(output_file) if isinstance(output_file, str) else output_file

    if args.backend == "vllm" and args.adapter_dir:
        raise SystemExit("--adapter-dir requires --backend swift; the vLLM backend only runs raw official weights.")

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
        )
    else:
        from interface.engine import VLLMEngine  # imported here so --help works without vllm installed

        engine = VLLMEngine(
            model=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_audios_per_prompt=args.hard_step_cap + 2,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
        )

    results = []
    for idx, entry in enumerate(benchmark, start=1):
        print(f"[{idx}/{len(benchmark)}] {entry['id']}", file=sys.stderr)
        system_prompt = resolve_system_prompt(entry, args)
        result = run_sample(
            engine, entry, args.work_dir, args.max_extra_steps, args.hard_step_cap, system_prompt,
            tool_call_format=args.tool_call_format,
        )
        results.append(result)
        with args.output_file.open("w", encoding="utf-8") as f:
            json.dump({"results": results, "summary": aggregate(results)}, f, indent=2)

    summary = aggregate(results)
    print(json.dumps(summary, indent=2))
    write_outputs(output_file, results, summary)

if __name__ == "__main__":
    main()
