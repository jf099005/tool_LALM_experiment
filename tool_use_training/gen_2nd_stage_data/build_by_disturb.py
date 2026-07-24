"""Build stage-2 tool-use training data: disturb -> recover.

Stage 1 (`gen_1st_stage_data/build_dataset.py`) teaches a model to reproduce an
arbitrary target audio B from a source A by calling tools -- fitting the tool-call
*instruction format*. Stage 2 teaches something narrower and directly tied to
benchmark performance (e.g. MMAU): given a QA pair whose audio has been degraded by a
data-augmentation op, call the tool(s) that clean it back up so the question is
answerable again -- the same move a model should make at inference time when the
benchmark audio is noisy, padded with junk, has a stray background event, or was
sped up / pitch-shifted.

Pipeline
--------
1. Load an input JSON list of QA pairs (dcase_subset.json-shaped: question / choice /
   answer / id / audio_path, ...). Sampling a subset from some larger benchmark is
   *not* this script's job -- see `sample_qa_pairs.py` for that.
2. For each QA pair, draw k in [--min-ops, --max-ops] distinct disturbance ops from
   `disturb_recover.DISTURB_REGISTRY` (add_noise, pad_noise, insert_background_event,
   change_speed, change_pitch -- audio_edit/'s full op set) and apply them in sequence,
   writing the disturbed audio under --output-dir.
3. Walk that disturbance chain in reverse; for each step, `disturb_recover` picks one
   applicable recovery strategy (a tools/-backed tool + parameters) that undoes or
   best-effort cleans up that specific disturbance. The whole reversed chain is then
   actually executed, one "turn" at a time, batching same-tool calls across all samples
   and dispatching each batch via `tools/tool_batch_execute.py` in that tool's conda env
   (`disturb_recover.TOOL_ENV`) -- mirrors what `apply_tools.py` does, but chains each
   turn's output into the next turn's input, which `apply_tools.py`'s own schedule
   format cannot express in one file (see step 4). Recovered audio lands under
   --tool-results-dir.
4. Because of that same limitation, the apply_tools.py-compatible schedule is written
   as one JSON file *per recovery turn* (tool_schedules/turn_1.json, turn_2.json, ...)
   instead of a single file -- turn t's "problem" audio_path is turn (t-1)'s real
   output, so re-running `python apply_tools.py --schedule_path tool_schedules/turn_t.json
   --output_root <root>/turn_t` for t = 1..N reproduces this script's own recovery
   step by step.
5. Two dcase_subset.json-shaped files are written: disturbed_subset.json (audio_path ->
   the disturbed audio) and recovered_subset.json (audio_path -> the final recovered
   audio) -- both directly usable as --subset_path for qwen25_with_tool_chain_evaluation.py.

See DISTURB_RECOVER_MAPPING.md (same directory) for the disturb-op -> recovery-tool
table and the reasoning behind each pairing.

Usage:
    python build_by_disturb.py \\
        --input-json /work/u1501463/mmau_subset.json \\
        --output-dir /work/u1501463/gen_2nd_stage_disturb \\
        --tool-results-dir /work/u1501463/gen_2nd_stage_recover \\
        --min-ops 1 --max-ops 1

Run under an env with librosa/soundfile (e.g. `Whisper` or `ms-swift` -- the same ones
`gen_1st_stage_data/build_dataset.py` needs) since step 2's disturbance ops depend on
them directly; step 3's recovery execution shells out to each tool's own env, so it
does not further constrain which env this script itself runs under.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import disturb_recover as dr  # noqa: E402


# tools/-package tool names where `Tool.requires_output_path()` is True (see
# tools/abstract_tool.py) -- these dispatch through tool_batch_execute.py's own
# conda env, so this can't just import the real Tool classes to ask them directly.
_TOOLS_REQUIRING_OUTPUT_PATH = {
    "denoise", "pitch_shift", "time_stretch", "human_voice_enhance", "super_resolution",
    "amplitude_normalize", "loudness_normalize", "remove_dc_offset", "spectral_normalize",
    "trim_silence", "pre_emphasis",
}


def load_qa_pairs(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise SystemExit(f"{path} must contain a JSON list of QA-pair dicts.")
    return data


def get_audio_path(item: Dict[str, Any]) -> str:
    audio_path = item.get("audio_path") or item.get("audio_url")
    if not audio_path:
        raise ValueError(f"QA pair {item.get('id')!r} has no audio_path/audio_url.")
    return str(Path(audio_path).expanduser().resolve())


def to_dcase_entry(item: Dict[str, Any], audio_path: str) -> Dict[str, Any]:
    """Project a QA pair + a (possibly new) audio path into dcase_subset.json's shape."""
    entry = {
        "question": item.get("question", ""),
        "choice": item.get("choice", []),
        "answer": item.get("answer", "N/A"),
        "id": item["id"],
        "audio_url": audio_path,
        "question_type": item.get("question_type"),
        "_json_path": item.get("_json_path"),
        "audio_path": audio_path,
    }
    return entry


# ---------------------------------------------------------------------------
# Step 2: disturb
# ---------------------------------------------------------------------------

def disturb_item(
    item: Dict[str, Any],
    sample_dir: Path,
    min_ops: int,
    max_ops: int,
    rng: random.Random,
) -> Dict[str, Any]:
    """Apply k in [min_ops, max_ops] distinct disturbance ops to one QA pair's audio.

    Returns a record with the disturbance chain (each step's op/params/durations) and
    the final disturbed audio path. Raises on failure so the caller can retry with a
    fresh random draw.
    """
    source = Path(get_audio_path(item))
    sample_dir.mkdir(parents=True, exist_ok=True)

    op_names = dr.available_disturb_names()
    k = rng.randint(min_ops, max_ops)
    k = max(0, min(k, len(op_names)))
    chosen_ops = rng.sample(op_names, k)

    current_path = source
    chain: List[Dict[str, Any]] = []
    for step_index, op_name in enumerate(chosen_ops, start=1):
        input_duration = dr.get_duration_seconds(current_path)
        out_path = sample_dir / f"disturb_{step_index}_{op_name}.wav"
        params, current_path = dr.DISTURB_REGISTRY[op_name].apply(current_path, out_path, rng, input_duration)
        output_duration = dr.get_duration_seconds(current_path)
        chain.append({
            "op": op_name,
            "params": params,
            "_input_duration": input_duration,
            "_output_duration": output_duration,
            "audio_path": str(current_path),
        })

    if chain:
        disturbed_path = sample_dir / "disturbed.wav"
        if current_path != disturbed_path:
            current_path.replace(disturbed_path)
            chain[-1]["audio_path"] = str(disturbed_path)
    else:
        # k == 0: nothing to disturb -- disturbed/recovered both alias the source
        # directly rather than making a pointless (and format-lossy, if the
        # source isn't already a WAV) copy.
        disturbed_path = source

    return {
        "id": item["id"],
        "source_audio": str(source),
        "disturb_chain": chain,
        "disturbed_audio": str(disturbed_path),
    }


# ---------------------------------------------------------------------------
# Step 3: build the (reversed) recovery chain
# ---------------------------------------------------------------------------

def build_recovery_chain(disturb_chain: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    recovery_chain: List[Dict[str, Any]] = []
    for fwd_step in reversed(disturb_chain):
        merged_params = {
            **fwd_step["params"],
            "_input_duration": fwd_step["_input_duration"],
            "_output_duration": fwd_step["_output_duration"],
        }
        strategy_name, tool_name, parameters = dr.build_recovery_step(fwd_step["op"], merged_params, rng)
        recovery_chain.append({
            "recovers_op": fwd_step["op"],
            "strategy": strategy_name,
            "tool": tool_name,
            "parameters": parameters,
        })
    return recovery_chain


# ---------------------------------------------------------------------------
# Step 3 (cont'd): execute the recovery chain, one turn at a time, batched by
# tool across every sample that has a step at that turn.
# ---------------------------------------------------------------------------

def run_batch(tool_name: str, batch_parameters: List[Dict[str, Any]], turn_dir: Path) -> List[Dict[str, Any]]:
    python_exe = dr.TOOL_ENV.get(tool_name, sys.executable)
    turn_dir.mkdir(parents=True, exist_ok=True)
    input_path = turn_dir / f"batch_{tool_name}_input.json"
    output_path = turn_dir / f"batch_{tool_name}_output.json"

    with input_path.open("w", encoding="utf-8") as handle:
        json.dump(batch_parameters, handle, indent=2)

    cmd = [
        python_exe,
        str(REPO_ROOT / "tools" / "tool_batch_execute.py"),
        "--tool-name", tool_name,
        "--input-file", str(input_path),
        "--output-file", str(output_path),
    ]
    print(f"  [{tool_name}] running {len(batch_parameters)} call(s) via {python_exe}", file=sys.stderr)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    with output_path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    if not isinstance(results, list):
        raise RuntimeError(f"Unexpected batch result for tool '{tool_name}': expected list, got {type(results).__name__}")
    return results


def execute_recovery_turns(
    items: List[Dict[str, Any]],
    schedule_dir: Path,
    tool_results_dir: Path,
    skip_execution: bool,
) -> None:
    """Run every sample's recovery chain turn by turn.

    Mutates each item in place: sets `current_audio_path` to the running recovered
    audio (starting from the disturbed audio) and appends per-turn execution records
    to `turn_results`. Also writes one apply_tools.py-compatible schedule JSON per
    turn under `schedule_dir`, independent of whether execution actually ran.
    """
    max_chain_len = max((len(it["recovery_chain"]) for it in items), default=0)
    schedule_dir.mkdir(parents=True, exist_ok=True)

    for turn in range(1, max_chain_len + 1):
        turn_items = [it for it in items if len(it["recovery_chain"]) >= turn and not it.get("failed")]
        if not turn_items:
            continue

        if skip_execution and turn > 1:
            # turn_1's audio_path is known upfront (the disturbed audio), but
            # turn >= 2's audio_path is turn (turn-1)'s *real* tool output --
            # which only exists once that turn actually ran. Without running
            # anything we cannot fill it in correctly, so stop here rather
            # than emit a schedule file with a wrong (stale) audio_path.
            prev_turn = turn - 1
            print(
                f"--skip-recovery-execution: stopping schedule generation after turn {prev_turn} "
                f"({len(turn_items)} sample(s) have longer chains) -- turn {turn}'s input audio only "
                f"exists once turn {prev_turn} actually runs. Drop --skip-recovery-execution, or run "
                f"turn_{prev_turn}.json through apply_tools.py yourself and re-derive later turns from "
                "its output.",
                file=sys.stderr,
            )
            break

        schedule_entries = []
        for it in turn_items:
            step = it["recovery_chain"][turn - 1]
            problem = {"id": it["id"], "audio_path": it["current_audio_path"]}
            schedule_entries.append([problem, [{"tool": step["tool"], "parameters": step["parameters"]}]])

        schedule_path = schedule_dir / f"turn_{turn}.json"
        with schedule_path.open("w", encoding="utf-8") as handle:
            json.dump(schedule_entries, handle, indent=2)
        print(f"Turn {turn}: wrote schedule for {len(turn_items)} sample(s) to {schedule_path}", file=sys.stderr)

        if skip_execution:
            continue

        by_tool: Dict[str, List[Dict[str, Any]]] = {}
        for it in turn_items:
            by_tool.setdefault(it["recovery_chain"][turn - 1]["tool"], []).append(it)

        turn_root = tool_results_dir / f"turn_{turn}"
        for tool_name, tool_items in by_tool.items():
            batch_parameters = []
            dest_paths = []
            for it in tool_items:
                step = it["recovery_chain"][turn - 1]
                params = dict(step["parameters"])
                params["audio_path"] = it["current_audio_path"]

                dest_dir = tool_results_dir / it["id"]
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / f"turn{turn}_{tool_name}.wav"
                dest_paths.append(dest_path)
                if tool_name in _TOOLS_REQUIRING_OUTPUT_PATH:
                    params["output_path"] = str(dest_path)

                batch_parameters.append(params)

            try:
                results = run_batch(tool_name, batch_parameters, turn_root)
            except (subprocess.CalledProcessError, RuntimeError) as exc:
                # A whole-batch failure (subprocess crash, or a non-list result --
                # e.g. the tool's own execute_batch raised, such as a busy/OOM GPU
                # for a heavy tool like remove_target) shouldn't take down every
                # other sample's recovery chain; isolate it to this tool's items.
                print(f"  [{tool_name}] batch failed, marking {len(tool_items)} sample(s) failed: {exc}", file=sys.stderr)
                for it in tool_items:
                    it["failed"] = True
                    it["failure_reason"] = f"turn {turn} tool {tool_name}: batch failed ({exc})"
                continue

            for it, call_params, dest_path, result in zip(tool_items, batch_parameters, dest_paths, results):
                if result.get("status") != "success":
                    it["failed"] = True
                    it["failure_reason"] = f"turn {turn} tool {tool_name}: {result.get('message', result)}"
                    continue

                produced_path = result.get("output_path") or result.get("clip_path")
                if not produced_path:
                    it["failed"] = True
                    it["failure_reason"] = f"turn {turn} tool {tool_name}: no output_path in result"
                    continue

                produced_path = Path(produced_path)
                if produced_path != dest_path:
                    shutil.copy(produced_path, dest_path)

                it["current_audio_path"] = str(dest_path)
                it.setdefault("turn_results", []).append({
                    "turn": turn,
                    "tool": tool_name,
                    "parameters": {k: v for k, v in call_params.items() if k not in ("audio_path", "output_path")},
                    "output_path": str(dest_path),
                })


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_one_sample(
    item: Dict[str, Any],
    disturb_work_dir: Path,
    min_ops: int,
    max_ops: int,
    rng: random.Random,
) -> Dict[str, Any]:
    sample_id = item["id"]
    sample_dir = disturb_work_dir / str(sample_id)
    record = disturb_item(item, sample_dir, min_ops, max_ops, rng)
    record["recovery_chain"] = build_recovery_chain(record["disturb_chain"], rng)
    record["current_audio_path"] = record["disturbed_audio"]
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Disturb QA-pair audio, then build+run a tool-call recovery chain.")
    parser.add_argument("--input-json", type=Path, required=True, help="JSON list of QA pairs (dcase_subset.json-shaped).")
    parser.add_argument("--output-dir", type=Path, default=Path("/work/u1501463/gen_2nd_stage_disturb"),
                         help="Where disturbed audio + disturbed_subset.json + manifest.json are written.")
    parser.add_argument("--tool-results-dir", type=Path, default=Path("/work/u1501463/gen_2nd_stage_recover"),
                         help="Where recovered audio + tool_schedules/*.json are written (the 'tool-results' location).")
    parser.add_argument("--min-ops", type=int, default=1, help="Minimum number of disturbance ops per sample.")
    parser.add_argument("--max-ops", type=int, default=1, help="Maximum number of disturbance ops per sample.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts-per-sample", type=int, default=3)
    parser.add_argument("--skip-recovery-execution", action="store_true",
                         help="Only build+write the recovery tool-call chain and schedule JSON(s); don't actually run the tools.")
    parser.add_argument("--manifest-file", type=Path, default=None, help="Default: <output-dir>/manifest.json")
    parser.add_argument("--disturbed-subset-file", type=Path, default=None, help="Default: <output-dir>/disturbed_subset.json")
    parser.add_argument("--recovered-subset-file", type=Path, default=None, help="Default: <tool-results-dir>/recovered_subset.json")
    parser.add_argument("--schedule-dir", type=Path, default=None, help="Default: <tool-results-dir>/tool_schedules")
    args = parser.parse_args()

    args.output_dir = args.output_dir.expanduser().resolve()
    args.tool_results_dir = args.tool_results_dir.expanduser().resolve()

    manifest_file = args.manifest_file or (args.output_dir / "manifest.json")
    disturbed_subset_file = args.disturbed_subset_file or (args.output_dir / "disturbed_subset.json")
    recovered_subset_file = args.recovered_subset_file or (args.tool_results_dir / "recovered_subset.json")
    schedule_dir = args.schedule_dir or (args.tool_results_dir / "tool_schedules")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tool_results_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    qa_pairs = load_qa_pairs(args.input_json)
    print(f"Loaded {len(qa_pairs)} QA pairs from {args.input_json}", file=sys.stderr)

    disturb_work_dir = args.output_dir / "disturbed"
    records: List[Dict[str, Any]] = []
    for idx, item in enumerate(qa_pairs, start=1):
        item.setdefault("id", f"sample_{uuid.uuid4().hex[:12]}")
        for attempt in range(args.max_attempts_per_sample):
            try:
                record = build_one_sample(item, disturb_work_dir, args.min_ops, args.max_ops, rng)
                records.append(record)
                ops = [s["op"] for s in record["disturb_chain"]]
                print(f"[{idx}/{len(qa_pairs)}] {item['id']}: disturbed with {ops}", file=sys.stderr)
                break
            except Exception:
                print(f"Attempt {attempt + 1} failed for {item['id']}:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
        else:
            print(f"Giving up on {item['id']} after {args.max_attempts_per_sample} attempts.", file=sys.stderr)

    execute_recovery_turns(
        records,
        schedule_dir=schedule_dir,
        tool_results_dir=args.tool_results_dir,
        skip_execution=args.skip_recovery_execution,
    )

    qa_by_id = {item["id"]: item for item in qa_pairs}

    with manifest_file.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)
    print(f"Wrote manifest for {len(records)} samples to {manifest_file}", file=sys.stderr)

    disturbed_subset = [
        to_dcase_entry(qa_by_id[rec["id"]], rec["disturbed_audio"])
        for rec in records
    ]
    with disturbed_subset_file.open("w", encoding="utf-8") as handle:
        json.dump(disturbed_subset, handle, indent=2, ensure_ascii=False)
    print(f"Wrote disturbed subset ({len(disturbed_subset)} items) to {disturbed_subset_file}", file=sys.stderr)

    recovered_subset = []
    skipped = 0
    for rec in records:
        if rec.get("failed") or not rec["recovery_chain"]:
            # No recovery steps needed (k==0) -> recovered == disturbed == source;
            # a genuine failure means current_audio_path stopped advancing partway,
            # which is still the best audio we could produce, so include it either way.
            skipped += 1 if rec.get("failed") else 0
        recovered_subset.append(to_dcase_entry(qa_by_id[rec["id"]], rec["current_audio_path"]))
    with recovered_subset_file.open("w", encoding="utf-8") as handle:
        json.dump(recovered_subset, handle, indent=2, ensure_ascii=False)
    print(
        f"Wrote recovered subset ({len(recovered_subset)} items, {skipped} incomplete due to tool failures) "
        f"to {recovered_subset_file}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
