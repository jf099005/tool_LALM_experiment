"""Build a synthetic tool-use dataset for training LALMs to infer audio-editing
tool-call chains.

For each sample: pick a source audio A, draw k in [min_tools, max_tools] distinct
tools from the available registry, apply them to A in sequence to produce a
target audio B, then emit a (Question, Answer) pair where the Answer is a pure
tool-call JSON trace (no natural language) that reproduces B from A.

Each audio involved is given a unique audio_id (audio_0 = A, audio_1 = B, then
audio_2, audio_3, ... for every tool-call output, final step included, in the
order they're produced) -- see `audio_id_map` on each entry. Tool calls
reference their input audio by `parameters["audio_id"]` rather than a literal
path, and declare an `output_audio_id` for their own output -- an id that
doesn't exist yet at call time. The final step's output id is a fresh one too
(never audio_1): a model can't know in advance that a call's output will
exactly equal the pre-revealed target, so the chain's end is signalled purely
by the explicit trailing `{"done": true}` turn, not by an id choice.

Usage:
    python build_dataset.py --num-samples 200 --output-dir /work/u1501463/gen_tool_usage_QA

See tools/synthetic_registry.py for which tools are active -- it depends on what's
importable in the current interpreter (run under the project's `ms-swift` env
for librosa/soundfile-backed tools, or `deepfilternet`/`audiosr`/`sam_audio`
to unlock those specific heavy tools).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional

# Each worker process handles one sample at a time on a small (few-second)
# clip -- BLAS/OpenMP intra-op threading buys nothing there and just causes
# oversubscription once samples are parallelized across processes. Must be
# set (via setdefault, so an explicit env config still wins) before numpy
# ever gets imported -- e.g. transitively through tool_registry below --
# since these libraries fix their thread count at first import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tools import synthetic_registry as tool_registry  # noqa: E402
from question_templates import render_question  # noqa: E402
from official_system_prompt import compose_system_prompt  # noqa: E402
from interface import protocol as agent_protocol  # noqa: E402

DEFAULT_AUDIOSET_DIR = Path("/work/u1501463/audioset_20k/20k/train")
DEFAULT_VCTK_DIR = Path("/work/u1501463/VCTK/wav48_silence_trimmed")


def load_excluded_stems(exclude_path: Optional[Path]) -> set[str]:
    """Collect the source-audio stems used by a previously generated dataset.

    `source_audio` in a dataset JSON is either the original file untouched
    (audioset, already .wav) copied as `A_<name>.wav`, or a from-scratch WAV
    conversion of the original (vctk, from .flac) that keeps the original
    stem. Stripping the `A_` prefix when present lets both cases match back
    to the raw file's stem in `collect_source_files`.
    """
    if exclude_path is None:
        return set()
    with open(exclude_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    stems: set[str] = set()
    for entry in data:
        source_audio = entry.get("source_audio")
        if not source_audio:
            continue
        stem = Path(source_audio).stem
        stems.add(stem)
        if stem.startswith("A_"):
            stems.add(stem[len("A_"):])
    return stems


def collect_source_files(
    sources: List[str],
    limit: int | None = None,
    excluded_stems: set[str] | None = None,
) -> List[Path]:
    files: List[Path] = []
    if "audioset" in sources:
        files.extend(sorted(DEFAULT_AUDIOSET_DIR.glob("*.wav")))
    if "vctk" in sources:
        files.extend(sorted(DEFAULT_VCTK_DIR.glob("*/*.flac")))
    if excluded_stems:
        files = [f for f in files if f.stem not in excluded_stems]
    if limit:
        files = files[:limit]
    return files


def assign_audio_ids(num_steps: int) -> tuple[List[str], List[str]]:
    """Compute (input_id, output_id) per step for a chain of `num_steps` tool calls.

    Ids mirror the order audios are introduced to the model: audio_0 is the
    source, audio_1 is the target. Every step's output -- including the final
    one -- mints a fresh id (audio_2, audio_3, ...) that doesn't exist until
    that step runs. The final step's real output happens to be byte-identical
    to the target, but it keeps its own fresh id rather than aliasing
    audio_1: at generation time the model has no way to know a given call
    will exactly reproduce the target before the tool actually runs it, so
    the chain's end is signalled only by the explicit trailing
    `{"done": true}` turn.
    """
    input_ids: List[str] = []
    output_ids: List[str] = []
    for step_index in range(1, num_steps + 1):
        input_ids.append("audio_0" if step_index == 1 else output_ids[-1])
        output_ids.append(f"audio_{step_index + 1}")
    return input_ids, output_ids


def to_swift_sample(
    entry: Dict[str, Any],
    audio_token: str = "<audio>",
    base_system_prompt: str | None = None,
    system_prompt_model_dir: str | None = None,
    system_prompt_model_type: str | None = None,
) -> Dict[str, Any]:
    """Convert one dataset entry into an ms-swift SFT row.

    ms-swift's multimodal custom-dataset format expects a `messages` list plus a
    parallel `audios` list; each `audio_token` occurrence in the user content is
    bound, in order, to the corresponding path in `audios`. Every occurrence is
    tagged with the audio_id it represents (e.g. `<audio_0><audio>`) so the
    model has an explicit handle to refer back to that audio by id in a later
    tool call, instead of relying on positional ordering alone. The literal
    source/target paths embedded in the human-readable `question` text are
    swapped for a tagged `audio_token` here, since the model perceives the
    audio itself, not its path.

    Every row gets an explicit system turn: the target model's own official
    default system message (auto-detected from `system_prompt_model_dir`, or
    overridden via `base_system_prompt`) followed by the tool catalogue --
    see `official_system_prompt.compose_system_prompt`. The tool catalogue
    used to be repeated inline in the user question instead; it now lives
    only in the system turn, so `question_templates.py` no longer renders it.

    The answer is rendered as a multi-turn agent/tool dialogue: one assistant
    turn per tool call, each followed by a "tool" turn carrying that step's
    real output audio (tagged with the id the tool call itself declared via
    `output_audio_id`) for the next call to reference -- this is the same
    turn structure `tool_use_benchmark/run_eval.py` drives at inference time.
    A final explicit `{"done": true}` assistant turn closes the chain, so the
    model has a learned stop signal instead of relying on running out of
    turns.

    The turn JSON/text itself is rendered via `interface.protocol`'s shared
    helpers (`render_tool_call_json` / `render_tool_result_message` /
    `render_done_json`), not reimplemented here -- `interface/agent.py` (live
    inference) and `testing_tool_use_benchmark/run_eval.py` (eval) render the
    exact same wording from the same functions, so a trained model never sees
    an out-of-distribution turn format at inference/eval time.
    """
    audio_id_map = entry["audio_id_map"]
    audios: List[str] = []

    def tag(audio_id: str) -> str:
        # ms-swift binds each `audio_token` occurrence in the text, in order, to
        # one entry in `audios` -- it does not dedupe by audio_id. So every
        # occurrence must append a path, even if the same id were ever tagged
        # more than once.
        audios.append(audio_id_map[audio_id])
        return f"<{audio_id}>{audio_token}"

    question_text = entry["question"].replace(entry["source_audio"], tag("audio_0"), 1)
    question_text = question_text.replace(entry["target_audio"], tag("audio_1"), 1)

    system_prompt = compose_system_prompt(
        tools_block=entry["tools_block"],
        base_system_prompt=base_system_prompt,
        model_dir=system_prompt_model_dir,
        model_type=system_prompt_model_type,
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question_text}]

    tool_calls = entry["answer"]["tool_calls"]
    for step_idx, tool_call in enumerate(tool_calls, start=1):
        messages.append({
            "role": "assistant",
            "content": agent_protocol.render_tool_call_json(
                tool_call["tool_name"], tool_call["parameters"], tool_call["output_audio_id"]
            ),
        })
        output_id = tool_call["output_audio_id"]
        messages.append({
            "role": "tool",
            "content": agent_protocol.render_tool_result_message(
                step_idx, tool_call["tool_name"], [tag(output_id)]
            ),
        })
    messages.append({"role": "assistant", "content": agent_protocol.render_done_json()})

    return {
        "messages": messages,
        "audios": audios,
    }


def build_one_sample(
    sample_id: str,
    source_file: Path,
    work_dir: Path,
    tool_names: List[str],
    min_tools: int,
    max_tools: int,
    rng: random.Random,
) -> Dict[str, Any]:
    sample_dir = work_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    audio_a = tool_registry.ensure_wav(source_file, sample_dir)
    if audio_a.parent != sample_dir:
        import shutil
        copied = sample_dir / f"A_{audio_a.name}"
        shutil.copy(str(audio_a), str(copied))
        audio_a = copied

    k = rng.randint(min_tools, max_tools)
    k = min(k, len(tool_names))
    chosen_tools = rng.sample(tool_names, k)
    input_ids, output_ids = assign_audio_ids(len(chosen_tools))

    current_path = audio_a
    tool_calls: List[Dict[str, Any]] = []
    step_outputs: List[Path] = []

    for step_index, name in enumerate(chosen_tools, start=1):
        duration = tool_registry.get_duration_seconds(current_path)
        out_path = sample_dir / f"step{step_index}_{name}.wav"
        params, current_path = tool_registry.REGISTRY[name].apply(current_path, out_path, rng, duration)
        params = dict(params)
        params.pop("audio_path", None)
        params["audio_id"] = input_ids[step_index - 1]
        tool_calls.append({
            "tool_name": name,
            "parameters": params,
            "output_audio_id": output_ids[step_index - 1],
        })
        step_outputs.append(current_path)

    audio_b = sample_dir / "B.wav"
    if current_path != audio_b:
        current_path.replace(audio_b)
        step_outputs[-1] = audio_b

    audio_id_map = {"audio_0": str(audio_a), "audio_1": str(audio_b)}
    for step_index in range(1, len(chosen_tools) + 1):
        # The final step's fresh id maps to the same file as audio_1 (its
        # output *is* the target) -- two ids for one file, not a conflict.
        audio_id_map[output_ids[step_index - 1]] = str(step_outputs[step_index - 1])

    tools_block = tool_registry.describe_available_tools()
    question = render_question(source=str(audio_a), target=str(audio_b), rng=rng)

    return {
        "id": sample_id,
        "source_audio": str(audio_a),
        "target_audio": str(audio_b),
        "available_tools": tool_registry.available_tool_names(),
        "tools_block": tools_block,
        "num_steps": len(chosen_tools),
        "question": question,
        "answer": {"tool_calls": tool_calls},
        "step_outputs": [str(p) for p in step_outputs],
        "audio_id_map": audio_id_map,
    }


def _build_sample_task(
    task_index: int,
    seed: int,
    source_files: List[Path],
    work_dir: Path,
    tool_names: List[str],
    min_tools: int,
    max_tools: int,
    max_attempts: int,
) -> Optional[Dict[str, Any]]:
    """Run in a worker process: build one sample, retrying with fresh draws.

    `task_index` seeds a private `random.Random` so results stay reproducible
    for a given `seed` regardless of how many workers run or in what order
    they finish (unlike a single shared `random.Random` mutated across
    threads, which can't be replayed deterministically).
    """
    rng = random.Random(seed + task_index)
    sample_id = f"sample_{uuid.uuid4().hex[:12]}"
    source_file = rng.choice(source_files)

    for attempt in range(max_attempts):
        try:
            return build_one_sample(
                sample_id=sample_id,
                source_file=source_file,
                work_dir=work_dir,
                tool_names=tool_names,
                min_tools=min_tools,
                max_tools=max_tools,
                rng=rng,
            )
        except Exception:
            print(f"Attempt {attempt + 1} failed for {sample_id} ({source_file}):", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            source_file = rng.choice(source_files)

    print(f"Giving up on {sample_id} after {max_attempts} attempts.", file=sys.stderr)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic audio tool-use QA dataset.")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--min-tools", type=int, default=1)
    parser.add_argument("--max-tools", type=int, default=4)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["audioset", "vctk"], 
        default=["audioset"],
        help="Which source datasets to draw audio A from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/work/u1501463/gen_tool_usage_QA"),
        help="Directory to write generated audio (A/B + intermediates) into.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path(__file__).resolve().parent / "tool_usage_qa.json",
        help="Path to write the resulting JSON dataset.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help=(
            "Path to an existing dataset JSON (e.g. tool_usage_qa.json) whose "
            "source audio files must not be drawn from when generating new "
            "samples. Pass 'None' or omit to disable."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts-per-sample", type=int, default=5)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of worker processes to build samples in parallel. "
            "1 = sequential (original behavior). Each active tool here is "
            "CPU-only, so this scales with core count -- avoid setting it "
            "above os.cpu_count()."
        ),
    )
    parser.add_argument(
        "--swift-output-file",
        type=Path,
        default=None,
        help=(
            "Optional path to also write an ms-swift SFT-ready JSONL file "
            "(messages + audios, audio_token in place of literal paths)."
        ),
    )
    parser.add_argument(
        "--audio-token",
        type=str,
        default="<audio>",
        help="Placeholder token substituted for each audio path in the swift user message.",
    )
    parser.add_argument(
        "--base-system-prompt",
        type=str,
        default=None,
        help=(
            "Explicit override for the system prompt's base greeting (the part before the "
            "tool catalogue). If omitted, it's auto-detected from --system-prompt-model-dir "
            "(or --system-prompt-model-type as a fallback) -- see official_system_prompt.py."
        ),
    )
    parser.add_argument(
        "--system-prompt-model-dir",
        type=str,
        default=None,
        help=(
            "Path to the target model's own directory (the one it'll actually be trained "
            "from), used to auto-detect its official default system message from "
            "chat_template.json/tokenizer_config.json. Swap this when training a different "
            "model so the system prompt matches that model's own convention instead of "
            "assuming Qwen's."
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

    exclude_path = None
    if args.exclude and args.exclude.lower() != "none":
        exclude_path = Path(args.exclude)
    excluded_stems = load_excluded_stems(exclude_path)
    if excluded_stems:
        print(f"Excluding {len(excluded_stems)} source audio stem(s) from {exclude_path}", file=sys.stderr)

    source_files = collect_source_files(args.sources, excluded_stems=excluded_stems)
    if not source_files:
        raise SystemExit(f"No source audio files found for sources={args.sources}")

    tool_names = tool_registry.available_tool_names()
    if not tool_names:
        raise SystemExit("No tools are available in the current environment -- check tool_registry dependencies.")
    print(f"Available tools ({len(tool_names)}): {tool_names}", file=sys.stderr)

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[int, Dict[str, Any]] = {}

    if args.workers <= 1:
        # Sequential path: same behavior as before, minus the process-pool
        # bookkeeping. Kept separate rather than routed through the pool so
        # --workers 1 has zero parallel-machinery overhead.
        next_index = 0
        while len(results) < args.num_samples:
            entry = _build_sample_task(
                task_index=next_index,
                seed=args.seed,
                source_files=source_files,
                work_dir=args.output_dir,
                tool_names=tool_names,
                min_tools=args.min_tools,
                max_tools=args.max_tools,
                max_attempts=args.max_attempts_per_sample,
            )
            next_index += 1
            if entry is not None:
                results[len(results)] = entry
                print(
                    f"[{len(results)}/{args.num_samples}] {entry['id']}: {entry['num_steps']} steps",
                    file=sys.stderr,
                )
    else:
        # Each sample is independent (own sample_dir, own private rng seeded
        # from task_index), so samples parallelize cleanly across processes.
        # Tasks that exhaust their retries are replaced by a freshly-seeded
        # task rather than just dropped, so the run still ends with exactly
        # `num_samples` entries -- matching the original sequential behavior.
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            next_index = 0
            pending: Dict[Any, int] = {}
            for _ in range(args.num_samples):
                future = executor.submit(
                    _build_sample_task,
                    next_index,
                    args.seed,
                    source_files,
                    args.output_dir,
                    tool_names,
                    args.min_tools,
                    args.max_tools,
                    args.max_attempts_per_sample,
                )
                pending[future] = next_index
                next_index += 1

            while len(results) < args.num_samples:
                done, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    entry = future.result()
                    if entry is not None:
                        results[len(results)] = entry
                        print(
                            f"[{len(results)}/{args.num_samples}] {entry['id']}: {entry['num_steps']} steps",
                            file=sys.stderr,
                        )
                    else:
                        replacement = executor.submit(
                            _build_sample_task,
                            next_index,
                            args.seed,
                            source_files,
                            args.output_dir,
                            tool_names,
                            args.min_tools,
                            args.max_tools,
                            args.max_attempts_per_sample,
                        )
                        pending[replacement] = next_index
                        next_index += 1

    dataset: List[Dict[str, Any]] = [results[i] for i in range(len(results))]

    with args.output_file.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, indent=2)

    print(f"Wrote {len(dataset)} samples to {args.output_file}", file=sys.stderr)

    if args.swift_output_file:
        args.swift_output_file.parent.mkdir(parents=True, exist_ok=True)
        with args.swift_output_file.open("w", encoding="utf-8") as handle:
            for entry in dataset:
                swift_sample = to_swift_sample(
                    entry,
                    audio_token=args.audio_token,
                    base_system_prompt=args.base_system_prompt,
                    system_prompt_model_dir=args.system_prompt_model_dir,
                    system_prompt_model_type=args.system_prompt_model_type,
                )
                handle.write(json.dumps(swift_sample, ensure_ascii=False) + "\n")
        print(f"Wrote {len(dataset)} ms-swift SFT rows to {args.swift_output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
