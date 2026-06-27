"""Build a synthetic tool-use dataset for training LALMs to infer audio-editing
tool-call chains.

For each sample: pick a source audio A, draw k in [min_tools, max_tools] distinct
tools from the available registry, apply them to A in sequence to produce a
target audio B, then emit a (Question, Answer) pair where the Answer is a pure
tool-call JSON trace (no natural language) that reproduces B from A.

Usage:
    python build_dataset.py --num-samples 200 --output-dir /work/u1501463/gen_tool_usage_QA

See tool_registry.py for which tools are active -- it depends on what's
importable in the current interpreter (run under the project's `ms-swift` env
for librosa/soundfile-backed tools, or `deepfilternet`/`audiosr`/`sam_audio`
to unlock those specific heavy tools).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generate.gen_tool_usage_QA import tool_registry  # noqa: E402
from generate.gen_tool_usage_QA.question_templates import render_question  # noqa: E402

DEFAULT_AUDIOSET_DIR = Path("/work/u1501463/audioset_20k/20k/train")
DEFAULT_VCTK_DIR = Path("/work/u1501463/VCTK/wav48_silence_trimmed")


def collect_source_files(sources: List[str], limit: int | None = None) -> List[Path]:
    files: List[Path] = []
    if "audioset" in sources:
        files.extend(sorted(DEFAULT_AUDIOSET_DIR.glob("*.wav")))
    if "vctk" in sources:
        files.extend(sorted(DEFAULT_VCTK_DIR.glob("*/*.flac")))
    if limit:
        files = files[:limit]
    return files


def placeholder_for_step(step_index: int) -> str:
    """step_index is 1-based; step 1 reads the source audio A."""
    return "<AUDIO_A>" if step_index == 1 else f"<OUTPUT_OF_STEP_{step_index - 1}>"


def to_swift_sample(
    entry: Dict[str, Any],
    audio_token: str = "<audio>",
    system_prompt: str | None = None,
) -> Dict[str, Any]:
    """Convert one dataset entry into an ms-swift SFT row.

    ms-swift's multimodal custom-dataset format expects a `messages` list plus a
    parallel `audios` list; each `audio_token` occurrence in the user content is
    bound, in order, to the corresponding path in `audios`. The literal source/
    target paths embedded in the human-readable `question` text are swapped for
    `audio_token` here, since the model perceives the audio itself, not its path.
    """
    question_text = entry["question"].replace(entry["source_audio"], audio_token, 1)
    question_text = question_text.replace(entry["target_audio"], audio_token, 1)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question_text})
    messages.append({"role": "assistant", "content": json.dumps(entry["answer"], ensure_ascii=False)})

    return {
        "messages": messages,
        "audios": [entry["source_audio"], entry["target_audio"]],
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

    current_path = audio_a
    tool_calls: List[Dict[str, Any]] = []

    for step_index, name in enumerate(chosen_tools, start=1):
        duration = tool_registry.get_duration_seconds(current_path)
        out_path = sample_dir / f"step{step_index}_{name}.wav"
        params, current_path = tool_registry.REGISTRY[name].apply(current_path, out_path, rng, duration)
        params = dict(params)
        params["audio_path"] = placeholder_for_step(step_index)
        tool_calls.append({"tool_name": name, "parameters": params})

    audio_b = sample_dir / "B.wav"
    if current_path != audio_b:
        current_path.replace(audio_b)

    tools_block = tool_registry.describe_available_tools()
    question = render_question(source=str(audio_a), target=str(audio_b), tools_block=tools_block, rng=rng)

    return {
        "id": sample_id,
        "source_audio": str(audio_a),
        "target_audio": str(audio_b),
        "available_tools": tool_registry.available_tool_names(),
        "num_steps": len(chosen_tools),
        "question": question,
        "answer": {"tool_calls": tool_calls},
    }


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts-per-sample", type=int, default=5)
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
        "--swift-system-prompt",
        type=str,
        default=None,
        help="Optional system message to prepend to every swift SFT row.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    source_files = collect_source_files(args.sources)
    if not source_files:
        raise SystemExit(f"No source audio files found for sources={args.sources}")

    tool_names = tool_registry.available_tool_names()
    if not tool_names:
        raise SystemExit("No tools are available in the current environment -- check tool_registry dependencies.")
    print(f"Available tools ({len(tool_names)}): {tool_names}", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset: List[Dict[str, Any]] = []
    while len(dataset) < args.num_samples:
        source_file = rng.choice(source_files)
        sample_id = f"sample_{uuid.uuid4().hex[:12]}"

        for attempt in range(args.max_attempts_per_sample):
            try:
                entry = build_one_sample(
                    sample_id=sample_id,
                    source_file=source_file,
                    work_dir=args.output_dir,
                    tool_names=tool_names,
                    min_tools=args.min_tools,
                    max_tools=args.max_tools,
                    rng=rng,
                )
                dataset.append(entry)
                print(f"[{len(dataset)}/{args.num_samples}] {sample_id}: {entry['num_steps']} steps", file=sys.stderr)
                break
            except Exception:
                print(f"Attempt {attempt + 1} failed for {sample_id} ({source_file}):", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                source_file = rng.choice(source_files)
        else:
            print(f"Giving up on {sample_id} after {args.max_attempts_per_sample} attempts.", file=sys.stderr)

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
                    system_prompt=args.swift_system_prompt,
                )
                handle.write(json.dumps(swift_sample, ensure_ascii=False) + "\n")
        print(f"Wrote {len(dataset)} ms-swift SFT rows to {args.swift_output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
