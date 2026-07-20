"""Command-line entry point: point a LALM at one or more audio files plus a
natural-language instruction, and let it drive its own tool-call chain.

Usage (fine-tuned checkpoint, ms-swift env):
    ~/miniconda3/envs/ms-swift/bin/python -m interface.cli \
        --model Qwen/Qwen2.5-Omni-7B \
        --adapter-dir output/v10-20260625-215142/checkpoint-500 \
        --audio /path/to/input.wav \
        --instruction "Remove the background noise and normalize the loudness." \
        --work-dir /work/u1501463/interface_runs/demo \
        --output-file /work/u1501463/interface_runs/demo/result.json

Usage (raw official model, zero-shot, vLLM backend):
    ~/miniconda3/envs/vllm_UQ/bin/python -m interface.cli \
        --backend vllm --model Qwen/Qwen2.5-Omni-7B \
        --audio /path/to/input.wav \
        --instruction "Extract the vocals from this clip." \
        --work-dir /work/u1501463/interface_runs/demo2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import ToolCallingAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a LALM through the tool-calling interface.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Omni-7B", help="Base model id/path.")
    parser.add_argument("--adapter-dir", type=str, default=None, help="LoRA checkpoint dir from ms-swift SFT.")
    parser.add_argument(
        "--backend", choices=["swift", "vllm"], default="swift",
        help="'swift' = ms-swift TransformersEngine (required for --adapter-dir; also runs any raw official "
        "model ms-swift recognizes). 'vllm' = plain vLLM, fastest path for an unmodified official checkpoint, "
        "no adapter support.",
    )
    parser.add_argument(
        "--model-type", type=str, default=None,
        help="Swift backend only: explicit model_type if it can't be auto-detected from --model.",
    )
    parser.add_argument("--max-model-len", type=int, default=20000, help="vLLM backend only.")
    parser.add_argument("--max-num-seqs", type=int, default=8, help="vLLM backend only.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument(
        "--audio", action="append", required=True, dest="audio_paths",
        help="Path to an input audio file. Repeat for multiple inputs (audio_0, audio_1, ... in order given).",
    )
    parser.add_argument("--instruction", type=str, required=True, help="Natural-language task instruction.")
    parser.add_argument("--max-steps", type=int, default=8, help="Hard cap on tool-call turns.")
    parser.add_argument(
        "--system-prompt", type=str, default=None,
        help="Override the auto-generated tool-use protocol system prompt. Pass an empty string for no "
        "system turn at all (matches a fine-tuned checkpoint's SFT data, which had none).",
    )
    parser.add_argument(
        "--no-system-prompt", action="store_true",
        help="Shorthand for --system-prompt '' -- run with no system turn.",
    )

    parser.add_argument("--work-dir", type=Path, required=True, help="Directory to write step-output audio into.")
    parser.add_argument("--output-file", type=Path, default=None, help="Optional path to write the result JSON to.")
    args = parser.parse_args()

    if args.backend == "vllm" and args.adapter_dir:
        raise SystemExit("--adapter-dir requires --backend swift; the vLLM backend only runs raw official weights.")

    system_prompt = "" if args.no_system_prompt else args.system_prompt

    if args.backend == "swift":
        from .engine import SwiftEngine  # imported here so --help works without swift installed

        engine = SwiftEngine(
            model=args.model,
            adapter_dir=args.adapter_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            model_type=args.model_type,
        )
    else:
        from .engine import VLLMEngine  # imported here so --help works without vllm installed

        engine = VLLMEngine(
            model=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_audios_per_prompt=args.max_steps + len(args.audio_paths) + 1,
        )

    agent = ToolCallingAgent(engine, system_prompt=system_prompt)
    result = agent.run(
        instruction=args.instruction,
        audio_paths=args.audio_paths,
        work_dir=args.work_dir,
        max_steps=args.max_steps,
    )

    output = result.to_json()
    print(json.dumps(output, indent=2), file=sys.stderr)

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with args.output_file.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Wrote result to {args.output_file.resolve()}", file=sys.stderr)

    print(f"stop_reason={result.stop_reason} final_audio={result.final_audio_path}")


if __name__ == "__main__":
    main()
