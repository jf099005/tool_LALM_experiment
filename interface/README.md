# Tool-calling interface for the LALM

Lets a LALM (fine-tuned checkpoint or a raw official model, zero-shot) drive
its own audio-editing tool-call chain: given one or more input audios and a
natural-language instruction, the model emits one tool call per turn, this
interface executes it for real against the real tools in [`tools/`](../tools),
and feeds the real result back as the next turn's input, until the model
signals it's done or a step budget is hit.

This generalizes the turn structure `tool_use_training/gen_1st_stage_data/build_dataset.py`
teaches during SFT (`to_swift_sample`) and `testing_tool_use_benchmark/run_eval.py`
drives for its benchmark — but for an arbitrary instruction over arbitrary
input audio, not only the fixed "reconstruct target B from source A" task the
benchmark measures, and with no notion of a ground-truth chain to score
against. If you want tool-calling *accuracy metrics* against known
source/target pairs, use `testing_tool_use_benchmark/` instead; use this
folder to actually run the model on real audio for a real task.

## Files

- `protocol.py` — the JSON tool-call convention (one
  `{"tool_name", "parameters", "output_audio_id"}` object per assistant
  turn, `{"done": true}` to close the chain) and the system prompt, built
  from the real tool schemas via `tools.generate_tool_descriptions` (not the
  synthetic, randomly-parameterized subset `tools/synthetic_registry.py`
  exposes for training-data generation).
- `executor.py` — runs one predicted `(tool_name, parameters)` call for real
  against `tools.TOOL_NAME_TO_CLASS`, and extracts whatever new audio (if
  any) the tool produced (`output_path` / `clip_path` / `separated_files`).
  Tools like `asr` that produce no new audio, only text, are handled too.
- `engine.py` — two interchangeable backends behind one
  `generate_turn(messages, audios) -> str` interface: `SwiftEngine` (ms-swift's
  `TransformersEngine`, for a fine-tuned LoRA checkpoint *or* any raw
  official model id ms-swift recognizes) and `VLLMEngine` (plain vLLM, no
  ms-swift dependency — fastest way to run an unmodified official checkpoint).
- `agent.py` — `ToolCallingAgent`, which drives the multi-turn loop and
  returns an `AgentResult` (full transcript, every audio id produced along
  the way, the final audio, and why the chain stopped).
- `cli.py` — a command-line driver: point it at a model, one or more audio
  files, and an instruction; see `python -m interface.cli --help`.

## Usage

All of this runs under the `ms-swift` conda env
(`~/miniconda3/envs/ms-swift`) for the `swift` backend, since it already has
librosa/soundfile/numpy/scipy alongside ms-swift itself. Heavy ML tools
(`human_voice_enhance`, `super_resolution`, `extract_target`, `remove_target`,
`source_separation`) need their own conda envs' dependencies importable to
actually run (see each tool module's top-level try/except) — if unavailable,
a call to one of them fails at execution time with a clear error rather than
failing at import time, so the rest of the tool catalogue still works.

```bash
SWIFT_PY=~/miniconda3/envs/ms-swift/bin/python

# Fine-tuned checkpoint
$SWIFT_PY -m interface.cli \
    --model Qwen/Qwen2.5-Omni-7B \
    --adapter-dir output/v10-20260625-215142/checkpoint-500 \
    --audio /path/to/input.wav \
    --instruction "Remove the background noise and normalize the loudness." \
    --work-dir /work/u1501463/interface_runs/demo \
    --output-file /work/u1501463/interface_runs/demo/result.json

# Raw official model, zero-shot, fast vLLM path
VLLM_PY=~/miniconda3/envs/vllm_UQ/bin/python
$VLLM_PY -m interface.cli \
    --backend vllm --model Qwen/Qwen2.5-Omni-7B \
    --audio /path/to/input.wav \
    --instruction "Extract the vocals from this clip." \
    --work-dir /work/u1501463/interface_runs/demo2
```

Or drive it programmatically:

```python
from interface import ToolCallingAgent
from interface.engine import SwiftEngine

engine = SwiftEngine(model="Qwen/Qwen2.5-Omni-7B", adapter_dir="output/checkpoint-500")
agent = ToolCallingAgent(engine)
result = agent.run(
    instruction="Remove the background noise and normalize the loudness.",
    audio_paths=["/path/to/input.wav"],
    work_dir="/work/u1501463/interface_runs/demo",
)
print(result.stop_reason, result.final_audio_path)
```

## Protocol notes

- **`audio_id`** is a purely textual convention (never a real file path) the
  model uses to refer to an audio it's already seen — an input audio
  (`audio_0`, `audio_1`, ...) or an `output_audio_id` it declared in an
  earlier turn. If a call omits `audio_id`, the agent defaults it to
  whichever audio the previous turn most recently produced.
- **Multi-output tools.** `source_separation` (and `extract_target`/`remove_target`
  when run through the base `SourceSeparationTool` path) can return more than
  one new audio (`target`, and `residual` if requested). The agent registers
  the `target` stem under the call's `output_audio_id` and, if present, the
  `residual` stem under `{output_audio_id}_residual`.
- **Text-only tools.** `asr` produces no new audio — its result (the
  transcript) is fed back to the model as plain text instead of an audio tag,
  and the "current audio" pointer is left unchanged.
- **Stopping.** A run ends when: the model's output isn't parseable JSON
  (`unparseable_output`), it emits `{"done": true}` (`model_signaled_done`),
  it repeats an identical call (`repeated_call`), it references an unknown
  `audio_id` (`unknown_audio_id`), a predicted tool call fails to execute
  (`tool_execution_failed`), or `--max-steps` is exhausted
  (`max_steps_reached`). `AgentResult.stop_reason` / `result["stop_reason"]`
  in the CLI's JSON output reports which.
- **System prompt.** By default `ToolCallingAgent` builds one from
  `protocol.build_system_prompt()`, spelling out the JSON-per-turn convention
  and the tool catalogue — needed for a zero-shot/official model, which has
  never seen this convention. A checkpoint fine-tuned on
  `tool_use_training/gen_1st_stage_data`'s SFT data was trained with **no**
  system turn at all, so pass `system_prompt=""` (or the CLI's
  `--no-system-prompt`) to match that.
