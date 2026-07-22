# Tool-use benchmark for trained LALMs

Evaluates a model trained with the `tool_use_training/gen_1st_stage_data` recipe
(`build_dataset.py`) on the task it was trained for: given source audio A and
target audio B, predict the tool-call chain that turns A into B. This folder
is self-contained and does not modify anything under `generate/` or `tools/`.

Two things get measured per sample:

1. **Tool-calling ability** — for each turn the model emits a tool call,
   actually execute it against the real audio. Reports how many predicted
   calls parsed + executed successfully, and how the predicted tool-name
   sequence compares to ground truth (exact-match, precision/recall/F1).
2. **Audio fidelity** — the audio the model's own tool chain actually
   produces is compared against the ground-truth target B (log-mel cosine,
   MFCC cosine, duration ratio, etc; see `audio_metrics.compare_audio`).

## Files

- `build_benchmark.py` — generates held-out (A, B, ground-truth tool chain)
  triples, reusing `tools/synthetic_registry.py` and `question_templates.py`
  so the benchmark matches the training distribution. Use a different `--seed`
  (and optionally `--held-out-fraction`) from whatever generated `train.jsonl`
  so samples don't leak into eval.
- `audio_metrics.py` — length-robust audio similarity metrics.
- `run_eval.py` — drives the multi-turn loop (model emits a tool call ->
  executor runs it -> result audio fed back as the next turn's input, mirroring
  `build_dataset.to_swift_sample`'s turn structure) and writes `results.json`.
  All of the actual tool/LALM interaction plumbing is shared with
  [`interface/`](../interface): the model backends (`interface.engine.SwiftEngine`
  / `VLLMEngine`), the per-turn JSON parsing (`interface.protocol.parse_turn`),
  and predicted tool-call execution (`interface.executor.run_tool_call`).
  This folder only keeps what's specific to *scoring* against a known
  source/target pair: the fixed A/B prompt framing, `DEFAULT_PROTOCOL_SYSTEM_PROMPT`,
  the step-budget-from-ground-truth-length logic, and the precision/recall/F1
  and audio-fidelity metrics below.

## Usage

All steps run under the `ms-swift` conda env (`~/miniconda3/envs/ms-swift`),
which already has librosa/soundfile/numpy/scipy alongside ms-swift itself.
Heavy ML tools (human_voice_enhance, super_resolution, extract/remove_target)
need their own envs (`deepfilternet`/`audiosr`/`sam_audio`) to be *registered*
at generation time — if you want them in the benchmark, build it from that env
instead; `run_eval.py`'s execution side will pick up whatever's importable.

```bash
SWIFT_PY=~/miniconda3/envs/ms-swift/bin/python

# 1. Build a held-out benchmark (use a seed disjoint from train.jsonl's generation run)
$SWIFT_PY build_benchmark.py \
    --num-samples 100 --seed 4242 \
    --output-dir /work/u1501463/tool_use_benchmark_audio \
    --output-file benchmark.json

# 2a. Evaluate the fine-tuned checkpoint (once ms-swift training has produced one)
$SWIFT_PY run_eval.py \
    --benchmark-file benchmark.json \
    --model Qwen/Qwen2.5-Omni-7B \
    --adapter-dir ../output/v10-20260625-215142/checkpoint-500 \
    --work-dir /work/u1501463/tool_use_benchmark_predictions \
    --output-file results_finetuned.json

# 2b. Evaluate the raw official Qwen2.5-Omni-7B (zero-shot, no adapter) -- fast path via vLLM
VLLM_PY=~/miniconda3/envs/vllm_UQ/bin/python
$VLLM_PY run_eval.py \
    --benchmark-file benchmark.json \
    --backend vllm \
    --model Qwen/Qwen2.5-Omni-7B \
    --work-dir /work/u1501463/tool_use_benchmark_predictions_baseline \
    --output-file results_baseline.json

# 2c. Evaluate some other official model ms-swift recognizes (no fine-tuning, no adapter)
$SWIFT_PY run_eval.py \
    --benchmark-file benchmark.json \
    --model Qwen/Qwen2-Audio-7B-Instruct \
    --model-type qwen2_audio \
    --work-dir /work/u1501463/tool_use_benchmark_predictions_qwen2audio \
    --output-file results_qwen2audio.json
```

`results.json` contains a per-sample `results` list plus an aggregate
`summary` (overall tool-call success rate, mean exact-match/precision/recall/F1,
mean audio-closeness metrics, and a breakdown by ground-truth chain length).
Re-running prints the same `summary` dict to stdout.

### Backends and zero-shot vs. fine-tuned

- `--backend swift` (default) drives `swift.infer_engine.TransformersEngine`.
  Pass `--adapter-dir <checkpoint>` to evaluate the fine-tuned LoRA checkpoint,
  or omit it to evaluate the raw base model id in `--model` (any model
  ms-swift's template registry recognizes — pass `--model-type` explicitly if
  it can't auto-detect one, e.g. for a model family other than Qwen2.5-Omni).
- `--backend vllm` drives plain vLLM directly with no ms-swift/adapter
  dependency at all (run under e.g. `vllm_UQ` or `vllm_qwen3`) — the fastest
  way to benchmark an unmodified official checkpoint, mirroring the prompt
  construction `qwen25_with_tool_chain_evaluation.py` already uses elsewhere
  in this repo. It does not support `--adapter-dir`.
- **System prompt.** The SFT training data has no system turn at all, so when
  `--adapter-dir` is set, `run_eval.py` defaults to no system prompt (matches
  training). A zero-shot/official model has never seen the implicit
  "one-tool-call-per-turn JSON" convention the fine-tuned model learned from
  supervision, so by default it's run with `DEFAULT_PROTOCOL_SYSTEM_PROMPT`
  (in `run_eval.py`), which spells out the expected JSON-per-turn format and
  an explicit `{"done": true}` stop signal — the fine-tuned chain has no such
  signal and instead just stops emitting valid tool-call JSON. Override with
  `--system-prompt "..."` or force none with `--no-system-prompt`.

## How audio difference is measured

There is no trained "loss" here -- `audio_metrics.compare_audio(pred, target)`
is a set of fixed, evaluation-time similarity metrics computed once per
sample on the final audio the model's tool chain actually produced. Both
clips are loaded mono at 16kHz. Concretely:

1. **Log-mel spectrogram cosine similarity** (`log_mel_cosine`). Compute a
   64-band log-mel spectrogram for each clip (`n_fft=1024`, `hop_length=256`,
   then `librosa.power_to_db`). If the clips have different durations (many
   tools change length -- clipping, trim_silence, pad_noise, time_stretch...),
   the *shorter* clip's spectrogram is resampled along the time axis (FFT-based
   interpolation, `scipy.signal.resample`) up to the *longer* clip's frame
   count, so the two matrices end up the same shape `(64, T)` before
   comparison. Cosine similarity is then taken over the flattened matrices:

   ```
   cos(pred, target) = (pred · target) / (||pred|| * ||target|| + eps)
   ```

   This is the main "how similar do these two sounds look in frequency-time
   space" signal, ranging from -1 (opposite) to 1 (identical).
2. **Log-mel L1 distance** (`log_mel_l1`): mean absolute difference between
   the same two aligned log-mel matrices, in dB -- a magnitude-sensitive
   companion to the cosine similarity (which is magnitude-invariant).
3. **MFCC cosine similarity** (`mfcc_cosine`): identical cosine-similarity
   computation, but on 13-coefficient MFCCs derived from those log-mel
   spectrograms instead -- a coarser, more timbre-focused view (MFCCs discard
   most fine spectral detail and keep the broad spectral envelope).
4. **Loudness difference** (`rms_db_diff`): `20*log10(RMS(pred)) - 20*log10(RMS(target))`,
   i.e. how much louder/quieter the prediction is overall.
5. **SI-SDR** (`si_sdr_db`): scale-invariant signal-to-distortion ratio,
   computed directly on the raw (sample-aligned) waveforms --
   `10*log10(||α·target||² / ||pred - α·target||²)` where `α` is the
   least-squares scale that best aligns `pred` to `target`. Only computed when
   `duration_ratio` is within 2% of 1.0, since it has no length-robust form;
   `None` otherwise.
6. **Composite `closeness_score`**, what `run_eval.py`'s aggregate report
   averages into `summary.mean_audio_closeness_score`:

   ```
   closeness_score = 0.25*(log_mel_cosine + 1) + 0.25*(mfcc_cosine + 1)
   ```

   i.e. the two cosine similarities each rescaled from `[-1, 1]` to `[0, 1]`
   and averaged 50/50. 1.0 means the reconstructed audio is spectrally
   identical to the target on both measures; 0.0 means maximally dissimilar.

Worked example: if a model's predicted chain stops one tool call short of the
ground truth (e.g. it correctly applies `spectral_normalize` but skips the
trailing `denoise`), `compare_audio` is run between *that partial-chain
output* and the full ground-truth target B -- so `closeness_score` directly
reflects how much the missing step actually mattered acoustically, rather
than just penalizing the missing tool-call symbolically (that's what
`tool_name_recall` is for, see above).

## Design notes

- **Stopping condition.** `run_eval.py` stops a sample when: the model's
  output isn't parseable JSON (`unparseable_output`), it explicitly signals
  `{"done": true}` (`model_signaled_done` — only expected from a zero-shot
  model run with the default protocol system prompt), it repeats an identical
  call (`repeated_call`), a predicted tool call fails to execute
  (`tool_execution_failed`), or a step budget (`len(gt_chain) +
  --max-extra-steps`, capped by `--hard-step-cap`) is exhausted
  (`max_steps_reached`). `summary.stop_reason_counts` reports the mix.
- **On a failed tool call**, the chain is cut there rather than continuing
  with the un-transformed audio — the audio-fidelity metric then reflects how
  far the (partial, valid) chain actually got.
- **`parameters["audio_path"]`** in a predicted call is always overwritten
  with the real current audio path before execution; the placeholder strings
  (`<AUDIO_A>`, `<OUTPUT_OF_STEP_i>`) are a textual training convention, not
  something the tools themselves understand.
