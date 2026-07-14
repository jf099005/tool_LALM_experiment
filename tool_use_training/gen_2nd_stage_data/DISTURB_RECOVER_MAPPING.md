# Disturb -> Recover mapping (stage-2 tool-use data)

Stage 1 (`gen_1st_stage_data/`) teaches a model to reproduce an arbitrary target audio
B from a source A by calling tools -- it fits the tool-call *format* and general
capability. Stage 2 is narrower and tied directly to benchmark performance (MMAU,
DCASE AudioQA, ...): a QA pair's audio gets degraded by a data-augmentation op, and the
model must learn to call the tool(s) that clean it back up so the question is
answerable again. That's the same move a model should make at inference time whenever
benchmark audio is noisy, padded with junk, contaminated by a stray event, or has been
sped up / pitch-shifted.

`build_by_disturb.py` drives the pipeline; `disturb_recover.py` owns the table below
plus the functions that generate disturbance parameters and, from those, the matching
recovery parameters.

## The table

| Disturbance op (`audio_edit/`) | What it does | Recovery tool(s) (`tools/`) | Why |
|---|---|---|---|
| `add_noise` | Mixes white/pink noise across the whole clip at a target SNR (3-15 dB here) | `denoise` — `spectral_subtraction`, `wiener`, or `adaptive` | Broadband additive noise has no exact inverse (the noise itself isn't retained), so recovery is a *cleanup*, not an inversion. Three algorithm choices give the model genuinely different, plausible tool calls for the same disturbance instead of one memorizable answer. |
| `pad_noise` | Prepends/appends `duration_sec` of noise, quiet relative to the clip (10-25 dB down) | `clipping` (exact) or `trim_silence` (heuristic) | Padding only *extends* the clip; the original content is untouched and known-length, so `clipping` at the exact pad boundary is an exact inverse. `trim_silence` is a plausible alternative that works without knowing the exact pad length, since the pad is quiet by construction. |
| `insert_background_event` | Overlays a labeled real-world AudioSet event (Dog/Bird/Wind/Rain/Siren/Traffic noise/Cat/Vehicle) at a timestamp, mixed at 0-8 dB SNR | `remove_target` (always applicable) or `clipping` (only if the event sits close enough to one edge that cutting it off keeps >= 60% of the clip) | The event is *mixed in*, not appended, so there's no exact inverse; `remove_target` (source separation) is the general answer. `clipping` is a cheap alternative but only a valid recovery when the contaminated region is a small edge slice — otherwise it discards most of the original content, which is worse than not recovering at all. |
| `change_speed` | Time-stretches the clip by a fixed rate (0.8/0.9/1.1/1.25) | `time_stretch` with `rate = 1 / original_rate` | `librosa.effects.time_stretch` is parametric and (up to STFT resynthesis artifacts) invertible by re-stretching with the reciprocal rate. |
| `change_pitch` | Pitch-shifts by a fixed number of semitones (-5/-3/3/5) | `pitch_shift` with `n_steps = -original_n_steps` | Same story: pitch shift is parametric and inverted by negating the shift. |

Label -> separation-target lookup used by the `remove_target` recovery strategy
(`tools/extract_remove_target.py`'s `SEPARATION_LABELS` is a fixed, broad vocabulary
that doesn't have a 1:1 match to AudioSet ontology labels, so this is a best-effort
mapping, defaulting to `"background sound"` for anything unmapped):

| Inserted event label | `target_description` |
|---|---|
| Dog, Bird, Cat, Siren, Vehicle | `"background sound"` |
| Wind, Rain, Traffic noise | `"noise"` |

## Design notes

**Multiple recovery strategies per disturbance.** Where a disturbance has no exact
inverse (`add_noise`, `insert_background_event`), several plausible recovery tool calls
are registered and one is drawn at random per sample (`disturb_recover.RecoveryStrategy`
+ `build_recovery_step`). This is the "find more ways to construct such a pair" part of
the design: it keeps the model from memorizing a single fixed answer per disturbance
type and instead learning the *category* of correct response.

**Recovering a chain = walking it backwards.** When several disturbance ops are
chained (`--max-ops > 1`), the recovery chain undoes them in reverse order — the last
disturbance applied is the first one recovered. This holds together because every
recovery op here is duration-exact: it either restores the pre-disturbance duration
exactly (`clipping` undoes `pad_noise`'s length change; the inverse `time_stretch` rate
undoes `change_speed`'s) or leaves duration untouched entirely (`denoise`,
`remove_target`, `pitch_shift`). So by the time the reverse chain reaches step *i*, the
audio is back on step *i*'s own original timeline (same duration, same onset/offset
alignment) regardless of how faithfully the lossy steps (`denoise`, `remove_target`)
restored *content* — only content fidelity is best-effort; timing bookkeeping never
drifts.

**Execution: batched per turn, not per sample.** `apply_tools.py`'s own schedule format
fixes one `audio_path` per problem and reuses it for every step in that problem's tool
chain — it does not chain step *i*'s output into step *i+1*'s input. A genuine
multi-step recovery chain therefore can't be expressed as a single schedule file.
`build_by_disturb.py` instead executes one *turn* at a time across every sample in the
batch (all samples needing a denoise call at turn 2 run together, etc.), advancing each
sample's own `current_audio_path` between turns, and writes the schedule for that turn
(`tool_schedules/turn_N.json`) using each sample's real, already-computed input audio
for that turn — so it's simultaneously an efficient batched executor and a faithful,
independently-replayable `apply_tools.py` input per turn.

**Two small existing-repo gaps fixed in support of this pipeline** (both minimal,
additive, non-breaking):
- `tools/denoise.py` didn't exist (`tools/tool_execute.py`'s dispatch table pointed
  "denoise" at `tools.denoise.DenoiseTool`, but the implementation lived in
  `denoise_old.py`), so any schedule calling `"denoise"` through
  `apply_tools.py`/`tool_batch_execute.py` failed with "Tool source file not found".
  Added a one-line re-export shim.
- `Tool.execute_batch` had no default implementation in `tools/abstract_tool.py` — only
  the tools that happened to define their own (`denoise`, `pitch_shift`, `time_stretch`,
  `remove_target`, ...) worked with `tools/tool_batch_execute.py`. `ClippingTool` and
  the whole `tools/normalize.py` family (including `trim_silence`, used above) had none
  and crashed batch execution. Added a generic sequential default on the `Tool` base
  class; tools that can share expensive setup across a batch still override it.

## Output shape

For an input QA pair `{id, audio_path, question, choice, answer, ...}`,
`build_by_disturb.py` produces:

- `<output-dir>/disturbed/<id>/disturbed.wav` (+ intermediate per-step files) and
  `<output-dir>/disturbed_subset.json` — dcase_subset.json-shaped, `audio_path` pointing
  at the disturbed audio.
- `<tool-results-dir>/<id>/turn<N>_<tool>.wav` and
  `<tool-results-dir>/recovered_subset.json` — same shape, `audio_path` pointing at the
  final recovered audio (or, if a tool call in the chain failed, the last audio that
  *did* successfully process — never a missing file).
- `<tool-results-dir>/tool_schedules/turn_1.json`, `turn_2.json`, ... — one
  `apply_tools.py`-compatible schedule per recovery turn.
- `<output-dir>/manifest.json` — full bookkeeping per sample: the disturbance chain
  (op, params, before/after duration), the recovery chain (strategy, tool, parameters),
  and every turn's real output path.

Both `disturbed_subset.json` and `recovered_subset.json` are ready to pass as
`--subset_path` to `qwen25_with_tool_chain_evaluation.py`.
