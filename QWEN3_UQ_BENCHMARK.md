# Qwen3-Omni UQ benchmark on the cluster — runbook for an agent

You are asked to run `qwen3_uq_benchmark.py` on nano5 for an exported item set, by writing
sbatch files for the user to submit. Read this whole file first, do the checks in §3 (no
GPU needed), then write the two sbatch files in §4. **Do not submit them yourself** — the
repo-root `CLAUDE.md` cluster rules apply (never run GPU work directly; hand `.sbatch` files
to the user). If a check fails and the fix isn't obvious, stop and report instead of guessing.

## 1. What the experiment is

The items were selected on a local PC from a Qwen2.5-Omni-3B uncertainty-quantification (UQ)
run over 50k AudioMCQ questions (Clotho + AudioCaps, "strong audio contribution"). Every item
exists twice: the **original** audio, and a **tool-applied** version where one audio tool
was applied (40% ML tools: SAM-Audio extract/remove, DeepFilterNet, AudioSR; 60% signal tools:
clipping, denoise, pitch/time, normalisation …). The exported set holds the items whose UQ
metric changed most with the tool (e.g. `p_true_delta_ge0.1` = P(True) rose by ≥ 0.1).

`qwen3_uq_benchmark.py` asks Qwen3-Omni the same MCQs on both versions — "without tool"
(original) vs "with tool" (tool-applied) — in lockstep, and measures accuracy, accuracy gain,
the T→T/T→F/F→T/F→F confusion and the UQ metrics (predictive entropy, length-normalised
entropy, discrete semantic entropy over 10 samples at T=1.0, and P(True) self-verification),
overall and per tool. **No audio tools run on the cluster**: the tool outputs are already in
the exported set.

## 2. Inputs

**Exported set** (uploaded from the PC by `export_and_push.sh` / `push_exports_to_nano5.sh`):

```
/work/u1501463/LALM_UQ/exports/<SET>/        e.g. <SET> = p_true_delta_ge0.1
  original_subset.json       list of {id, question, choice[], answer, audio_path, ...}
  tool_applied_subset.json   same ids, same order, tool-applied audio
  manifest.json              per id: tool_chain[{tool, parameters, ...}]  (per-tool breakdown)
  filter.json                selection rule + the Qwen2.5 values/correctness per item
  original/<id>/original.wav, tool_applied/<id>/tool_applied.wav
  confusion.png              Qwen2.5 confusion matrix for this set (for reference)
```

The `audio_path` values in the jsons are paths **on the PC**; the script re-roots them to the
set's actual location using `filter.json`'s `export_dir` — nothing to fix by hand.

**Paths** are defined once in `cluster_paths.sh` (env vars override):
`REMOTE_EXPORT_ROOT=/work/u1501463/LALM_UQ/exports`,
`REMOTE_QWEN3_OUTPUT_ROOT=/work/u1501463/LALM_UQ/qwen3_uq`,
`REMOTE_HF_CACHE=/work/u1501463/hf_cache`.

**Code** lives on branch `local-uq-pc` (based on `main`, *not* on `qwen_protocol`). If the
cluster checkout is on `qwen_protocol`, don't merge — take just the files this needs:

```bash
git fetch origin
git checkout origin/local-uq-pc -- qwen3_uq_benchmark.py run_qwen3_uq_benchmark.sh cluster_paths.sh \
    qwen25_with_tool_chain_evaluation.py uncertainty_quantification_tools/ QWEN3_UQ_BENCHMARK.md
```

`qwen3_uq_benchmark.py` imports, from the repo root: `qwen25_with_tool_chain_evaluation`
(result-file helpers; it imports `prompts`, `librosa`, `vllm`), `qwen3_with_tool_chain_evaluation`
(`build_standard_prompt`, the project's Qwen3-Omni prompt) and `uncertainty_quantification_tools`.
`prompts/` and `qwen3_with_tool_chain_evaluation.py` already exist on both branches.

## 3. Checks before writing the sbatch (CPU only)

1. **Set present**: `ls $REMOTE_EXPORT_ROOT/<SET>` shows the files above; `original_subset.json`
   and `tool_applied_subset.json` have the same length and ids; a few `original/<id>/original.wav`
   files exist. If the set is missing, stop: the user still has to run `export_and_push.sh` locally.
2. **Code present**: the files from §2 are in the checkout; `python -m py_compile qwen3_uq_benchmark.py`.
3. **Environment**: use `verl-omni` — its vLLM is the `0.27.0+cu129` wheel that runs on the
   GPU nodes' driver 550 (the stock PyPI wheel does not). Inside it, without a GPU:
   `python -c "import vllm, librosa, tqdm; from vllm import LLM, SamplingParams; print(vllm.__version__)"`
   and confirm the Qwen3-Omni thinker is a registered vLLM architecture
   (`python -c "from vllm import ModelRegistry; print([a for a in ModelRegistry.get_supported_archs() if 'Qwen3Omni' in a])"`).
   Also `python -c "import qwen3_uq_benchmark"` from the repo root (imports only; no model load).
   If an import is missing, `pip install` into verl-omni is allowed — note it in `ENV_NOTES`-style.
4. **Model**: prefer the local copy of `Qwen3-Omni-30B-A3B-Instruct` under `/work/u1501463/`
   (find the directory with its `config.json`) over a hub download; pass it as `MODEL`.
5. **GPU type**: nano5 GPU nodes are H100 → use `DTYPE=bfloat16` (the script's `float16`
   default dates from V100 runs). `TP=2` is the safe choice for the 30B-A3B model (≈61 GB of
   bf16 weights + audio encoder + KV cache).

## 4. The sbatch files to write

Write both in the repo root (the scripts `cd` there), following the cluster pattern:

**a) smoke test** — 8 items, separate output dir, short time limit:

```bash
#!/bin/bash
#SBATCH --job-name=qwen3_uq_smoke
#SBATCH --partition=normal
#SBATCH --account=MST113234
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH -o ./logs/qwen3_uq_smoke_%j.log
#SBATCH -e ./logs/qwen3_uq_smoke_%j.err

ml load miniconda3/24.11.1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate verl-omni
unset ROCR_VISIBLE_DEVICES          # Slurm exports it; some GPU stacks reject it
export PYTHONUNBUFFERED=1
mkdir -p logs
source ./cluster_paths.sh

SET=p_true_delta_ge0.1               # <- the exported set to run
MODEL=/work/u1501463/<path to Qwen3-Omni-30B-A3B-Instruct>   # <- from check 4
python qwen3_uq_benchmark.py \
    --subset_dir "${REMOTE_EXPORT_ROOT}/${SET}" \
    --output_dir "${REMOTE_QWEN3_OUTPUT_ROOT}/${SET}_smoke" \
    --model "${MODEL}" --tensor_parallel_size 2 --dtype bfloat16 \
    --download_dir "${REMOTE_HF_CACHE}" \
    --batch_size 8 --limit 8 \
    --uq_num_samples 10 --uq_sample_temperature 1.0 --uq_p_true --no-uq_semantic_entropy
```

**b) full run** — same, but without `--limit`, `--output_dir "${REMOTE_QWEN3_OUTPUT_ROOT}/${SET}"`,
`--batch_size 16`, and a longer `--time` (a few hundred items × 2 sides finish well within
2 h on 2×H100; scale for bigger sets). Equivalent: `SET=<SET> MODEL=... TP=2 DTYPE=bfloat16
./run_qwen3_uq_benchmark.sh` inside the sbatch body.

Hand the user the smoke sbatch first; only after its log looks right (§5), the full one. The
full run is resumable: if it hits the time limit, resubmitting the same sbatch continues where
it stopped (progress is appended per batch to `*.partial.jsonl`).

## 5. Checking the smoke run (from its log / output dir)

Output dir contains `Qwen3-Omni-30B-A3B-Instruct_original_uq_results_no_tool_chain_results.json`,
`..._tool_applied_...json` (8 entries each) and `..._summary.json`. Verify:

- each entry's `prediction` is a real answer (e.g. `"B. Dog barking"`), not empty or garbled —
  garbage here means the audio placeholder/prompt does not match the model;
- `uncertainty` has non-null `predictive_entropy`, `length_normalized_entropy`,
  `discrete_semantic_entropy`, and `p_true` between 0 and 1 for most items. All-null
  entropies mean the vLLM version stopped returning `cumulative_logprob`/`logprobs` in the
  expected shape (see `_sequence_logprob` in `uncertainty_quantification_tools/uncertainty.py`);
  all-null `p_true` means the first generated token's top-20 logprobs had no True/False token —
  print `p_true_raw_response` to see what the model answered;
- the summary prints both accuracies and the confusion counts.

## 6. What to report back

From `<tag>_summary.json` (also printed at the end of the log): `overall.accuracy_original`,
`overall.accuracy_tool_applied`, `overall.accuracy_gain`, `overall.confusion`,
`overall.net_correct_gain`, the mean of each UQ metric per side and its mean Δ, the
`per_tool` table, and `reference_run` (Qwen2.5-Omni-3B accuracies on the same items, for
comparison). The summary can be regenerated without the GPU:
`python qwen3_uq_benchmark.py --subset_dir ... --output_dir ... --summary_only`.
For a confusion-matrix figure on the Qwen3 results, `uq_threshold_confusion.py` accepts the
two result files via `--original/--tool-applied` (see the comment at the end of
`run_qwen3_uq_benchmark.sh`).

## 7. Known untested points (the script has not been run anywhere yet)

- Written against vLLM 0.10's `LLM.generate` / `SamplingParams(n=…, logprobs=…)` output
  objects; verify on 0.27 via the smoke run (§5). `mm_processor_cache_gb` is only passed when
  the installed vLLM has it.
- `limit_mm_per_prompt={"audio": 1}`; image/video are left at vLLM defaults. If engine start-up
  profiling OOMs, lower `--gpu_memory_utilization`/`--max_model_len` (default 20000; 8192 is plenty
  for ≤45 s clips) before changing anything else.
- The GPU nodes have no nvcc, so FlashInfer's JIT top-k/top-p sampler fails at engine start-up
  (`Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`, smoke job 427150);
  the sbatch files set `VLLM_USE_FLASHINFER_SAMPLER=0` (PyTorch sampler, same distribution).
- Earlier stage-3 smoke jobs on these nodes died with `pthread_create failed: Resource
  temporarily unavailable`; if that appears, cap threads in the sbatch (`OMP_NUM_THREADS=2`,
  `MKL_NUM_THREADS=2`, `RAYON_NUM_THREADS=2`) as `stage3_mcq_rl/smoke_test.sbatch` does.

## 8. Keep fixed (comparability with the Qwen2.5 results)

- The prompt (`build_standard_prompt` from `qwen3_with_tool_chain_evaluation.py`) and its system
  prompt (the Qwen persona used by that script — intentionally *not* the empty stage-1
  `qwen3_omni` persona), greedy decoding for the answer, 10 samples at T=1.0, P(True) with
  `max_tokens=5`, and the "correct = answer text contained in the prediction" rule.
- Don't edit `uncertainty_quantification_tools/` semantics; it is shared with the running
  Qwen2.5 experiment.
