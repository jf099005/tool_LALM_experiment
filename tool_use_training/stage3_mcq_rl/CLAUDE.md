# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Scope: `tool_use_training/stage3_mcq_rl/` — stage-3 RL. The repo-root `CLAUDE.md` (cluster rules: never run compute directly, write `.sbatch` for the user to submit; conda envs; tool architecture) is also loaded and still applies. `README.md` here (Chinese) is the user-facing summary of the same design.

## What this subproject does

GSPO (verl-omni, LoRA r=32 on q/k/v/o) trains Qwen3-Omni, starting from the merged stage-1 tool-use checkpoint, to answer audio MCQs while **actually calling audio tools mid-episode**:

```
system    stage-1's exact tool catalogue (byte-identical to /work/u1501463/stage1_full_toolset/train.jsonl)
user      task instruction + "Input audio: <audio_0><audio>" + question + (A)..(D)   (options reshuffled per item)
assistant <tool_call>{"name": ..., "arguments": {...}}</tool_call>          ← policy (loss)
tool      Output of step 1: <audio_1><audio>   (real tool output audio)       ← env (masked)
assistant <answer>B</answer>                                                ← policy (loss)
```
At most 4 assistant turns (`multi_turn.max_assistant_turns`). Reward = 1.0·correct + 0.1·well-formed − 0.05·min(tool_errors,3) (+ optional `STAGE3_TOOL_BONUS`, default 0).

## Files and how they connect

- `mcq_protocol.py` — the only definition of the MCQ prompt, option lettering/shuffle, `<answer>` extraction, and the system prompt (`model_type="qwen3_omni"` → empty persona). Imported by the data builder, agent loop and reward; change wording here only.
- `build_mcq_dataset.py` — writes `train.parquet` (compa_r 8k strong-audio-contribution AudioMCQ items, dcase2025 all, audioset_labels 2k synthesized), `val.parquet` (200 stratified MMAU test-mini), `eval_mmau_test_mini.parquet` (1000), `eval_mmar.parquet` (999), `stats.json`. Row schema: `prompt`, `audios` (path list), `agent_name="audio_tool_mcq_agent"`, `reward_model.ground_truth` (letter), `extra_info`. Eval sets keep full denominators (dedup duplicated options; fuzzy-map answer text, flagged in `extra_info.repaired`); training drops malformed items. audioset distractors only come from ontology roots Animal / Natural sounds / Sounds of things.
- `audio_tool_agent_loop.py` — `AudioToolMCQAgentLoop` (registered by `agent_loop.yaml`, selected per row by `agent_name`). Uses `interface.protocol.parse_turn` + `interface.executor.run_tool_call`; ids assigned like `interface/agent.py`. Episode stats go into `extra_fields`.
- `reward.py` — verl `custom_reward_function`; reads `extra_info["final_text"]`/`stop_reason` (set by the agent loop), not `solution_str`.
- `launch_with_tool_pool.py` — starts heavy-tool servers (`tools_registry.serve_heavy_tools()`), then runs the training command as a child so Ray workers inherit `HEAVY_TOOL_POOL_DIR`. Server-only env: `HEAVY_TOOL_GPU` (default 3), `HEAVY_TOOL_LD_LIBRARY_PATH` (cuda/12.4 libs for sam_audio).
- `run_grpo.sh` — all hydra overrides (adapted from verl-omni `examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_avqa_v1.sh`). Env-overridable: `MODEL_PATH DATA_DIR TRAIN_FILE VAL_FILES OUT_ROOT EXP_NAME NGPUS ROLLOUT_TP GPU_MEM_UTIL LOGGER TOTAL_STEPS`; extra args = hydra overrides.
- `*.sbatch` — `merge_stage1` (swift-qwen3 env, `swift export --merge_lora`), `build_data`, `smoke_test`, `smoke_test_1gpu` (same checks on 1 GPU for when 4-GPU jobs queue: vllm TP=1 at 0.88 mem, heavy-tool servers on CPU via `HEAVY_TOOL_GPU=""`, batch 2×2), `train`, `eval` (val_only; `CKPT=<global_step_dir>` for RL checkpoints, else evaluates `MODEL_PATH`).

## Commands

Submission order (user runs these from this directory): `sbatch merge_stage1.sbatch` → `sbatch build_data.sbatch [build_mcq_dataset args]` → `sbatch smoke_test.sbatch` → `sbatch train.sbatch [hydra overrides]` → `CKPT=... sbatch eval.sbatch`.

CPU-only agent-loop test (seconds, no GPU; fake LLM server + real `clipping` tool; asserts placeholder/audio count, loss mask, token parity with the HF chat template, reward):
```bash
E=~/.conda/envs/verl-omni
$E/bin/python tests/smoke_agent_loop.py --parquet /work/u1501463/stage3_mcq_rl/data_smoke/train.parquet --row 0
```
Tiny data build for debugging (light IO only): `$E/bin/python build_mcq_dataset.py --output-dir /work/u1501463/stage3_mcq_rl/data_smoke --compa-r-n 20 --dcase-n 20 --audioset-n 20 --val-n 8`.

Outputs: `/work/u1501463/stage3_mcq_rl/{data,ckpt,val_generations,tensorboard}/<EXP_NAME>`; Slurm logs in `./logs/`; hydra run dirs in `./outputs/`.

## Invariants that are easy to break

- **Token parity with stage-1**: the system turn must stay byte-identical to stage-1 SFT data, and the rollout token stream must equal the HF template rendering of the same conversation (the smoke test checks this). The loop forces `ContinuousTokenModelFamily.QWEN_VL` (verl has no family for `qwen3_omni_moe` and would fall back to a generic builder that drops the `\n` after `<|im_end|>`).
- **Audio in messages must be arrays, not paths**: verl's Continuous Token builder re-renders the whole history through the processor (e.g. for the generation-prompt delta); the Whisper feature extractor can't take paths. The loop swaps the user turn's audio path for the loaded array after `process_multi_modal_info`, and tool-output audio is loaded via the same dataset class (16 kHz + hop padding) so rollout and actor recompute see identical token counts.
- **Audio list order == placeholder order**: append each tool output to `audios` only after its tool message is merged. Never commit a tool turn that pushes the response past `response_length` (truncating inside an audio placeholder run misaligns features) — the loop stops instead.
- **Inputs may not be real WAV**: CompA-R is mp3; DCASE/MMAU contain mp3/flac named `.wav`. The `wave`-based tools need RIFF WAV, so `_execute` calls `tools_registry.ensure_wav` (checks the RIFF header) on the input first.
- **Heavy tools need the pool**: the agent loop raises if heavy tools are registered but `HEAVY_TOOL_POOL_DIR` is unset; always launch through `launch_with_tool_pool.py`.
- **Reward metric keys** must be numeric and present for every sample (no NaN — verl averages them). Use products like `acc_and_used_tool` instead of conditional means.
- **Never train on MMAU/MMAR**; they are val/eval only.

## verl / verl-omni internals relied on

- `main_omni` uses the V1 trainer (`verl/trainer/ppo/v1/`, TransferQueue `AgentLoopWorkerTQ`). Reward is computed inside the agent-loop worker: `AgentLoopOutput.extra_fields` → `tool_extra_fields` → merged into `extra_info` by the naive reward manager.
- Validation sampling only takes `val_kwargs.temperature/top_p/top_k` (we use temperature 0 = greedy); `val_only` + `resume_mode=resume_path` is how `eval.sbatch` evaluates checkpoints.
- verl's `_compute_multi_modal_inputs` decodes with `skip_special_tokens=True`, which strips `<|audio_pad|>`; features are still recomputed from the per-sample audio list in order (same path verl-omni's AVQA example relies on).
- Stock `ToolAgentLoop` supports only image tool outputs — that's why this custom loop exists. Source: `~/.conda/envs/verl-omni/lib/python3.10/site-packages/verl/experimental/agent_loop/`, verl-omni at `~/verl-omni`.

## Current status (2026-10-26)

All env changes are logged in `ENV_NOTES.md` (keep it updated). Latest first:

- `smoke_test` job 412571 (hgpn04, 10-27, budget 2-GPU version): steps 0–1 PASSED; Ray hang gone (reached vLLM init in ~4 min). Step 2 FAILED: `pthread_create failed: Resource temporarily unavailable` in raylet, Ray dashboard agent and the vLLM TP worker (`ProcessGroupGloo` → `RuntimeError: Resource temporarily unavailable`) — the job ran out of threads/processes. Cluster cgroup has no pids limit but `ConstrainRAMSpace=yes`; exact limit hit unknown (`sacct` is hook-blocked). New `smoke_test.sbatch` caps thread pools (`OMP/MKL/OPENBLAS/NUMEXPR/RAYON_NUM_THREADS=2`, `RAY_num_server_call_thread=2`), cuts agent-loop/reward workers 8→2, dataloader workers 8→0, disables the Ray dashboard, sets `--mem=320G`, raises soft `ulimit -u` to hard, and logs the limits + thread counts. If train/eval hit the same error, port those settings to `train.sbatch`/`eval.sbatch`.

- `smoke_test` job 410458 (hgpn19, 10-26): step 0 preflight OK (flashinfer patch works), step 1 PASSED, heavy-tool pool up. Step 2 then **hung silently** right after "Started a local Ray instance" — the TaskRunner actor never started (no output at all) until the 2 h time limit; cause unknown because stdout was buffered and Ray's `/tmp/ray` logs were lost. `smoke_test.sbatch` now sets `PYTHONUNBUFFERED=1`, `ray_kwargs.ray_init.num_cpus=$SLURM_CPUS_PER_TASK` (verl's config says to fix it under Slurm), wraps step 2 in a 30 min `timeout`, and copies Ray session logs to `/work/u1501463/stage3_mcq_rl/smoke/ray_logs/<jobid>`. Rerun and read those logs (`raylet.out`, `worker-*.err`, `gcs_server.out`) if it hangs again. Since 10-27 the smoke test is a budget run: 2 GPUs (layout derived from `SLURM_GPUS_ON_NODE`; `sbatch --gpus-per-node=4` mirrors train), `--time=00:40:00`, vLLM `enforce_eager`, 2048-token responses, val on 2 items + 1 train step, tool timeout 180 s.

- `smoke_test` job 390038 (hgpn17): step 1 PASSED; Ray/verl init and the vllm+cu129 import got past the earlier failures; step 2 FAILED at vLLM worker init: flashinfer `comm/fd_exchange.py` uses `array.array[int]` (Python ≥ 3.12 only; env is 3.10). Fixed 2026-10-26 by a site-packages patch (`from __future__ import annotations`), reproducible via `env_patches/apply_env_patches.sh`; `smoke_test.sbatch` now has a step-0 import preflight. Needs a rerun of `smoke_test.sbatch` — rollout on the 550 driver is still unverified.

Older:

- Not yet run: `merge_stage1`, `build_data` (only `/work/u1501463/stage3_mcq_rl/data_smoke` exists), `train`, `eval`.
- `smoke_test` job 387727 (node hgpn20, base model since no merged ckpt yet): step 1 (CPU agent loop) PASSED; heavy-tool pool started all 3 envs. Step 2 FAILED at actor init on Slurm's `ROCR_VISIBLE_DEVICES` → fixed: `run_grpo.sh` now `unset`s it.
- GPU nodes run NVIDIA driver 550.127.08 (CUDA ≤ 12.4). vllm was a CUDA-13 build → replaced with the official `vllm-0.27.0+cu129` wheel (`pip install --no-deps`, from wheels.vllm.ai) plus `nvidia-cuda-{nvcc,cccl}-cu12==12.9.*` (its `humming-kernels[cu12]` extra); the `nvidia/cu13/lib` LD path hack is gone. vllm-omni is pure Python (no rebuild needed). Rollout on the 550 driver relies on CUDA 12 minor-version compat — not yet verified on a GPU node; rerun `smoke_test.sbatch`.
- `smoke_test` job 388440 (hgpn05): step 1 PASSED, ROCR fix worked; step 2 FAILED at actor model init: `FlashAttention2 ... package doesn't seem to be installed` (transformers fell back to the HF kernels hub `kernels-community/flash-attn2`, which has no torch213 build). No flash-attn build exists for torch 2.13 (GitHub wheels stop at torch 2.8). Fixed: `run_grpo.sh` now uses `ATTN_IMPL=sdpa` (default) with `use_remove_padding=False` (remove-padding needs flash-attn varlen; with sdpa packed samples would cross-attend). Setting `ATTN_IMPL=flash_attention_2` re-enables remove-padding, but only if flash-attn gets compiled from source for torch 2.13 first. vllm rollout on the 550 driver is still unverified (this job died before rollout init).
- Env changes already made in verl-omni: `kernels` pinned to 0.12.3 (0.16 broke `transformers` import); added `librosa`, `qwen-omni-utils`, `qwen-vl-utils`.
