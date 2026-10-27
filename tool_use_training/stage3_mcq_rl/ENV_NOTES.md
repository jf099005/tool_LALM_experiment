# `verl-omni` env: changes made and how to reproduce them

Record of every manual change to `/home/u1501463/.conda/envs/verl-omni` needed to run stage-3,
plus what a new developer should do when building the env from scratch. Cluster context: NCHC
H100 nodes, NVIDIA driver 550.127.08 (CUDA ≤ 12.4 driver API; CUDA 12.x runtimes work via
minor-version compatibility, CUDA 13 does not).

Env snapshot: Python **3.10**, torch 2.13.0+cu129, transformers 5.10.4, verl 0.10.0.dev0,
verl-omni 0.2.0rc1 (editable, `~/verl-omni`), vllm 0.27.0+cu129, vllm-omni 0.27.0rc2,
flashinfer-python 0.6.16.post3.

## Changes, in the order they were found

| # | Date | Symptom | Change | Survives `pip install -U`? |
|---|---|---|---|---|
| 1 | 10-12 | `from transformers import AutoProcessor` → `ValueError: Either a revision or a version must be specified` | `pip install "kernels>=0.12,<0.13"` (0.12.3; transformers 5.10 declares this range, 0.16 was installed) | no — re-pin after upgrades |
| 2 | 10-12 | tools_registry light tools / `QwenOmniRLHFDataset` import errors | `pip install librosa qwen-omni-utils qwen-vl-utils` | yes |
| 3 | 10-15 | PyPI `vllm==0.27.0` is a CUDA-13 build (`libcudart.so.13`), can't run on driver 550 | replaced with `vllm-0.27.0+cu129` from wheels.vllm.ai via `pip install --no-deps`, plus `nvidia-cuda-nvcc-cu12==12.9.*` `nvidia-cuda-cccl-cu12==12.9.*` | **no** — never `pip install -U vllm` from PyPI here |
| 4 | 10-15 | verl `Worker.__init__`: `Please don't set ROCR_VISIBLE_DEVICES when HIP/CUDA_VISIBLE_DEVICES is set` (smoke 387727) | `run_grpo.sh` does `unset ROCR_VISIBLE_DEVICES` (Slurm exports both) | n/a (script) |
| 5 | 10-16 | no flash-attn wheel exists for torch 2.13 (smoke 388440) | `run_grpo.sh`: `ATTN_IMPL=sdpa`, `use_remove_padding=False` | n/a (script) |
| 6 | 10-26 | every vLLM rollout worker: `flashinfer/comm/fd_exchange.py:55 … array.array[int] … TypeError: 'type' object is not subscriptable` (smoke 390038) | added `from __future__ import annotations` to that file (site-packages edit). Automated + idempotent: `env_patches/apply_env_patches.sh` | **no** — re-run the patch script after any flashinfer reinstall |

### Why #6 happens

`array.array` became subscriptable only in Python 3.12. vLLM imports `flashinfer.comm` unconditionally
while building its tensor-parallel communicator (`cuda_communicator.py`), so no vLLM flag avoids it,
and the pinned flashinfer version is exactly what vllm 0.27.0 requires. Only this one line in
`flashinfer/comm/` was found; other Python-3.12-only code elsewhere in the stack could still surface
on later code paths — if a new `TypeError`/`SyntaxError` points into site-packages, suspect the same
root cause.

## Building the env from scratch (new developer)

**Preferred: use Python 3.12** (what verl-omni's `docs/start/install.md` uses). Then #6 is not needed,
and the whole class of Python-version breakage goes away:

1. `conda create -n verl-omni python=3.12`, follow verl-omni's install guide.
2. Install vllm as the **cu129** wheel matching the pinned version (#3), not the PyPI default — check
   `nvidia-smi` driver version first; driver ≥ 580 can use the default CUDA-13 wheel.
3. Apply #1 and #2. `run_grpo.sh` already contains #4/#5.
4. `bash env_patches/apply_env_patches.sh` is harmless on 3.12 (it reports "not needed").

**Reusing Python 3.10 (current env):** do steps 2–3 above, then run
`bash env_patches/apply_env_patches.sh` (CPU-only; patches + verifies `import flashinfer.comm`).
`smoke_test.sbatch` step 0 re-checks this on the compute node and stops early with a pointer here.

On this cluster, run env pip as `env PATH=/home/u1501463/.conda/envs/verl-omni/bin:/usr/bin pip ...`
(the Claude Code hook only permits bare `pip`, which otherwise resolves to the OS python).
