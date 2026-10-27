"""Run a command with the heavy-tool servers (DeepFilterNet / AudioSR / SAM-Audio) up.

    python launch_with_tool_pool.py -- bash run_grpo.sh [hydra overrides...]

Starts one persistent `tools/heavy_tool_server.py` per conda env (models load lazily, once)
via `tools_registry.serve_heavy_tools()`, which exports HEAVY_TOOL_POOL_DIR. The training
command is launched as a child, so the Ray driver and every Ray worker inherit that variable
and `interface.executor.run_tool_call` reaches the servers instead of reloading a model per
call. `audio_tool_agent_loop.py` refuses to run without it.

Server-only environment (restored before the training command starts):
  HEAVY_TOOL_GPU             e.g. "3": pin the tool servers to one GPU (they'd otherwise use cuda:0);
                             unset = inherit the job's CUDA_VISIBLE_DEVICES.
  HEAVY_TOOL_LD_LIBRARY_PATH LD_LIBRARY_PATH for the servers -- sam_audio's torchcodec needs the
                             system CUDA 12.4 NPP libs (`module load cuda/12.4`), which must not
                             leak into the verl-omni (torch cu129 / flashinfer) processes.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools import tools_registry  # noqa: E402


def main() -> int:
    argv = sys.argv[1:]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        raise SystemExit(__doc__)

    server_only = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("HEAVY_TOOL_GPU"),
        "LD_LIBRARY_PATH": os.environ.get("HEAVY_TOOL_LD_LIBRARY_PATH"),
    }
    saved = {key: os.environ.get(key) for key in server_only}
    for key, value in server_only.items():
        if value is not None:
            os.environ[key] = value  # inherited by the tool servers spawned below
    with tools_registry.serve_heavy_tools():
        child_env = dict(os.environ)
        for key, value in saved.items():
            if value is None:
                child_env.pop(key, None)
            else:
                child_env[key] = value
        print(f"[launch_with_tool_pool] tools={tools_registry.available_tool_names()} "
              f"pool={child_env.get('HEAVY_TOOL_POOL_DIR')}", flush=True)
        return subprocess.call(argv, env=child_env)


if __name__ == "__main__":
    sys.exit(main())
