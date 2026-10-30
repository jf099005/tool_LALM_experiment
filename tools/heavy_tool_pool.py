"""Client + lifecycle for running heavy ML tool calls (DeepFilterNet / AudioSR / SAM-Audio) in
their own conda envs.

Two ways a call gets executed, chosen automatically by `run_tool`:

- **Pooled** (inside `with serve(env_pythons):`): one long-lived `heavy_tool_server.py` per
  conda env holds each model in memory and answers calls over a Unix socket, serially. The
  socket directory is advertised through `HEAVY_TOOL_POOL_DIR`, so worker processes/threads
  the generator spawns *after* entering `serve()` (ProcessPoolExecutor, ThreadPoolExecutor)
  pick it up with no plumbing.
- **One-shot** (no `serve()` active): a fresh `tool_batch_execute.py` subprocess per call,
  which reloads the model every time -- fine for a handful of calls, far too slow (and,
  fanned out over many workers, too GPU-hungry) for dataset generation.

Stdlib only, so it imports in every env the generators run under.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent

POOL_DIR_ENV = "HEAVY_TOOL_POOL_DIR"
_SERVERS_FILE = "servers.json"

# Model load + queueing behind other workers can legitimately take minutes.
_CALL_TIMEOUT_SECONDS = 3600
_SERVER_START_TIMEOUT_SECONDS = 180


def _server_socket_for(env_python: str) -> Optional[str]:
    pool_dir = os.environ.get(POOL_DIR_ENV)
    if not pool_dir:
        return None
    try:
        mapping = json.loads((Path(pool_dir) / _SERVERS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return mapping.get(env_python)


def _call_server(socket_path: str, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    request = (json.dumps({"tool": tool_name, "params": params}) + "\n").encode("utf-8")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(_CALL_TIMEOUT_SECONDS)
            sock.connect(socket_path)
            sock.sendall(request)
            line = sock.makefile("rb").readline()
    except OSError as exc:
        raise RuntimeError(f"{tool_name}: heavy tool server at {socket_path} unreachable: {exc}") from exc
    if not line:
        raise RuntimeError(f"{tool_name}: heavy tool server closed the connection without a reply (crashed?)")

    response = json.loads(line)
    if not response.get("ok"):
        raise RuntimeError(f"{tool_name} failed: {response.get('error')}")
    return response["result"]


def _call_one_shot(tool_name: str, env_python: str, params: Dict[str, Any]) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = Path(tmp_dir) / "input.json"
        output_path = Path(tmp_dir) / "output.json"
        input_path.write_text(json.dumps([params]), encoding="utf-8")
        subprocess.run(
            [
                env_python,
                str(TOOLS_DIR / "tool_batch_execute.py"),
                "--tool-name", tool_name,
                "--input-file", str(input_path),
                "--output-file", str(output_path),
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )
        results = json.loads(output_path.read_text(encoding="utf-8"))

    if not isinstance(results, list) or len(results) != 1:
        # tool_batch_execute.py writes a single {"status": "error", "error": ...} dict when
        # the whole batch raised, rather than a per-item list.
        detail = results.get("error") if isinstance(results, dict) else results
        raise RuntimeError(f"{tool_name} failed: {detail!r}")
    return results[0]


def run_tool(tool_name: str, env_python: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one heavy tool call and return its result dict.

    `params` is sent as-is (include `output_path` for tools that require one). Raises
    RuntimeError if the call fails, whichever way the tool reports it.
    """
    socket_path = _server_socket_for(env_python)
    if socket_path is not None:
        result = _call_server(socket_path, tool_name, params)
    else:
        result = _call_one_shot(tool_name, env_python, params)

    if result.get("status") not in (None, "success"):
        raise RuntimeError(f"{tool_name} failed: {result.get('message') or result.get('error')}")
    return result


@contextlib.contextmanager
def serve(env_pythons: Iterable[str]) -> Iterator[None]:
    """Start one persistent server per distinct conda env python for the `with` body.

    A no-op when there is nothing to serve or a pool is already active (nested use). Servers
    load their models lazily, on the first call for a given tool, so listing an env whose
    tool never gets drawn costs nothing beyond an idle process.
    """
    env_pythons = sorted(set(env_pythons))
    if not env_pythons or os.environ.get(POOL_DIR_ENV):
        yield
        return

    # AF_UNIX paths are capped near 108 bytes, so keep this short (TMPDIR can be deep).
    try:
        pool_dir = Path(tempfile.mkdtemp(prefix="htp_", dir="/tmp"))
    except OSError:
        pool_dir = Path(tempfile.mkdtemp(prefix="htp_"))

    procs: list[subprocess.Popen] = []
    mapping: Dict[str, str] = {}
    try:
        for index, env_python in enumerate(env_pythons):
            socket_path = str(pool_dir / f"s{index}.sock")
            procs.append(
                subprocess.Popen(
                    [env_python, str(TOOLS_DIR / "heavy_tool_server.py"), "--socket", socket_path],
                    cwd=str(REPO_ROOT),
                )
            )
            mapping[env_python] = socket_path

        deadline = time.monotonic() + _SERVER_START_TIMEOUT_SECONDS
        for proc, (env_python, socket_path) in zip(procs, mapping.items()):
            while not Path(socket_path).exists():
                if proc.poll() is not None:
                    raise RuntimeError(f"heavy tool server for {env_python} exited during startup (code {proc.returncode})")
                if time.monotonic() > deadline:
                    raise RuntimeError(f"heavy tool server for {env_python} did not start within {_SERVER_START_TIMEOUT_SECONDS}s")
                time.sleep(0.2)

        (pool_dir / _SERVERS_FILE).write_text(json.dumps(mapping), encoding="utf-8")
        os.environ[POOL_DIR_ENV] = str(pool_dir)
        print(f"[heavy_tool_pool] serving {len(procs)} env(s): {sorted(mapping)}", file=sys.stderr, flush=True)
        yield
    finally:
        os.environ.pop(POOL_DIR_ENV, None)
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        shutil.rmtree(pool_dir, ignore_errors=True)
