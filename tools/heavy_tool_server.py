"""Long-lived worker for the heavy ML tools (DeepFilterNet / AudioSR / SAM-Audio).

Run under the tool's own conda env (see `tools_registry._CONDA_ENV_PYTHON`), started by
`heavy_tool_pool.serve()`. It listens on a Unix socket and answers one JSON line per
connection, serially:

    request : {"tool": "<tool_name>", "params": {...}}\\n
    response: {"ok": true, "result": {...}}\\n  |  {"ok": false, "error": "..."}\\n

Why this exists instead of one `tool_batch_execute.py` subprocess per call: every such call
reloads its model (SAM-Audio-large takes tens of seconds and ~10 GB of GPU memory), and the
dataset generators fan out across many worker processes. One server per env keeps each model
loaded once, and serving requests serially keeps GPU memory bounded no matter how many
generator workers are queued behind it.

Stdlib-only at import time, so it starts in any env; tool modules (and their heavy
dependencies) are imported lazily on the first request for that tool.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socketserver
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

_TOOL_CLASSES: Dict[str, Any] = {}


def _tool_class(tool_name: str):
    if tool_name not in _TOOL_CLASSES:
        from tool_execute import get_tool_class

        _TOOL_CLASSES[tool_name] = get_tool_class(tool_name)
    return _TOOL_CLASSES[tool_name]


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        line = self.rfile.readline()
        if not line:
            return
        try:
            request = json.loads(line)
            results = _tool_class(request["tool"]).execute_batch([request["params"]])
            if not isinstance(results, list) or len(results) != 1:
                raise RuntimeError(f"Unexpected batch result: {results!r}")
            response: Dict[str, Any] = {"ok": True, "result": results[0]}
        except Exception as exc:  # noqa: BLE001 - reported back to the caller, server stays up
            traceback.print_exc()
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))


class _Server(socketserver.UnixStreamServer):
    request_queue_size = 512


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve heavy ML tool calls over a Unix socket.")
    parser.add_argument("--socket", required=True, help="Unix socket path to listen on.")
    args = parser.parse_args()

    # Anything a tool/library prints to stdout would otherwise fill an unread pipe.
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    # Die with the parent (Linux) so an aborted generator never leaves a GPU-holding orphan.
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001
        pass

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    with _Server(args.socket, _Handler) as server:
        print(f"heavy_tool_server listening on {args.socket}", file=sys.stderr, flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
