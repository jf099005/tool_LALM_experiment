"""Re-export shim: `tools/tool_execute.py`'s dispatch table (`_TOOL_MODULES`) looks
up the "denoise" tool at `tools.denoise.DenoiseTool`, but the implementation lives in
`denoise_old.py`. Without this file, any schedule that calls "denoise" through
`apply_tools.py` / `tool_batch_execute.py` fails with "Tool source file not found".
"""

from __future__ import annotations

from .denoise_old import DenoiseTool

__all__ = ["DenoiseTool"]
