"""Execute one model-predicted tool call against real audio.

Deliberately goes straight to `tools.TOOL_NAME_TO_CLASS` -- the real `Tool`
subclasses under `tools/` -- rather than `tools/synthetic_registry.py`. That
registry exists only to *synthesize* ground-truth training samples (it
invents random parameters and is deliberately limited to whatever's
importable in the ms-swift env used for data generation); it isn't meant to
replay an arbitrary parameter dict a model produced, and gates on its own
active-tool set rather than the full one defined in `tools/`. Both
`agent.ToolCallingAgent` and `testing_tool_use_benchmark/run_eval.py` drive
predicted tool calls through this same module -- generalized to run any tool
in `tools.TOOL_NAME_TO_CLASS`, including ones synthetic_registry never
registers, like `asr` and `source_separation` -- and to tolerate tools whose
result carries no new audio at all (`asr`) or more than one (`source_separation`).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import TOOL_NAME_TO_CLASS  # noqa: E402
from tools.abstract_tool import ToolValidationError  # noqa: E402


class UnknownToolError(ValueError):
    """Raised when the model names a tool that isn't in `tools.TOOL_NAME_TO_CLASS`."""


class ToolExecutionError(RuntimeError):
    """Wraps any validation or runtime failure while executing a tool call."""


def run_tool_call(
    tool_name: str,
    parameters: Dict[str, Any],
    input_audio_path: str,
    output_dir: Path,
    step_index: int,
) -> Dict[str, Any]:
    """Run one (tool_name, parameters) call for real, returning the tool's raw result dict.

    `parameters["audio_id"]` (whatever id string the model emitted, e.g.
    "audio_0") is a textual convention the model uses to refer back to a
    prior audio -- the underlying tool implementations don't understand it,
    so it's dropped here and replaced with `input_audio_path`, the real file
    the caller (`agent.ToolCallingAgent`) already resolved that id to.

    Raises `UnknownToolError` if `tool_name` isn't a real tool, or
    `ToolExecutionError` wrapping any validation/runtime failure -- callers
    should catch these per step rather than letting one bad call abort the
    whole chain.
    """
    if tool_name not in TOOL_NAME_TO_CLASS:
        raise UnknownToolError(
            f"'{tool_name}' is not an available tool (have: {sorted(TOOL_NAME_TO_CLASS)})"
        )

    cls = TOOL_NAME_TO_CLASS[tool_name]
    params = dict(parameters or {})
    params.pop("audio_id", None)
    params["audio_path"] = str(input_audio_path)

    schema_props = cls.parameter_schema().get("properties", {})
    if "output_path" in schema_props and not params.get("output_path"):
        output_dir.mkdir(parents=True, exist_ok=True)
        params["output_path"] = str(output_dir / f"step{step_index}_{tool_name}.wav")

    try:
        cls.validate_parameters(params)
        return cls.execute(params)
    except ToolValidationError as exc:
        raise ToolExecutionError(f"{tool_name} validation failed: {exc}") from exc
    except (UnknownToolError, ToolExecutionError):
        raise
    except Exception as exc:  # noqa: BLE001 - surface any backend failure as an execution failure
        raise ToolExecutionError(f"{tool_name} raised {type(exc).__name__}: {exc}") from exc


def extract_audio_outputs(result: Dict[str, Any]) -> Dict[str, str]:
    """Pull every new audio file path out of a tool result, keyed by stem name.

    Most tools produce exactly one new audio, under `output_path` (or
    `clip_path` for `ClippingTool`) -- returned here under the empty-string
    key. `SourceSeparationTool` (and friends) can produce up to two, under
    `separated_files` (`{"target": ..., "residual": ...}`). `ASRTool`
    produces no new audio at all -- its output is the `transcript` field
    only -- so this returns `{}` for it.
    """
    separated = result.get("separated_files")
    if separated:
        return dict(separated)
    for key in ("output_path", "clip_path"):
        value = result.get(key)
        if value:
            return {"": value}
    return {}
