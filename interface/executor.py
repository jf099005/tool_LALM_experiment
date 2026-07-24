"""Execute one model-predicted tool call against real audio.

Looks the tool class up in `tools.TOOL_NAME_TO_CLASS` (needed for the actual
`execute()`/`validate_parameters()` implementation) but only *allows* a call
whose `tool_name` is in `tools.tools_registry`'s curated, project-wide
toolset -- the same one `interface/protocol.py` advertises to the model and
`tool_use_training/gen_1st_stage_data/build_dataset.py` trains on. A tool
class can exist under `tools/` (e.g. `asr`, `source_separation`, or any of
the heavy ML tools) without being part of this project's registered toolset;
this module refuses to execute those rather than silently running a call the
model was never taught to make. Both `agent.ToolCallingAgent` and
`testing_tool_use_benchmark/run_eval.py` drive predicted tool calls through
this same module, and both already restrict what they'll even attempt to
call to the same registered set (see `protocol.audio_to_audio_tool_names`),
so this is a belt-and-suspenders check against a hallucinated tool name
slipping through.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import TOOL_NAME_TO_CLASS  # noqa: E402
from tools import tools_registry  # noqa: E402
from tools.abstract_tool import ToolValidationError  # noqa: E402


class UnknownToolError(ValueError):
    """Raised when `tool_name` isn't part of `tools.tools_registry`'s registered toolset."""


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
    `parameters["output_audio_id"]` (the fresh id the model chose to name
    this call's own result -- see `tools.abstract_tool.Tool.to_function_schema`)
    is dropped the same way: it's consumed by the caller after this returns,
    never by the tool implementation itself.

    Raises `UnknownToolError` if `tool_name` isn't in `tools_registry`'s
    registered toolset (whether or not it's a real `Tool` subclass under
    `tools/`), or `ToolExecutionError` wrapping any validation/runtime
    failure -- callers should catch these per step rather than letting one
    bad call abort the whole chain.
    """
    registered_names = tools_registry.available_tool_names()
    if tool_name not in registered_names:
        raise UnknownToolError(
            f"'{tool_name}' is not a registered tool (have: {sorted(registered_names)})"
        )

    cls = TOOL_NAME_TO_CLASS[tool_name]
    params = dict(parameters or {})
    params.pop("audio_id", None)
    params.pop("output_audio_id", None)
    params["audio_path"] = str(input_audio_path)
    # `output_path` isn't a documented tool parameter -- the harness always decides
    # it, never the model -- so drop anything the model may have put here before
    # validating, regardless of whether this tool even needs one.
    params.pop("output_path", None)

    try:
        cls.validate_parameters(params)
        if cls.requires_output_path():
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"step{step_index}_{tool_name}.wav")
            return cls.execute(params, output_path)
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
