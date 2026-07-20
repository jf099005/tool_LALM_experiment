"""Execute a model-predicted tool call against a real audio file.

`synthetic_registry.REGISTRY[name].apply(...)` (used by build_dataset.py) *invents*
random parameters to synthesize ground truth -- it's not usable to replay an
arbitrary parameter dict that a model produced. This module calls the
underlying `tools.abstract_tool.Tool` subclasses (or the manual audio_edit
ops) directly with whatever parameters the model output, so we can tell
whether the model's own tool call is valid and runs.

Uses `tools.TOOL_NAME_TO_CLASS` (the package's single tool_name -> Tool class
mapping, see `_tool_table.py`) rather than keeping its own independent copy --
that used to be deliberate, to avoid silently breaking every prediction if
`tool_registry`'s internal shape drifted out from under this file. Now that
both live in the same package and share one table, that drift can't happen
anymore, so the duplication bought nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from . import TOOL_NAME_TO_CLASS
from . import synthetic_registry as tr
from .abstract_tool import ToolValidationError


class UnknownToolError(ValueError):
    """Raised when the model names a tool that isn't in the active registry."""


class ToolExecutionError(RuntimeError):
    """Wraps any failure (validation or runtime) while executing a predicted tool call."""


_MANUAL_OP_ATTRS = {
    "add_noise": "_ae_add_noise",
    "pad_noise": "_ae_pad_noise",
    "insert_event": "_ae_insert_event",
}


def known_tool_names() -> list[str]:
    return tr.available_tool_names()


def execute_predicted_tool_call(
    tool_name: str,
    parameters: Dict[str, Any],
    current_audio_path: Path,
    output_path: Path,
) -> Path:
    """Run one predicted (tool_name, parameters) step on `current_audio_path`.

    `parameters["audio_id"]` (whatever id string the model emitted, e.g.
    "audio_0") is dropped and replaced with a real `audio_path` pointing at
    the actual current audio file -- the id is purely a textual convention
    from training data the model uses to refer back to a prior audio, not
    something the underlying tool implementations understand.

    Returns the path the output audio was written to. Raises UnknownToolError
    or ToolExecutionError on any failure; callers should catch these per step
    rather than letting one bad step abort the whole sample.
    """
    if tool_name not in tr.REGISTRY:
        raise UnknownToolError(f"'{tool_name}' is not an available tool (have: {known_tool_names()})")

    if not Path(current_audio_path).exists():
        raise ToolExecutionError(f"Current audio file missing: {current_audio_path}")

    params = dict(parameters or {})
    params.pop("audio_id", None)
    params["audio_path"] = str(current_audio_path)

    try:
        if tool_name in TOOL_NAME_TO_CLASS:
            return _execute_classed_tool(tool_name, params, output_path)
        if tool_name in _MANUAL_OP_ATTRS:
            return _execute_manual_op(tool_name, params, output_path)
        raise ToolExecutionError(f"'{tool_name}' has no execution path wired up in predicted_executor.py")
    except ToolValidationError as exc:
        raise ToolExecutionError(f"{tool_name} validation failed: {exc}") from exc
    except (UnknownToolError, ToolExecutionError):
        raise
    except Exception as exc:  # noqa: BLE001 - surface any backend failure as an execution failure
        raise ToolExecutionError(f"{tool_name} raised {type(exc).__name__}: {exc}") from exc


def _execute_classed_tool(tool_name: str, params: Dict[str, Any], output_path: Path) -> Path:
    cls = TOOL_NAME_TO_CLASS[tool_name]
    schema_props = cls.parameter_schema().get("properties", {})
    if "output_path" in schema_props:
        params["output_path"] = str(output_path)

    cls.validate_parameters(params)
    result = cls.execute(params)

    produced = result.get("output_path") or result.get("clip_path")
    if not produced:
        raise ToolExecutionError(f"{tool_name}.execute() returned no output/clip path: {result}")
    return tr._finalize(Path(produced), output_path)  # noqa: SLF001 - intentional reuse of the shared rename helper


def _execute_manual_op(tool_name: str, params: Dict[str, Any], output_path: Path) -> Path:
    fn = getattr(tr, _MANUAL_OP_ATTRS[tool_name], None)
    if fn is None:
        raise ToolExecutionError(f"'{tool_name}' op is unavailable in this environment (audio_edit not importable)")

    params["output_path"] = str(output_path)
    result = fn(**params)
    produced = result.get("output_path")
    if not produced:
        raise ToolExecutionError(f"{tool_name}() returned no output_path: {result}")
    return tr._finalize(Path(produced), output_path)  # noqa: SLF001
