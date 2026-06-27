"""Execute a model-predicted tool call against a real audio file.

`tool_registry.REGISTRY[name].apply(...)` (used by build_dataset.py) *invents*
random parameters to synthesize ground truth -- it's not usable to replay an
arbitrary parameter dict that a model produced. This module calls the
underlying `tools.abstract_tool.Tool` subclasses (or the manual audio_edit
ops) directly with whatever parameters the model output, so we can tell
whether the model's own tool call is valid and runs.

Deliberately imports the `Tool` subclasses itself rather than reaching into
`tool_registry`'s internals beyond `REGISTRY`/`available_tool_names()`/
`_finalize` -- `tool_registry.py` lives outside this folder and has changed
shape across this project before (e.g. losing/gaining a `TOOL_NAME_TO_CLASS`
mapping), so this module keeps its own independent map of tool_name -> Tool
class to avoid silently breaking every prediction if that internal drifts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generate.gen_tool_usage_QA import tool_registry as tr  # noqa: E402
from tools.abstract_tool import ToolValidationError  # noqa: E402
from tools.clipping import ClippingTool  # noqa: E402
from tools.denoise_old import DenoiseTool  # noqa: E402
from tools.normalize import (  # noqa: E402
    AmplitudeNormalizeTool,
    DCOffsetRemovalTool,
    LoudnessNormalizeTool,
    PreEmphasisTool,
    SpectralNormalizeTool,
    TrimSilenceTool,
)
from tools.pitch_time import PitchShiftTool, TimeStretchTool  # noqa: E402


class UnknownToolError(ValueError):
    """Raised when the model names a tool that isn't in the active registry."""


class ToolExecutionError(RuntimeError):
    """Wraps any failure (validation or runtime) while executing a predicted tool call."""


_CLASSED_TOOLS = {
    "clipping": ClippingTool,
    "denoise": DenoiseTool,
    "amplitude_normalize": AmplitudeNormalizeTool,
    "loudness_normalize": LoudnessNormalizeTool,
    "remove_dc_offset": DCOffsetRemovalTool,
    "pre_emphasis": PreEmphasisTool,
    "spectral_normalize": SpectralNormalizeTool,
    "trim_silence": TrimSilenceTool,
    "pitch_shift": PitchShiftTool,
    "time_stretch": TimeStretchTool,
}

try:
    from tools.human_voice_enhance import HumanVoiceAmplifyTool, HumanVoiceEnhanceTool  # noqa: E402

    _CLASSED_TOOLS["human_voice_enhance"] = HumanVoiceEnhanceTool
    _CLASSED_TOOLS["human_voice_amplify"] = HumanVoiceAmplifyTool
except ImportError:
    pass

try:
    from tools.super_resolution import SuperResolutionTool  # noqa: E402

    _CLASSED_TOOLS["super_resolution"] = SuperResolutionTool
except ImportError:
    pass

try:
    from tools.extract_remove_target import ExtractTargetTool, RemoveTargetTool  # noqa: E402

    _CLASSED_TOOLS["extract_target"] = ExtractTargetTool
    _CLASSED_TOOLS["remove_target"] = RemoveTargetTool
except ImportError:
    pass

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

    `parameters["audio_path"]` (whatever placeholder string the model emitted,
    e.g. "<AUDIO_A>") is ignored and overwritten with the real current audio
    path -- the placeholder is purely a textual convention from training data,
    not something tools understand.

    Returns the path the output audio was written to. Raises UnknownToolError
    or ToolExecutionError on any failure; callers should catch these per step
    rather than letting one bad step abort the whole sample.
    """
    if tool_name not in tr.REGISTRY:
        raise UnknownToolError(f"'{tool_name}' is not an available tool (have: {known_tool_names()})")

    if not Path(current_audio_path).exists():
        raise ToolExecutionError(f"Current audio file missing: {current_audio_path}")

    params = dict(parameters or {})
    params["audio_path"] = str(current_audio_path)

    try:
        if tool_name in _CLASSED_TOOLS:
            return _execute_classed_tool(tool_name, params, output_path)
        if tool_name in _MANUAL_OP_ATTRS:
            return _execute_manual_op(tool_name, params, output_path)
        raise ToolExecutionError(f"'{tool_name}' has no execution path wired up in tool_executor.py")
    except ToolValidationError as exc:
        raise ToolExecutionError(f"{tool_name} validation failed: {exc}") from exc
    except (UnknownToolError, ToolExecutionError):
        raise
    except Exception as exc:  # noqa: BLE001 - surface any backend failure as an execution failure
        raise ToolExecutionError(f"{tool_name} raised {type(exc).__name__}: {exc}") from exc


def _execute_classed_tool(tool_name: str, params: Dict[str, Any], output_path: Path) -> Path:
    cls = _CLASSED_TOOLS[tool_name]
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
