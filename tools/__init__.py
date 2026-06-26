from typing import Dict, List, Type

from .abstract_tool import Tool, ToolValidationError
from .asr import ASRTool
from .clipping import ClippingTool
from .human_voice_enhance import HumanVoiceEnhanceTool
from .normalize import (
    AmplitudeNormalizeTool,
    DCOffsetRemovalTool,
    LoudnessNormalizeTool,
    PreEmphasisTool,
    SpectralNormalizeTool,
    TrimSilenceTool,
)
from .source_separation import SourceSeparationTool
from .super_resolution import SuperResolutionTool
from .pitch_time import PitchShiftTool, TimeStretchTool
# from .extract_remove_source import ExtractSourceTool, RemoveSourceTool
from .extract_remove_target import ExtractTargetTool, RemoveTargetTool
from .tool_execute import _TOOL_MODULES

TOOL_NAMES = sorted(_TOOL_MODULES.keys())


TOOL_CLASSES: List[Type[Tool]] = [
    ASRTool,
    ClippingTool,
    HumanVoiceEnhanceTool,
    AmplitudeNormalizeTool,
    LoudnessNormalizeTool,
    DCOffsetRemovalTool,
    SpectralNormalizeTool,
    TrimSilenceTool,
    PreEmphasisTool,
    SourceSeparationTool,
    ExtractTargetTool,
    RemoveTargetTool,
    SuperResolutionTool,
    PitchShiftTool,
    TimeStretchTool,
]


def _format_parameter_schema(schema: Dict[str, Dict[str, object]]) -> List[str]:
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    if not properties:
        return ["- None"]

    formatted: List[str] = []
    for name in sorted(properties):
        spec = properties[name]
        param_type = spec.get("type", "any")
        details = [f"{name} ({param_type})"]

        if name in required:
            details.append("required")
        else:
            details.append("optional")

        if spec.get("format"):
            details.append(f"format={spec['format']}")

        if spec.get("enum") is not None:
            details.append(f"allowed={spec['enum']}")

        if spec.get("description"):
            details.append(str(spec["description"]))

        formatted.append("- " + ", ".join(details))

    return formatted


def generate_tool_descriptions(tool_classes: List[Type[Tool]] | None = None) -> str:
    """Generate a human-readable description of each tool and its input parameters."""
    tool_classes = tool_classes or TOOL_CLASSES
    lines: List[str] = []

    for tool_cls in tool_classes:
        lines.append(f"{tool_cls.name()}: {tool_cls.description()}")
        lines.append("Parameters:")
        lines.extend(_format_parameter_schema(tool_cls.parameter_schema()))
        lines.append("")

    return "\n".join(lines).strip()

__all__ = [
    "Tool",
    "ToolValidationError",
    "ASRTool",
    "ClippingTool",
    "HumanVoiceEnhanceTool",
    "BandPassFilterTool",
    "DynamicCompressTool",
    "HarmonicEnhanceTool",
    "SpectralEnhanceTool",
    "TemporalCorrectTool",
    "AmplitudeNormalizeTool",
    "DCOffsetRemovalTool",
    "LoudnessNormalizeTool",
    "PreEmphasisTool",
    "SpectralNormalizeTool",
    "TrimSilenceTool",
    "SourceSeparationTool",
    "ExtractSourceTool",
    "RemoveSourceTool",
    "ExtractTargetTool",
    "RemoveTargetTool",
    "SuperResolutionTool",
    "PitchShiftTool",
    "TimeStretchTool",
    "generate_tool_descriptions",
]
