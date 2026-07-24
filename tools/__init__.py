import importlib
from typing import Dict, List, Type

from .abstract_tool import Tool, ToolValidationError
from ._tool_table import TOOL_MODULES

TOOL_NAMES: List[str] = sorted(TOOL_MODULES.keys())

# Single source of truth for tool_name -> class, built from `_tool_table.TOOL_MODULES`
# (see that file's docstring for why this stays data-driven instead of one-off
# `from .xxx import YyyTool` lines: it's the same table `tools/tool_execute.py`
# uses for its own lazy, per-subprocess loading, so the two can never drift again).
TOOL_NAME_TO_CLASS: Dict[str, Type[Tool]] = {}
for _name, (_module_path, _class_name) in TOOL_MODULES.items():
    _module = importlib.import_module(_module_path)
    TOOL_NAME_TO_CLASS[_name] = getattr(_module, _class_name)
del _name, _module_path, _class_name, _module

TOOL_CLASSES: List[Type[Tool]] = [TOOL_NAME_TO_CLASS[name] for name in TOOL_NAMES]


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


def tool_function_schemas(tool_classes: List[Type[Tool]] | None = None) -> List[Dict[str, object]]:
    """Model-facing function-calling schemas for each tool (see `Tool.to_function_schema`).

    Sibling of `generate_tool_descriptions` -- that one renders a hand-written
    prose catalogue (used by the DCASE/MCQ prompt pipeline in `prompts/` and
    `audio_edit/editor.py`); this returns structured schemas for callers that
    render their own wire convention on top (see `tool_call_formats.py`,
    consumed by `interface.protocol.build_system_prompt`).
    """
    tool_classes = tool_classes or TOOL_CLASSES
    return [tool_cls.to_function_schema() for tool_cls in tool_classes]


__all__ = [
    "Tool",
    "ToolValidationError",
    "TOOL_NAMES",
    "TOOL_CLASSES",
    "TOOL_NAME_TO_CLASS",
    "generate_tool_descriptions",
    "tool_function_schemas",
]
