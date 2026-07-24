"""Pluggable tool-calling wire conventions.

Different target model families use different textual conventions for how a
system prompt advertises available tools, how an assistant turn expresses a
tool call, and how "no more calls" is signaled. `ToolCallFormat` factors that
convention out of the rest of the agent/data-generation code (`tools/`,
`interface/protocol.py`, `tool_use_training/gen_1st_stage_data/build_dataset.py`)
so swapping to a differently-trained model family later is a matter of adding
one subclass and selecting it by name (`--tool-call-format`), rather than
editing every call site.

`QwenToolCallFormat` reproduces Qwen2.5/Qwen3's own Hermes-style function
calling convention -- <tools>/<tool_call> -- as documented in their public
chat_template (verified locally against the Qwen3-Omni chat_template.json
snapshot, and matching ms-swift's own `swift.agent_template.hermes` module,
which is the default `agent_template` for every Qwen `swift`-backend
template). Tool *results* are deliberately left unwrapped by this module: for
training data consumed by ms-swift (the default `template_backend='swift'`)
and for `interface.engine.SwiftEngine` inference, ms-swift's own hermes
agent_template already wraps a `role: "tool"` message's raw content in
`<tool_response></tool_response>` at tokenize time -- wrapping it again here
would double-wrap. `interface.engine.VLLMEngine` (the one path with no such
automatic layer, since it talks to a raw checkpoint with its own hand-rolled
ChatML renderer) does that wrapping itself, right before rendering.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ToolCallFormat(ABC):
    """One model family's wire convention for the tool-calling protocol."""

    name: str

    @abstractmethod
    def render_tools_preamble(self, tool_schemas: List[Dict[str, Any]]) -> str:
        """Render the tool-catalogue section of the system prompt."""

    @abstractmethod
    def render_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Render one assistant turn's tool-call content.

        `arguments` is the full flat argument dict the model must produce,
        including protocol-level keys like `audio_id` / `output_audio_id`
        alongside the tool's own parameters -- there's no separate channel
        for those in a function-calling schema, so they travel as ordinary
        arguments (see `tools.abstract_tool.Tool.to_function_schema`).
        """

    @abstractmethod
    def render_done(self) -> str:
        """Render the closing assistant turn's content once the task is done."""

    @abstractmethod
    def parse_turn(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """Parse one raw assistant turn.

        Returns `{"tool_name": ..., "parameters": {...}}` for a call (with
        `output_audio_id` folded into `parameters` if the model gave one),
        `{"done": True}` if the turn signals completion, or `None` if it's
        unparseable garbage.
        """


class QwenToolCallFormat(ToolCallFormat):
    """Qwen2.5/Qwen3's official Hermes-style function-calling convention."""

    name = "qwen"

    _TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

    def render_tools_preamble(self, tool_schemas: List[Dict[str, Any]]) -> str:
        tool_lines = "\n".join(json.dumps(schema, ensure_ascii=False) for schema in tool_schemas)
        return (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n"
            f"{tool_lines}\n"
            "</tools>\n\n"
            "For each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>"
        )

    def render_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        payload = json.dumps({"name": tool_name, "arguments": arguments}, ensure_ascii=False)
        return f"<tool_call>\n{payload}\n</tool_call>"

    def render_done(self) -> str:
        return "Task complete."

    def parse_turn(self, raw_text: str) -> Optional[Dict[str, Any]]:
        match = self._TOOL_CALL_RE.search(raw_text)
        if match is None:
            # No <tool_call> block -- an ordinary reply, i.e. the model is done.
            return {"done": True}
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict) or "name" not in obj:
            return None
        parameters = obj.get("arguments") or {}
        if not isinstance(parameters, dict):
            return None
        return {"tool_name": obj["name"], "parameters": dict(parameters)}


class LegacyJSONToolCallFormat(ToolCallFormat):
    """The project's original hand-rolled flat-JSON convention.

    Kept as a selectable option (rather than deleted) so a future non-Qwen
    model that wasn't trained on Qwen's own tags can still use a convention
    without touching any call site -- just pass `--tool-call-format legacy`
    (or add another `ToolCallFormat` subclass alongside it).
    """

    name = "legacy"

    def render_tools_preamble(self, tool_schemas: List[Dict[str, Any]]) -> str:
        lines = []
        for schema in tool_schemas:
            fn = schema["function"]
            lines.append(f"- {fn['name']}: {fn['description']}")
        return "Available tools:\n" + "\n".join(lines)

    def render_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        arguments = dict(arguments)
        output_audio_id = arguments.pop("output_audio_id", None)
        payload = {"tool_name": tool_name, "parameters": arguments}
        if output_audio_id is not None:
            payload["output_audio_id"] = output_audio_id
        return json.dumps(payload, ensure_ascii=False)

    def render_done(self) -> str:
        return json.dumps({"done": True})

    def parse_turn(self, raw_text: str) -> Optional[Dict[str, Any]]:
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        if obj.get("done"):
            return {"done": True}
        if "tool_name" not in obj:
            return None
        parameters = dict(obj.get("parameters") or {})
        if obj.get("output_audio_id"):
            parameters["output_audio_id"] = obj["output_audio_id"]
        return {"tool_name": obj["tool_name"], "parameters": parameters}


TOOL_CALL_FORMATS: Dict[str, ToolCallFormat] = {
    "qwen": QwenToolCallFormat(),
    "legacy": LegacyJSONToolCallFormat(),
}


def get_tool_call_format(tool_call_format: str) -> ToolCallFormat:
    try:
        return TOOL_CALL_FORMATS[tool_call_format]
    except KeyError:
        raise ValueError(
            f"Unknown tool-call format '{tool_call_format}'. Available: {sorted(TOOL_CALL_FORMATS)}"
        ) from None
