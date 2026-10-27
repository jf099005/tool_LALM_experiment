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

    def diagnose(self, raw_text: str) -> Dict[str, Any]:
        """Tell apart "no tool call here" from "the wrapper tokens are broken
        but the JSON underneath is fine" -- a distinction `parse_turn` can't
        make, since a broken wrapper means it never finds the JSON at all and
        falls through to its generic failure/`done` case.

        Returns `{"tag_ok": bool, "content": Optional[dict]}`, where
        `content` (when not `None`) is `{"tool_name": ..., "parameters":
        {...}}` recovered from anywhere in `raw_text` regardless of wrapper
        correctness. Formats with no wrapper-token convention to get wrong
        (e.g. `LegacyJSONToolCallFormat`) keep the base implementation, which
        reports `tag_ok=True` unconditionally -- there's no token-formatting
        failure mode to flag for them.
        """
        return {"tag_ok": True, "content": None}


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

    def diagnose(self, raw_text: str) -> Dict[str, Any]:
        tag_ok = self._TOOL_CALL_RE.search(raw_text) is not None
        content = None
        if not tag_ok:
            # Only bother recovering content when the strict tags didn't
            # match -- when they did, `parse_turn` already has the answer.
            content = self._extract_lenient_call(raw_text)
        return {"tag_ok": tag_ok, "content": content}

    @staticmethod
    def _extract_lenient_call(raw_text: str) -> Optional[Dict[str, Any]]:
        """Best-effort recovery of a `{"name": ..., "arguments": {...}}`
        payload from anywhere in `raw_text`, tolerating a missing, truncated,
        or typo'd `<tool_call>`/`</tool_call>` wrapper (e.g. `<\\tool_call>`).
        Scans every balanced-brace `{...}` substring rather than a single
        greedy regex, since the wrapper tokens/surrounding prose aren't valid
        JSON and a naive `{.*}` match would swallow them into the candidate.
        """
        for candidate in QwenToolCallFormat._iter_json_objects(raw_text):
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or "name" not in obj:
                continue
            parameters = obj.get("arguments")
            if not isinstance(parameters, dict):
                continue
            return {"tool_name": obj["name"], "parameters": dict(parameters)}
        return None

    @staticmethod
    def _iter_json_objects(text: str):
        """Yield each top-level balanced-brace `{...}` substring in `text`."""
        start = None
        depth = 0
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        yield text[start:i + 1]
                        start = None


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
