from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional


class ToolValidationError(ValueError):
    """Raised when a tool receives invalid parameters."""


class Tool(ABC):
    """Abstract base class for tool implementations."""

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def description(cls) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def required_parameters(cls) -> List[str]:
        schema = cls.parameter_schema()
        return schema.get("required", [])

    @classmethod
    def produces_audio(cls) -> bool:
        """Whether this tool's result is a new audio (True for every tool except
        ASR, whose result is text-only). Lets callers that must stay audio-to-audio
        only -- e.g. `interface.protocol.audio_to_audio_tool_names()` -- filter the
        catalogue without hardcoding tool names.
        """
        return True

    @classmethod
    def requires_output_path(cls) -> bool:
        """Whether callers must supply a separate `output_path` argument to `execute()`.

        True for tools with one unambiguous output file (denoise, the normalize
        family, pitch/time, voice enhance, super-resolution) -- `output_path` isn't
        a schema-validated parameter for these, it's a required second argument the
        caller (the harness, not the model) must always provide. False for tools
        whose output is auto-derived or non-file (clipping, source separation,
        extract/remove target, ASR), which keep the original `execute(parameters)`.
        """
        return False

    @classmethod
    def to_function_schema(cls) -> Dict[str, Any]:
        """Build this tool's model-facing function-calling schema (OpenAI/Qwen
        function-schema shape: `{"type": "function", "function": {...}}`).

        Swaps the harness-internal `audio_path` (the model never sees or
        produces a literal path) for the model-facing `audio_id` -- an id
        naming an already-seen audio, optional since it defaults to whatever
        the model's previous turn produced (see `interface.protocol`) -- and,
        for every tool that produces a new audio, adds `output_audio_id`: a
        fresh id the model must choose to name that call's result. Neither
        key is part of `parameter_schema()` (the *execution*-time schema
        `execute()`/`validate_parameters()` check against) since
        `interface.executor.run_tool_call` resolves and strips them before a
        real call -- they're purely a chaining convention layered on top by
        the agent protocol, not a real parameter of the underlying tool.

        `description()` itself is left untouched (it's shared with the
        path-based DCASE/MCQ prompt pipeline in `prompts/`/`audio_edit/editor.py`,
        which really does invoke tools with a literal `audio_path`) -- but
        several tools' `description()` text still says "Requires audio_path"
        (written before the audio_id-based agent protocol existed and never
        updated), which would otherwise contradict the `audio_id` property
        this method just built. Substituted for the model-facing view only.
        """
        schema = cls.parameter_schema()
        properties = {k: v for k, v in schema.get("properties", {}).items() if k != "audio_path"}
        required = [r for r in schema.get("required", []) if r != "audio_path"]
        description = cls.description().replace("audio_path", "audio_id")

        properties["audio_id"] = {
            "type": "string",
            "description": (
                "Id of an audio you've already seen (an input audio, or an output_audio_id "
                "you declared in an earlier turn). If omitted, defaults to whichever audio "
                "your previous turn most recently produced (or the sole input audio, on your "
                "first turn)."
            ),
        }
        if cls.produces_audio():
            properties["output_audio_id"] = {
                "type": "string",
                "description": (
                    "A fresh id, distinct from every id used so far, to name this call's "
                    "output audio."
                ),
            }
            required = required + ["output_audio_id"]

        return {
            "type": "function",
            "function": {
                "name": cls.name(),
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    @classmethod
    def validate_parameters(cls, parameters: Dict[str, Any]) -> None:
        schema = cls.parameter_schema()
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        missing = [name for name in required if name not in parameters]
        if missing:
            raise ToolValidationError(f"Missing required parameters: {missing}")

        for name, value in parameters.items():
            if name not in properties:
                raise ToolValidationError(f"Unexpected parameter: {name}")

            spec = properties[name]
            expected_type = spec.get("type")
            if expected_type:
                if expected_type == "array":
                    if not isinstance(value, list):
                        raise ToolValidationError(f"Parameter '{name}' must be an array")
                elif expected_type == "string":
                    if not isinstance(value, str):
                        raise ToolValidationError(f"Parameter '{name}' must be a string")

            if spec.get("format") == "HH:MM:SS.mmm":
                cls._validate_timestamp(name, value)

            if spec.get("enum") is not None and value is not None:
                if isinstance(value, list):
                    invalid = [item for item in value if item not in spec["enum"]]
                    if invalid:
                        raise ToolValidationError(
                            f"Parameter '{name}' contains invalid values: {invalid}. Allowed: {spec['enum']}"
                        )
                elif value not in spec["enum"]:
                    raise ToolValidationError(
                        f"Parameter '{name}' has invalid value '{value}'. Allowed: {spec['enum']}"
                    )

    @classmethod
    def _validate_timestamp(cls, name: str, value: Any) -> None:
        if not isinstance(value, str):
            raise ToolValidationError(f"Parameter '{name}' must be a string in HH:MM:SS.mmm format")

        try:
            datetime.strptime(value, "%H:%M:%S.%f")
        except ValueError as exc:
            raise ToolValidationError(
                f"Parameter '{name}' must be in HH:MM:SS.mmm format: {exc}"
            ) from exc

    @classmethod
    @abstractmethod
    def execute(cls, parameters: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def execute_batch(cls, batch_parameters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sequentially run `execute` over a batch, isolating per-item failures.

        Tools that can share expensive setup across a batch (e.g. one shared model
        load) override this; this default just means every `Tool` subclass works
        with `tools/tool_batch_execute.py` without requiring a bespoke override.

        Each item may carry an `output_path` key alongside its tool parameters --
        for tools where `requires_output_path()` is True, that key is required, is
        popped out before schema validation, and is passed to `execute()` as a
        separate argument rather than as a validated parameter.
        """
        if not isinstance(batch_parameters, list):
            raise ToolValidationError("Batch parameters must be a list of parameter dictionaries.")

        results: List[Dict[str, Any]] = []
        for item in batch_parameters:
            if not isinstance(item, dict):
                raise ToolValidationError("Each batch item must be a parameter dictionary.")
            parameters = dict(item)
            try:
                if cls.requires_output_path():
                    output_path = parameters.pop("output_path", None)
                    if not output_path:
                        raise ToolValidationError("Missing required 'output_path' for this batch item.")
                    results.append(cls.execute(parameters, output_path))
                else:
                    results.append(cls.execute(parameters))
            except Exception as exc:
                results.append({
                    "audio_path": parameters.get("audio_path"),
                    "status": "failure",
                    "output_path": None,
                    "message": str(exc),
                })
        return results
