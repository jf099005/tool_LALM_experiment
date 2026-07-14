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
        """
        if not isinstance(batch_parameters, list):
            raise ToolValidationError("Batch parameters must be a list of parameter dictionaries.")

        results: List[Dict[str, Any]] = []
        for parameters in batch_parameters:
            if not isinstance(parameters, dict):
                raise ToolValidationError("Each batch item must be a parameter dictionary.")
            try:
                results.append(cls.execute(parameters))
            except Exception as exc:
                results.append({
                    "audio_path": parameters.get("audio_path"),
                    "status": "failure",
                    "output_path": None,
                    "message": str(exc),
                })
        return results
