from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict


def parse_coefficients(value: str) -> Any:
    """Parse coefficients input from JSON, file path, or comma-separated numbers."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        candidate = Path(value)
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        if "," in value:
            items = [item.strip() for item in value.split(",") if item.strip()]
            if all(re.fullmatch(r"-?\d+(?:\.\d+)?", item) for item in items):
                return [float(item) if "." in item else int(item) for item in items]

        return value


_TOOL_MODULES = {
    "asr": ("tools.asr", "ASRTool"),
    "clipping": ("tools.clipping", "ClippingTool"),
    "denoise": ("tools.denoise", "DenoiseTool"),
    "amplitude_normalize": ("tools.normalize", "AmplitudeNormalizeTool"),
    "loudness_normalize": ("tools.normalize", "LoudnessNormalizeTool"),
    "remove_dc_offset": ("tools.normalize", "DCOffsetRemovalTool"),
    "spectral_normalize": ("tools.normalize", "SpectralNormalizeTool"),
    "trim_silence": ("tools.normalize", "TrimSilenceTool"),
    "pre_emphasis": ("tools.normalize", "PreEmphasisTool"),
    "source_separation": ("tools.source_separation", "SourceSeparationTool"),
    "extract_target": ("tools.extract_remove_target", "ExtractTargetTool"),
    "remove_target": ("tools.extract_remove_target", "RemoveTargetTool"),
    "human_voice_enhance": ("tools.human_voice_enhance", "HumanVoiceEnhanceTool"),
    "super_resolution": ("tools.super_resolution", "SuperResolutionTool"),
    "pitch_shift": ("tools.pitch_time", "PitchShiftTool"),
    "time_stretch": ("tools.pitch_time", "TimeStretchTool"),
}


def get_tool_class(tool_name: str) -> type:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    if tool_name not in _TOOL_MODULES:
        raise ValueError(f"Unknown tool name: {tool_name}")

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    module_name, class_name = _TOOL_MODULES[tool_name]
    module_path = repo_root / "tools" / f"{module_name.split('.')[-1]}.py"
    if not module_path.exists():
        raise ImportError(f"Tool source file not found: {module_path}")

    package_name = module_name.split(".")[0]
    if package_name not in sys.modules:
        package = ModuleType(package_name)
        package.__path__ = [str(repo_root / package_name)]
        sys.modules[package_name] = package

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module '{module_name}' from '{module_path}'")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ImportError(
            f"Unable to import tool module '{module_name}'. Ensure dependencies for '{tool_name}' are installed."
        ) from exc

    try:
        tool_cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Tool class '{class_name}' not found in module '{module_name}'."
        ) from exc

    return tool_cls


def build_parameters(coefficients: Any, audio_path: str, audio_begin: str, audio_end: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "audio_path": audio_path,
        "audio_begin": audio_begin,
        "audio_end": audio_end,
    }

    if coefficients is None:
        return params

    if isinstance(coefficients, dict):
        params.update(coefficients)
    else:
        params["coefficients"] = coefficients

    return params


def write_result(result: Dict[str, Any], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a tool and save its result to tool_results.json.")
    parser.add_argument("--tool-name", required=True, help="Name of the tool to execute.")
    parser.add_argument(
        "--coefficients",
        required=False,
        help=(
            "JSON string, JSON file path, or comma-separated coefficients. "
            "If the parsed value is an object, its keys become tool parameters."
        ),
    )
    parser.add_argument(
        "--audio-path",
        default=str(Path(__file__).resolve().parent.parent / "tools_test" / "example.wav"),
        help="Path to the input WAV file."
    )
    parser.add_argument(
        "--audio-begin",
        default="00:00:00.000",
        help="Beginning timestamp for the audio segment in HH:MM:SS.mmm format.",
    )
    parser.add_argument(
        "--audio-end",
        default="00:00:03.000",
        help="Ending timestamp for the audio segment in HH:MM:SS.mmm format.",
    )
    parser.add_argument(
        "--output-file",
        default="tool_results.json",
        help="Path to write the tool result JSON.",
    )

    args = parser.parse_args()
    coefficients = parse_coefficients(args.coefficients) if args.coefficients is not None else None
    parameters = build_parameters(coefficients, args.audio_path, args.audio_begin, args.audio_end)

    tool_cls = get_tool_class(args.tool_name)

    try:
        result = tool_cls.execute(parameters)
    except Exception as exc:
        result = {
            "tool_name": args.tool_name,
            "parameters": parameters,
            "status": "error",
            "error": str(exc),
        }

    output_path = Path(args.output_file)
    write_result(result, output_path)
    print(f"Saved result to {output_path.resolve()}")

if __name__ == "__main__":
    main()
