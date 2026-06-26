
from __future__ import annotations
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, List

from .abstract_tool import ToolValidationError
from .source_separation import SourceSeparationTool

try:
    import torch
    import torchaudio
except ImportError:  # pragma: no cover
    torch = None
    torchaudio = None

try:
    from sam_audio import SAMAudio, SAMAudioProcessor
except ImportError:  # pragma: no cover
    SAMAudio = None
    SAMAudioProcessor = None

# SEPARATION_LABELS = ["vocals", "drums", "bass", "guitar", "piano", "other", "major speech", "minor speech", "singing", "rapping", "noisy environment", "quiet environment"] 
SEPARATION_LABELS = ["vocals", 'background sound', 'background music'] 

class ExtractTargetTool(SourceSeparationTool):
    @classmethod
    def name(cls) -> str:
        return "extract_target"

    @classmethod
    def description(cls) -> str:
        labels = ", ".join(f'"{l}"' for l in SEPARATION_LABELS)
        return (
            "Extract a specific sound source from a WAV audio segment. "
            f"The label must be one of the fixed supported values: {labels}. "
            "Saves only the extracted target channel."
        )

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"},
                "audio_begin": {
                    "type": "string",
                    "format": "HH:MM:SS.mmm",
                },
                "audio_end": {
                    "type": "string",
                    "format": "HH:MM:SS.mmm",
                },
                "target_description": {
                    "type": "string",
                        "enum": SEPARATION_LABELS,
                        "description": "The sound source to extract.",
                },
            },
            "required": ["audio_path", "audio_begin", "audio_end", "target_description"],
        }

    @classmethod
    def execute(cls, parameters: Dict[str, Any]) -> Dict[str, Any]:
        cls.validate_parameters(parameters)
        if SAMAudio is None or SAMAudioProcessor is None:
            raise ToolValidationError(
                "sam_audio is required. Install the sam_audio package or make sure it is importable."
            )
        if torch is None or torchaudio is None:
            raise ToolValidationError(
                "PyTorch and torchaudio are required. Install them first."
            )
        model, processor, device = cls._load_model_and_processor()
        return cls._run(parameters, model, processor, device)

    @classmethod
    def execute_batch(cls, parameters_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(parameters_list, list):
            raise ToolValidationError("parameters_list must be a list of parameter dictionaries.")
        if not parameters_list:
            return []
        if SAMAudio is None or SAMAudioProcessor is None:
            raise ToolValidationError(
                "sam_audio is required. Install the sam_audio package or make sure it is importable."
            )
        if torch is None or torchaudio is None:
            raise ToolValidationError(
                "PyTorch and torchaudio are required. Install them first."
            )
        model, processor, device = cls._load_model_and_processor()
        results = []
        for parameters in tqdm(parameters_list):
            cls.validate_parameters(parameters)
            results.append(cls._run(parameters, model, processor, device))
        return results

    @classmethod
    def _run(cls, parameters: Dict[str, Any], model, processor, device) -> Dict[str, Any]:
        audio_path = Path(parameters["audio_path"])
        audio_begin = parameters["audio_begin"]
        audio_end = parameters["audio_end"]
        label = parameters["target_description"]

        if not audio_path.exists():
            raise ToolValidationError(f"Audio file not found: {audio_path}")

        begin_seconds = cls._parse_timestamp(audio_begin)
        end_seconds = cls._parse_timestamp(audio_end)
        if end_seconds <= begin_seconds:
            raise ToolValidationError("audio_end must be greater than audio_begin.")

        audio_tensor = cls._load_audio_segment(audio_path, begin_seconds, end_seconds, processor.audio_sampling_rate)
        batch = processor([label], [audio_tensor])
        batch = batch.to(device)

        if device.type == "cuda":
            batch.audios = batch.audios.half()

        with torch.inference_mode():
            result = model.separate(batch, predict_spans=False, reranking_candidates=1)

        if len(result.target) != 1:
            raise ToolValidationError("Unexpected separation result shape.")

        label_suffix = label.replace(" ", "_")
        output_path = audio_path.parent / (
            f"{audio_path.stem}"
            f"_{audio_begin.replace(':', '-')}"
            f"_{audio_end.replace(':', '-')}"
            f"_{label_suffix}_extracted.wav"
        )
        cls._save_wav(output_path, result.target[0], processor.audio_sampling_rate)

        return {
            "audio_path": str(audio_path),
            "audio_begin": audio_begin,
            "audio_end": audio_end,
            "target_description": label,
            "status": "success",
            "output_path": str(output_path),
            "message": "Source extraction completed using SAMAudio.",
        }


class RemoveTargetTool(SourceSeparationTool):
    @classmethod
    def name(cls) -> str:
        return "remove_target"

    @classmethod
    def description(cls) -> str:
        labels = ", ".join(f'"{l}"' for l in SEPARATION_LABELS)
        return (
            "Remove a specific sound source from a WAV audio segment. "
            f"The label must be one of the fixed supported values: {labels}. "
            "Saves the residual audio with that source removed."
        )

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"},
                "audio_begin": {
                    "type": "string",
                    "format": "HH:MM:SS.mmm",
                },
                "audio_end": {
                    "type": "string",
                    "format": "HH:MM:SS.mmm",
                },
                "target_description": {
                    "type": "string",
                    "enum": SEPARATION_LABELS,
                    "description": "The sound source to remove.",
                },
            },
            "required": ["audio_path", "audio_begin", "audio_end", "target_description"],
        }

    @classmethod
    def execute(cls, parameters: Dict[str, Any]) -> Dict[str, Any]:
        cls.validate_parameters(parameters)
        if SAMAudio is None or SAMAudioProcessor is None:
            raise ToolValidationError(
                "sam_audio is required. Install the sam_audio package or make sure it is importable."
            )
        if torch is None or torchaudio is None:
            raise ToolValidationError(
                "PyTorch and torchaudio are required. Install them first."
            )
        model, processor, device = cls._load_model_and_processor()
        return cls._run(parameters, model, processor, device)

    @classmethod
    def execute_batch(cls, parameters_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(parameters_list, list):
            raise ToolValidationError("parameters_list must be a list of parameter dictionaries.")
        if not parameters_list:
            return []
        if SAMAudio is None or SAMAudioProcessor is None:
            raise ToolValidationError(
                "sam_audio is required. Install the sam_audio package or make sure it is importable."
            )
        if torch is None or torchaudio is None:
            raise ToolValidationError(
                "PyTorch and torchaudio are required. Install them first."
            )
        model, processor, device = cls._load_model_and_processor()
        results = []
        for parameters in tqdm(parameters_list):
            cls.validate_parameters(parameters)
            results.append(cls._run(parameters, model, processor, device))
        return results

    @classmethod
    def _run(cls, parameters: Dict[str, Any], model, processor, device) -> Dict[str, Any]:
        audio_path = Path(parameters["audio_path"])
        audio_begin = parameters["audio_begin"]
        audio_end = parameters["audio_end"]
        label = parameters["target_description"]

        if not audio_path.exists():
            raise ToolValidationError(f"Audio file not found: {audio_path}")

        begin_seconds = cls._parse_timestamp(audio_begin)
        end_seconds = cls._parse_timestamp(audio_end)
        if end_seconds <= begin_seconds:
            raise ToolValidationError("audio_end must be greater than audio_begin.")

        audio_tensor = cls._load_audio_segment(audio_path, begin_seconds, end_seconds, processor.audio_sampling_rate)
        batch = processor([label], [audio_tensor])
        batch = batch.to(device)

        if device.type == "cuda":
            batch.audios = batch.audios.half()

        with torch.inference_mode():
            result = model.separate(batch, predict_spans=False, reranking_candidates=1)

        if len(result.residual) != 1:
            raise ToolValidationError("Unexpected separation result shape.")

        label_suffix = label.replace(" ", "_")
        output_path = audio_path.parent / (
            f"{audio_path.stem}"
            f"_{audio_begin.replace(':', '-')}"
            f"_{audio_end.replace(':', '-')}"
            f"_{label_suffix}_removed.wav"
        )
        cls._save_wav(output_path, result.residual[0], processor.audio_sampling_rate)

        return {
            "audio_path": str(audio_path),
            "audio_begin": audio_begin,
            "audio_end": audio_end,
            "target_description": label,
            "status": "success",
            "output_path": str(output_path),
            "message": "Source removal completed using SAMAudio.",
        }
