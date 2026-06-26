from __future__ import annotations

import os
from tqdm import tqdm

from pathlib import Path
from typing import Dict, Any, List, Optional

from .abstract_tool import Tool, ToolValidationError

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


class SourceSeparationTool(Tool):
    @classmethod
    def name(cls) -> str:
        return "source_separation"

    @classmethod
    def description(cls) -> str:
        return (
            "Separate a WAV audio segment into a target component and residual."
            "Accepts open-ended natural language target_description (e.g., \"human voice\", \"background music\", \"bird chirping\", \"traffic noise\") to define the target sound."
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
                # "stems": {
                #     "type": "array",
                #     "items": {
                #         "type": "string",
                #         "enum": ["vocals", "drums", "bass", "guitar", "piano", "other"],
                #     },
                # },
                "target_description": {
                    "type": "string",
                    # "enum": ["vocals", "drums", "bass", "guitar", "piano", "other", "major speech", "minor speech", "singing", "rapping", "noisy environment", "quiet environment"],
                },
                "save_residual": {
                    "type": "boolean",
                    "description": "Whether to save the residual audio as a separate file. If false, only the target audio will be saved."
                }
            },
            "required": ["audio_path", "audio_begin", "audio_end"],
        }

    @classmethod
    def execute(cls, parameters: Dict[str, Any]) -> Dict[str, Any]:
        cls.validate_parameters(parameters)

        if SAMAudio is None or SAMAudioProcessor is None:
            raise ToolValidationError(
                "sam_audio is required for source separation. Install the sam_audio package or make sure it is importable."
            )

        if torch is None or torchaudio is None:
            raise ToolValidationError(
                "PyTorch and torchaudio are required for source separation. Install them first."
            )

        audio_path = parameters["audio_path"]
        audio_begin = parameters["audio_begin"]
        audio_end = parameters["audio_end"]
        # stems = parameters.get("stems") or ["vocals", "other"]
        target_description = parameters["target_description"]
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise ToolValidationError(f"Audio file not found: {audio_path}")

        begin_seconds = cls._parse_timestamp(audio_begin)
        end_seconds = cls._parse_timestamp(audio_end)
        if end_seconds <= begin_seconds:
            raise ToolValidationError("audio_end must be greater than audio_begin.")

        # if len(stems) > 2:
        #     raise ToolValidationError(
        #         "SAMAudio source separation only supports two output stems: target and residual."
        #     )

        local_model_dir = Path.cwd() / "models--facebook--sam-audio-large" / "snapshots" / "5f2cd3a9471a08c7282c06036be6893e18de8b70"
        model_source: str = str(local_model_dir) if local_model_dir.exists() else "facebook/sam-audio-large"

        model, processor, device = cls._load_model_and_processor()
        return cls._execute_single(parameters, model, processor, device)

    @classmethod
    def execute_batch(cls, parameters_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(parameters_list, list):
            raise ToolValidationError("parameters_list must be a list of parameter dictionaries.")

        if not parameters_list:
            return []

        if SAMAudio is None or SAMAudioProcessor is None:
            raise ToolValidationError(
                "sam_audio is required for source separation. Install the sam_audio package or make sure it is importable."
            )

        if torch is None or torchaudio is None:
            raise ToolValidationError(
                "PyTorch and torchaudio are required for source separation. Install them first."
            )

        model, processor, device = cls._load_model_and_processor()
        results: List[Dict[str, Any]] = []
        for parameters in tqdm(parameters_list):
            cls.validate_parameters(parameters)
            results.append(cls._execute_single(parameters, model, processor, device))

        return results

    @classmethod
    def _load_model_and_processor(cls):
        local_model_dir = Path.cwd() / "models--facebook--sam-audio-large" / "snapshots" / "5f2cd3a9471a08c7282c06036be6893e18de8b70"
        if local_model_dir.exists():
            model_source = str(local_model_dir)
            model_kwargs = {}
        else:
            model_source = "facebook/sam-audio-large"
            model_kwargs = {
                "cache_dir": "/work/u1501463/"
            }

        model = SAMAudio.from_pretrained(model_source, **model_kwargs)
        processor = SAMAudioProcessor.from_pretrained(model_source)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            model = model.half()

        model = model.eval().to(device)

        return model, processor, device

    @classmethod
    def _execute_single(
        cls,
        parameters: Dict[str, Any],
        model,
        processor,
        device: torch.device,
    ) -> Dict[str, Any]:
        audio_path = parameters["audio_path"]
        audio_begin = parameters["audio_begin"]
        audio_end = parameters["audio_end"]
        target_description = parameters.get("target_description", "")
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise ToolValidationError(f"Audio file not found: {audio_path}")

        begin_seconds = cls._parse_timestamp(audio_begin)
        end_seconds = cls._parse_timestamp(audio_end)
        if end_seconds <= begin_seconds:
            raise ToolValidationError("audio_end must be greater than audio_begin.")

        audio_tensor = cls._load_audio_segment(audio_path, begin_seconds, end_seconds, processor.audio_sampling_rate)
        batch = processor([target_description], [audio_tensor])
        batch = batch.to(device)

        if device.type == "cuda":
            batch.audios = batch.audios.half()

        with torch.inference_mode():
            result = model.separate(batch, predict_spans=False, reranking_candidates=1)

        if len(result.target) != 1 or len(result.residual) != 1:
            raise ToolValidationError("Unexpected separation result shape.")

        mapping = {
            "target": result.target[0],
            "residual": result.residual[0],
        }

        separated_files: Dict[str, str] = {}
        # for stem in ["target", "residual"]:
        #     if stem not in mapping:
        #         raise ToolValidationError(
        #             f"Stem '{stem}' is not supported. SAMAudio can only provide 'vocals' and 'other'."
        #         )
        for stem in ["target", "residual"]:
            if stem == 'residual' and not parameters.get('save_residual', False):
                continue
            target_description_suffix = target_description.replace(" ", "_") if target_description else stem
            output_path = audio_path.parent / f"{audio_path.stem}_{audio_begin.replace(':', '-')}_{audio_end.replace(':', '-')}_{target_description_suffix}_{stem}.wav"
            cls._save_wav(output_path, mapping[stem], processor.audio_sampling_rate)
            separated_files[stem] = str(output_path)

        return {
            "audio_path": str(audio_path),
            "audio_begin": audio_begin,
            "audio_end": audio_end,
            'target_description': target_description,
            'save_residual': parameters.get('save_residual', False),
            "status": "success",
            "separated_files": separated_files,
            "message": "Source separation completed using SAMAudio.",
        }

    @classmethod
    def _load_audio_segment(
        cls,
        path: Path,
        begin_seconds: float,
        end_seconds: float,
        target_sr: int,
    ) -> torch.Tensor:
        waveform, sr = torchaudio.load(str(path))
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            sr = target_sr

        start_frame = int(begin_seconds * sr)
        end_frame = int(end_seconds * sr)
        if start_frame < 0 or end_frame > waveform.size(-1):
            start_frame = max(0, start_frame)
            end_frame = min(waveform.size(-1), end_frame)
            # raise ToolValidationError(
            #     f"Requested range [{begin_seconds:.3f}, {end_seconds:.3f}] is out of bounds for file duration {waveform.size(-1) / sr:.3f}s."
            # )

        return waveform[:, start_frame:end_frame]

    @staticmethod
    def _save_wav(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
        torchaudio.save(str(path), wav.cpu().float().unsqueeze(0), sample_rate)

    @staticmethod
    def _parse_timestamp(value: str) -> float:
        try:
            time_text, millis_text = value.split(".")
            hours, minutes, seconds = [int(part) for part in time_text.split(":")]
            millis = int(millis_text.ljust(3, "0")[:3])
        except ValueError as exc:
            raise ToolValidationError(
                f"Invalid timestamp format '{value}'. Expected HH:MM:SS.mmm"
            ) from exc

        return hours * 3600 + minutes * 60 + seconds + millis / 1000.0
