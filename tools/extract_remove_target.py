
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
SEPARATION_LABELS = ["vocals", 'background sound', 'background music', 'noise'] 


class _Separated:
    def __init__(self, target, residual):
        self.target = target
        self.residual = residual


def _separate_chunked(model, processor, device, label: str, waveform) -> "_Separated":
    """SAM-Audio separation of a mono (1, T) waveform at the processor's rate.

    GPU memory grows with clip length (sam-audio-large fp16: ~8.7 GB peak at 10 s; a 30 s
    clip OOMs on a 12 GB card), so clips longer than SAM_AUDIO_CHUNK_SECONDS (default 10)
    are separated chunk by chunk and concatenated. Each chunk's output is trimmed/padded to
    the chunk's input length so the result stays aligned with the input timeline.
    """
    import os

    sr = processor.audio_sampling_rate
    chunk_len = int(float(os.environ.get("SAM_AUDIO_CHUNK_SECONDS", "10")) * sr)
    total = waveform.shape[-1]
    # Chunk boundaries; a final remainder under 1 s is merged into the previous chunk, since
    # SAM-Audio's codec reflect-pads its input and fails on very short chunks ("Padding size
    # should be less than the corresponding input dimension").
    bounds = list(range(0, total, chunk_len)) + [total]
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < sr:
        bounds.pop(-2)
    targets, residuals = [], []
    for start, end in zip(bounds[:-1], bounds[1:]):
        chunk = waveform[..., start:end]
        batch = processor([label], [chunk]).to(device)
        if device.type == "cuda":
            batch.audios = batch.audios.half()
        with torch.inference_mode():
            result = model.separate(batch, predict_spans=False, reranking_candidates=1)
        if len(result.target) != 1 or len(result.residual) != 1:
            raise ToolValidationError("Unexpected separation result shape.")
        n = chunk.shape[-1]
        for out, acc in ((result.target[0], targets), (result.residual[0], residuals)):
            out = out.detach().float().reshape(-1)[:n].cpu()
            if out.shape[0] < n:
                out = torch.nn.functional.pad(out, (0, n - out.shape[0]))
            acc.append(out)
        del batch, result
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return _Separated(torch.cat(targets), torch.cat(residuals))

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
                "target_description": {
                    "type": "string",
                        "enum": SEPARATION_LABELS,
                        "description": "The sound source to extract.",
                },
            },
            "required": ["audio_path", "target_description"],
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
        label = parameters["target_description"]

        if not audio_path.exists():
            raise ToolValidationError(f"Audio file not found: {audio_path}")

        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != processor.audio_sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, processor.audio_sampling_rate)
        result = _separate_chunked(model, processor, device, label, waveform)


        label_suffix = label.replace(" ", "_")
        output_path = audio_path.parent / f"{audio_path.stem}_{label_suffix}_extracted.wav"
        cls._save_wav(output_path, result.target, processor.audio_sampling_rate)

        return {
            "audio_path": str(audio_path),
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
                "target_description": {
                    "type": "string",
                    "enum": SEPARATION_LABELS,
                    "description": "The sound source to remove.",
                },
            },
            "required": ["audio_path", "target_description"],
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
        label = parameters["target_description"]

        if not audio_path.exists():
            raise ToolValidationError(f"Audio file not found: {audio_path}")

        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != processor.audio_sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, processor.audio_sampling_rate)
        result = _separate_chunked(model, processor, device, label, waveform)


        label_suffix = label.replace(" ", "_")
        output_path = audio_path.parent / f"{audio_path.stem}_{label_suffix}_removed.wav"
        cls._save_wav(output_path, result.residual, processor.audio_sampling_rate)

        return {
            "audio_path": str(audio_path),
            "target_description": label,
            "status": "success",
            "output_path": str(output_path),
            "message": "Source removal completed using SAMAudio.",
        }
