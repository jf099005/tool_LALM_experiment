from __future__ import annotations

from tqdm import tqdm
import wave
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .abstract_tool import Tool, ToolValidationError

try:
    import torch
    import torchaudio
    from audiosr import build_model, super_resolution
    _AUDIOSR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _AUDIOSR_AVAILABLE = False

_MODEL_CACHE: Dict[str, Any] = {}

_OUTPUT_SAMPLE_RATE = 48000


class SuperResolutionTool(Tool):
    @classmethod
    def name(cls) -> str:
        return "super_resolution"

    @classmethod
    def description(cls) -> str:
        return (
            "Upsample and restore high-frequency detail of an audio file using the AudioSR "
            "latent diffusion model, producing a 48kHz output. Useful for improving "
            "low-quality, band-limited, or low-sample-rate recordings."
        )

    @classmethod
    def requires_output_path(cls) -> bool:
        return True

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"},
                "model_name": {
                    "type": "string",
                    "enum": ["basic", "speech"],
                    "description": "AudioSR checkpoint to use. 'speech' is tuned for speech audio.",
                },
                "ddim_steps": {
                    "type": "number",
                    "description": "Number of DDIM sampling steps. Higher is slower but may improve quality.",
                },
                "guidance_scale": {
                    "type": "number",
                    "description": "Classifier-free guidance scale.",
                },
            },
            "required": ["audio_path"],
        }

    @classmethod
    def execute(cls, parameters: Dict[str, Any], output_path: str) -> Dict[str, Any]:
        cls.validate_parameters(parameters)

        if not _AUDIOSR_AVAILABLE:
            raise ToolValidationError(
                "audiosr is not installed. Install it with: pip install audiosr"
            )

        model_name = parameters.get("model_name", "basic")
        return cls._enhance_single(parameters, output_path, cls._get_model(model_name))

    @classmethod
    def execute_batch(cls, batch_parameters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(batch_parameters, list):
            raise ToolValidationError("Batch parameters must be a list of parameter dictionaries.")

        if not _AUDIOSR_AVAILABLE:
            raise ToolValidationError(
                "audiosr is not installed. Install it with: pip install audiosr"
            )

        results: List[Dict[str, Any]] = []
        for item in tqdm(batch_parameters):
            if not isinstance(item, dict):
                raise ToolValidationError("Each batch item must be a parameter dictionary.")
            parameters = dict(item)
            output_path = parameters.pop("output_path", None)
            try:
                if not output_path:
                    raise ToolValidationError("Missing required 'output_path' for this batch item.")
                cls.validate_parameters(parameters)
                model_name = parameters.get("model_name", "basic")
                results.append(cls._enhance_single(parameters, output_path, cls._get_model(model_name)))
            except ToolValidationError as exc:
                results.append({
                    "audio_path": parameters.get("audio_path"),
                    "status": "failure",
                    "output_path": None,
                    "message": str(exc),
                })

        return results

    @classmethod
    def _enhance_single(cls, parameters: Dict[str, Any], output_path: str, model) -> Dict[str, Any]:
        audio_path = Path(parameters["audio_path"])
        if not audio_path.exists():
            raise ToolValidationError(f"Audio file not found: {audio_path}")

        model_name = parameters.get("model_name", "basic")
        ddim_steps = int(parameters.get("ddim_steps", 50))
        guidance_scale = float(parameters.get("guidance_scale", 3.5))

        info = torchaudio.info(str(audio_path))
        original_duration = info.num_frames / info.sample_rate

        waveform = super_resolution(
            model,
            str(audio_path),
            ddim_steps=ddim_steps,
            guidance_scale=guidance_scale,
        )

        audio_out = np.asarray(waveform[0, 0], dtype=np.float32)
        target_length = round(original_duration * _OUTPUT_SAMPLE_RATE)
        audio_out = cls._match_length(audio_out, target_length)
        audio_out = np.clip(audio_out, -1.0, 1.0)

        output_path = Path(output_path)
        cls._save_wav(output_path, audio_out, _OUTPUT_SAMPLE_RATE)

        return {
            "audio_path": str(audio_path),
            "model_name": model_name,
            "status": "success",
            "output_path": str(output_path),
            "message": "Super resolution completed using AudioSR.",
        }

    @classmethod
    def _get_model(cls, model_name: str):
        if model_name not in _MODEL_CACHE:
            torch.set_float32_matmul_precision("high")
            _MODEL_CACHE[model_name] = build_model(model_name=model_name)
        return _MODEL_CACHE[model_name]

    @staticmethod
    def _match_length(audio: np.ndarray, target_len: int) -> np.ndarray:
        if len(audio) > target_len:
            return audio[:target_len]
        if len(audio) < target_len:
            return np.pad(audio, (0, target_len - len(audio)))
        return audio

    @staticmethod
    def _save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
        audio_int16 = np.int16(np.round(audio * np.iinfo(np.int16).max))
        with wave.open(str(path), "wb") as output_wav:
            output_wav.setnchannels(1)
            output_wav.setsampwidth(2)
            output_wav.setframerate(sample_rate)
            output_wav.writeframes(audio_int16.tobytes())
