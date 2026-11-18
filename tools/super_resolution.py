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
    import audiosr.utils as _audiosr_utils
    _AUDIOSR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _AUDIOSR_AVAILABLE = False

if _AUDIOSR_AVAILABLE:
    # Upstream bug (audiosr 0.0.7, utils.lowpass_filtering_prepare_inference): for near-silent
    # input, or when the detected cutoff lands on the top STFT bin, cutoff_freq becomes 24000 Hz
    # == Nyquist at fs=48000, so scipy's filter design raises "critical frequencies must be
    # 0 < Wn < 1". Upstream's intent there is "don't lowpass", so clamp just below Nyquist.
    _upstream_lowpass = _audiosr_utils.lowpass

    def _lowpass_below_nyquist(data, highcut, fs, order=5, _type="butter"):
        return _upstream_lowpass(data, min(highcut, 0.99 * fs / 2), fs, order=order, _type=_type)

    _audiosr_utils.lowpass = _lowpass_below_nyquist

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

        audio_out = cls._run_chunked(model, audio_path, info, ddim_steps, guidance_scale)
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
    def _run_chunked(cls, model, audio_path: Path, info, ddim_steps: int, guidance_scale: float) -> np.ndarray:
        """AudioSR pads its input to a multiple of 10.24 s and runs it in one pass, so GPU
        memory grows with clip length (~10.6 GB peak at 10 s; a 30 s clip OOMs on a 12 GB
        card). Clips longer than `AUDIOSR_CHUNK_SECONDS` (default 10.24) are split into
        chunks, each super-resolved on its own, then concatenated at 48 kHz."""
        import os
        import tempfile

        chunk_seconds = float(os.environ.get("AUDIOSR_CHUNK_SECONDS", "10.24"))
        duration = info.num_frames / info.sample_rate
        if duration <= chunk_seconds:
            waveform = super_resolution(model, str(audio_path), ddim_steps=ddim_steps, guidance_scale=guidance_scale)
            return np.asarray(waveform[0, 0], dtype=np.float32)

        audio, sr = torchaudio.load(str(audio_path))
        chunk_len = int(round(chunk_seconds * sr))
        pieces = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            for i, start in enumerate(range(0, audio.shape[1], chunk_len)):
                chunk = audio[:, start:start + chunk_len]
                n = chunk.shape[1]
                # Zero-pad a short final chunk to the full chunk length: for some short
                # lengths AudioSR's UNet fails with an off-by-one latent size ("Expected
                # size 64 but got size 63"). The padding is trimmed off the output below.
                if n < chunk_len:
                    chunk = torch.nn.functional.pad(chunk, (0, chunk_len - n))
                chunk_path = Path(tmp_dir) / f"chunk{i}.wav"
                torchaudio.save(str(chunk_path), chunk, sr, encoding="PCM_S", bits_per_sample=16)
                waveform = super_resolution(model, str(chunk_path), ddim_steps=ddim_steps, guidance_scale=guidance_scale)
                out = np.asarray(waveform[0, 0], dtype=np.float32)
                pieces.append(cls._match_length(out, round(n / sr * _OUTPUT_SAMPLE_RATE)))
        return np.concatenate(pieces)

    @classmethod
    def _get_model(cls, model_name: str):
        """Cached per checkpoint, with at most one on the GPU (each is ~6 GB, so 'basic' and
        'speech' don't both fit in 12 GB with activations); the other waits in CPU RAM.
        Built on CPU, then moved: build_model(device="cuda") puts the checkpoint on the GPU
        *and* the model, ~11 GB at load time."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name not in _MODEL_CACHE:
            torch.set_float32_matmul_precision("high")
            _MODEL_CACHE[model_name] = build_model(model_name=model_name, device="cpu")
        if device.type == "cuda":
            for other, m in _MODEL_CACHE.items():
                if other != model_name and getattr(m, "device", None) == device:
                    m.to("cpu")
                    m.device = torch.device("cpu")
            torch.cuda.empty_cache()
        model = _MODEL_CACHE[model_name]
        if getattr(model, "device", None) != device:
            model.to(device)
            model.device = device
        return model

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
