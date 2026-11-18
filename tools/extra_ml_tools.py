"""Extra ML audio tools (added 2026-11-07 for a small UQ experiment).

All three run in one conda env (`extra_tools`, see tools/setup_local_heavy_envs.sh) through the
persistent heavy-tool server, write a mono PCM16 WAV to `output_path`, and keep the input's
duration (so timestamps in the question stay valid):

- music_source_separation  Demucs htdemucs (Defossez et al.): split into drums / bass / other /
                           vocals; `stem` keeps one of them, or "no_vocals" (everything but vocals).
- speech_enhance           ClearerVoice-Studio MossFormer2_SE_48K speech enhancement (a second
                           denoiser family next to DeepFilterNet's human_voice_enhance).
- vad_keep_speech          Silero VAD: mute everything outside detected speech. Clips with no
                           detected speech are rejected (the tool doesn't apply to them).

Models load lazily and are cached per process. Weights download to EXTRA_TOOLS_MODEL_DIR
(default: $DATA_ROOT/models/extra_tools) instead of the working directory.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .abstract_tool import Tool, ToolValidationError

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

_CACHE: Dict[str, Any] = {}


def _model_dir() -> Path:
    d = os.environ.get("EXTRA_TOOLS_MODEL_DIR")
    if not d:
        d = str(Path(os.environ.get("DATA_ROOT", Path.home() / ".cache")) / "models" / "extra_tools")
    Path(d).mkdir(parents=True, exist_ok=True)
    return Path(d)


def _load_mono(path: str, target_sr: int | None = None):
    if sf is None:
        raise ToolValidationError("soundfile is required")
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if target_sr and sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio.astype(np.float32), sr


def _save(path: str, audio: np.ndarray, sr: int, n_samples: int) -> None:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) > n_samples:
        audio = audio[:n_samples]
    elif len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)))
    sf.write(path, np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")


class _ExtraTool(Tool):
    @classmethod
    def requires_output_path(cls) -> bool:
        return True

    @classmethod
    def execute(cls, parameters: Dict[str, Any]) -> Dict[str, Any]:
        return cls.execute_batch([parameters])[0]

    @classmethod
    def execute_batch(cls, batch_parameters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for item in batch_parameters:
            params = dict(item)
            output_path = params.pop("output_path", None)
            if not output_path:
                raise ToolValidationError("Missing required 'output_path'.")
            cls.validate_parameters(params)
            if not Path(params["audio_path"]).exists():
                raise ToolValidationError(f"Audio file not found: {params['audio_path']}")
            cls._run(params, output_path)
            results.append({"audio_path": params["audio_path"], "status": "success",
                            "output_path": str(output_path), "message": f"{cls.name()} done"})
        return results

    @classmethod
    def _run(cls, params: Dict[str, Any], output_path: str) -> None:  # pragma: no cover
        raise NotImplementedError


class MusicSourceSeparationTool(_ExtraTool):
    STEMS = ["vocals", "no_vocals", "drums", "bass", "other"]

    @classmethod
    def name(cls) -> str:
        return "music_source_separation"

    @classmethod
    def description(cls) -> str:
        return ("Separate music/mixed audio with Demucs (htdemucs) and keep one stem: vocals, drums, "
                "bass, other, or no_vocals (everything except vocals).")

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {"type": "object",
                "properties": {"audio_path": {"type": "string"},
                               "stem": {"type": "string", "enum": cls.STEMS}},
                "required": ["audio_path", "stem"]}

    @classmethod
    def _model(cls):
        if "demucs" not in _CACHE:
            import torch

            torch.hub.set_dir(str(_model_dir() / "torch_hub"))
            from demucs.pretrained import get_model

            model = get_model("htdemucs")
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _CACHE["demucs"] = (model.to(device), device)
        return _CACHE["demucs"]

    @classmethod
    def _run(cls, params, output_path):
        import torch
        from demucs.apply import apply_model

        model, device = cls._model()
        orig, orig_sr = _load_mono(params["audio_path"])
        audio, _ = _load_mono(params["audio_path"], model.samplerate)
        wav = torch.from_numpy(audio)[None].repeat(model.audio_channels, 1)  # mono -> stereo
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / (ref.std() + 1e-8)
        with torch.inference_mode():
            sources = apply_model(model, wav[None].to(device), device=device, split=True, overlap=0.25,
                                  progress=False)[0]
        sources = sources * (ref.std() + 1e-8) + ref.mean()
        names = list(model.sources)
        if params["stem"] == "no_vocals":
            out = sum(sources[i] for i, n in enumerate(names) if n != "vocals")
        else:
            out = sources[names.index(params["stem"])]
        out = out.mean(0).cpu().numpy()
        n_out = int(round(len(orig) / orig_sr * model.samplerate))
        _save(output_path, out, model.samplerate, n_out)


class SpeechEnhanceTool(_ExtraTool):
    MODEL = "MossFormer2_SE_48K"

    @classmethod
    def name(cls) -> str:
        return "speech_enhance"

    @classmethod
    def description(cls) -> str:
        return "Enhance speech / suppress noise with ClearerVoice MossFormer2_SE_48K (48 kHz output)."

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {"type": "object", "properties": {"audio_path": {"type": "string"}}, "required": ["audio_path"]}

    @classmethod
    def _model(cls):
        if "clearvoice" not in _CACHE:
            # ClearerVoice downloads its checkpoints relative to the working directory.
            cwd = os.getcwd()
            os.chdir(_model_dir())
            try:
                from clearvoice import ClearVoice

                _CACHE["clearvoice"] = ClearVoice(task="speech_enhancement", model_names=[cls.MODEL])
            finally:
                os.chdir(cwd)
        return _CACHE["clearvoice"]

    @classmethod
    def _run(cls, params, output_path):
        cv = cls._model()
        orig, orig_sr = _load_mono(params["audio_path"])
        cwd = os.getcwd()
        os.chdir(_model_dir())
        try:
            out = cv(input_path=params["audio_path"], online_write=False)
        finally:
            os.chdir(cwd)
        if isinstance(out, dict):
            out = next(iter(out.values()))
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        sr = 48000
        _save(output_path, out, sr, int(round(len(orig) / orig_sr * sr)))


class VADKeepSpeechTool(_ExtraTool):
    @classmethod
    def name(cls) -> str:
        return "vad_keep_speech"

    @classmethod
    def description(cls) -> str:
        return ("Detect speech with Silero VAD and mute everything outside the speech segments "
                "(duration unchanged). Fails on audio without detectable speech.")

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {"type": "object",
                "properties": {"audio_path": {"type": "string"}, "threshold": {"type": "number"}},
                "required": ["audio_path"]}

    @classmethod
    def _model(cls):
        if "silero" not in _CACHE:
            from silero_vad import load_silero_vad

            _CACHE["silero"] = load_silero_vad()
        return _CACHE["silero"]

    @classmethod
    def _run(cls, params, output_path):
        import torch
        from silero_vad import get_speech_timestamps

        model = cls._model()
        orig, sr = _load_mono(params["audio_path"])
        a16, _ = _load_mono(params["audio_path"], 16000)
        stamps = get_speech_timestamps(torch.from_numpy(a16), model, sampling_rate=16000,
                                       threshold=float(params.get("threshold", 0.5)), return_seconds=True)
        if not stamps:
            raise ToolValidationError("no speech detected; vad_keep_speech does not apply")
        mask = np.zeros(len(orig), dtype=np.float32)
        fade = int(0.01 * sr)
        for st in stamps:
            a, b = int(st["start"] * sr), min(len(orig), int(st["end"] * sr))
            mask[a:b] = 1.0
        if fade > 1:  # 10 ms ramps instead of hard cuts
            kernel = np.ones(fade, dtype=np.float32) / fade
            mask = np.convolve(mask, kernel, mode="same")
        _save(output_path, orig * mask, sr, len(orig))


class VoiceFixerTool(_ExtraTool):
    """VoiceFixer (Liu et al., Interspeech 2022): general speech restoration -- noise,
    reverberation, clipping and low bandwidth at once, 44.1 kHz output."""

    @classmethod
    def name(cls) -> str:
        return "speech_restore_voicefixer"

    @classmethod
    def description(cls) -> str:
        return ("Restore degraded speech with VoiceFixer (removes noise and reverberation, repairs "
                "clipping, extends bandwidth to 44.1 kHz).")

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {"type": "object", "properties": {"audio_path": {"type": "string"}}, "required": ["audio_path"]}

    @classmethod
    def _model(cls):
        if "voicefixer" not in _CACHE:
            from voicefixer import VoiceFixer

            _CACHE["voicefixer"] = VoiceFixer()
        return _CACHE["voicefixer"]

    @classmethod
    def _run(cls, params, output_path):
        import tempfile

        import torch

        vf = cls._model()
        orig, orig_sr = _load_mono(params["audio_path"])
        with tempfile.TemporaryDirectory() as tmp:
            out_tmp = os.path.join(tmp, "out.wav")
            vf.restore(input=params["audio_path"], output=out_tmp, cuda=torch.cuda.is_available(), mode=0)
            out, sr = _load_mono(out_tmp)
        _save(output_path, out, sr, int(round(len(orig) / orig_sr * sr)))


class DereverbWPETool(_ExtraTool):
    """Weighted Prediction Error dereverberation (Nakatani et al.; nara_wpe, Drude et al. 2018).
    Statistical, not learned: removes late reverberation only, no denoising."""

    @classmethod
    def name(cls) -> str:
        return "dereverb_wpe"

    @classmethod
    def description(cls) -> str:
        return "Remove reverberation with WPE (weighted prediction error) dereverberation; no denoising."

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {"type": "object",
                "properties": {"audio_path": {"type": "string"}, "taps": {"type": "number"}},
                "required": ["audio_path"]}

    @classmethod
    def _run(cls, params, output_path):
        import math

        from nara_wpe.utils import istft, stft
        from nara_wpe.wpe import wpe

        y, sr = _load_mono(params["audio_path"])
        size = 2 ** int(round(math.log2(sr * 0.032)))  # ~32 ms window at any sample rate
        shift = size // 4
        Y = stft(y[None, :].astype(np.float64), size=size, shift=shift).transpose(2, 0, 1)  # (F, D=1, T)
        Z = wpe(Y, taps=int(params.get("taps", 10)), delay=3, iterations=3, statistics_mode="full")
        z = istft(Z.transpose(1, 2, 0), size=size, shift=shift)[0]
        _save(output_path, z, sr, len(y))
