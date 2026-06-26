from __future__ import annotations

import wave
from pathlib import Path
from typing import Dict, Any

from .abstract_tool import Tool, ToolValidationError


class ClippingTool(Tool):
    @classmethod
    def name(cls) -> str:
        return "clipping"

    @classmethod
    def description(cls) -> str:
        return (
            "Extract a WAV audio segment from a source file between specified "
            "HH:MM:SS.mmm begin and end timestamps."
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
            },
            "required": ["audio_path", "audio_begin", "audio_end"],
        }

    @classmethod
    def execute(cls, parameters: Dict[str, Any]) -> Dict[str, Any]:
        cls.validate_parameters(parameters)

        audio_path = parameters["audio_path"]
        audio_begin = parameters["audio_begin"]
        audio_end = parameters["audio_end"]

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise ToolValidationError(f"Audio file not found: {audio_path}")

        if audio_path.suffix.lower() != ".wav":
            raise ToolValidationError("Clipping currently supports only WAV files.")

        begin_seconds = cls._parse_timestamp(audio_begin)
        end_seconds = cls._parse_timestamp(audio_end)
        if end_seconds <= begin_seconds:
            raise ToolValidationError("audio_end must be greater than audio_begin.")

        with wave.open(str(audio_path), "rb") as input_wav:
            n_channels = input_wav.getnchannels()
            sampwidth = input_wav.getsampwidth()
            framerate = input_wav.getframerate()
            n_frames = input_wav.getnframes()

            start_frame = int(begin_seconds * framerate)
            end_frame = int(end_seconds * framerate)

            if start_frame < 0 or end_frame > n_frames:
                raise ToolValidationError(
                    f"Requested clip range [{audio_begin}, {audio_end}] is out of bounds for file with duration {n_frames / framerate:.3f}s."
                )

            input_wav.setpos(start_frame)
            frames = input_wav.readframes(end_frame - start_frame)

        clip_filename = (
            f"{audio_path.stem}_{audio_begin.replace(':', '-')}_{audio_end.replace(':', '-')}{audio_path.suffix}"
        )
        clip_path = audio_path.parent / clip_filename

        with wave.open(str(clip_path), "wb") as output_wav:
            output_wav.setnchannels(n_channels)
            output_wav.setsampwidth(sampwidth)
            output_wav.setframerate(framerate)
            output_wav.writeframes(frames)

        return {
            "audio_path": str(audio_path),
            "audio_begin": audio_begin,
            "audio_end": audio_end,
            "status": "success",
            "clip_path": str(clip_path),
            "message": "Clipping operation extracted the requested WAV segment successfully.",
        }

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
