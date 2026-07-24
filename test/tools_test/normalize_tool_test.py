from pathlib import Path
import sys
import wave

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sam-audio"))

import numpy as np
from tools.normalize import (
    AmplitudeNormalizeTool,
    DCOffsetRemovalTool,
    LoudnessNormalizeTool,
    PreEmphasisTool,
    SpectralNormalizeTool,
    TrimSilenceTool,
)


def _save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    audio = np.clip(audio, -1.0, 1.0)
    int_samples = (audio * np.iinfo(np.int16).max).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int_samples.tobytes())


def main() -> None:
    print("Testing normalization tools...")
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.sin(2 * np.pi * 880 * t)
    test_audio = np.concatenate([np.zeros(1600), tone + 0.1, np.zeros(1600)])

    # source_wav = repo_root / "example.wav"
    # if not source_wav.exists():
    #     raise FileNotFoundError(f"Expected input file not found: {source_wav}")

    source_wav = Path.cwd() / "example.wav"
    if not source_wav.exists():
        raise FileNotFoundError(f"Expected input file not found: {source_wav}")

    tools = [
        (AmplitudeNormalizeTool, {"audio_path": str(source_wav), "audio_begin": "00:00:00.000", "audio_end": "00:00:01.000", "target_level": 0.8, "method": "peak"}),
        (LoudnessNormalizeTool, {"audio_path": str(source_wav), "audio_begin": "00:00:00.000", "audio_end": "00:00:01.000", "target_lufs": -16.0}),
        (DCOffsetRemovalTool, {"audio_path": str(source_wav), "audio_begin": "00:00:00.000", "audio_end": "00:00:01.000"}),
        (SpectralNormalizeTool, {"audio_path": str(source_wav), "audio_begin": "00:00:00.000", "audio_end": "00:00:01.000", "strength": 0.4}),
        (TrimSilenceTool, {"audio_path": str(source_wav), "audio_begin": "00:00:00.000", "audio_end": "00:00:01.000", "threshold_db": 20, "frame_length": 1024, "hop_length": 256}),
        (PreEmphasisTool, {"audio_path": str(source_wav), "audio_begin": "00:00:00.000", "audio_end": "00:00:01.000", "coef": 0.95}),
    ]

    for tool_class, params in tools:
        tool_name = tool_class.name()
        print(f"\nRunning {tool_name}...")
        output_path = source_wav.parent / f"{source_wav.stem}_{tool_name}_test.wav"
        try:
            result = tool_class.execute(params, str(output_path))
        except Exception as exc:
            print(f"{tool_name} failed: {exc}")
            raise

        output_path = Path(result["output_path"])
        print(f"  status: {result['status']}")
        print(f"  output_path: {output_path}")
        assert output_path.exists(), f"Expected output file was not created for {tool_name}."
        assert result["status"] == "success"

    print("\n✓ All normalization tool tests passed!")


if __name__ == "__main__":
    main()
