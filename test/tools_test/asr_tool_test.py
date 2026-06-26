from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'sam-audio'))

from tools.asr import ASRTool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a simple ASRTool smoke test.')
    parser.add_argument(
        '--audio-path',
        default='./example.wav',
        help='Path to a WAV audio file to transcribe.',
    )
    parser.add_argument(
        '--audio-begin',
        default='00:00:00.000',
        help='Clip begin timestamp in HH:MM:SS.mmm format.',
    )
    parser.add_argument(
        '--audio-end',
        default='00:00:03.000',
        help='Clip end timestamp in HH:MM:SS.mmm format.',
    )
    parser.add_argument(
        '--language',
        default='en',
        help='Target language code or auto-detect with `auto-detect`.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio_path)

    print('ASRTool available:', ASRTool.name())
    print('Using audio path:', audio_path)

    if not audio_path.exists():
        print('Audio file not found:', audio_path)
        return

    params = {
        'audio_path': str(audio_path),
        'audio_begin': args.audio_begin,
        'audio_end': args.audio_end,
        'language': args.language,
    }

    try:
        result = ASRTool.execute(params)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print('ASRTool execution failed:', exc)


if __name__ == '__main__':
    main()
