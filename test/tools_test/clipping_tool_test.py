from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'sam-audio'))

from tools.clipping import ClippingTool


def main() -> None:
    print('ClippingTool available:', ClippingTool.name())

    audio_path = './example.wav'  # Replace with a valid WAV file path
    params = {
        'audio_path': audio_path,
        'audio_begin': '00:00:01.000',
        'audio_end': '00:00:02.000',
    }

    try:
        result = ClippingTool.execute(params)
        print(result)
    except Exception as exc:
        print('ClippingTool execution failed:', exc)


if __name__ == '__main__':
    main()
