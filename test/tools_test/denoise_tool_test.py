from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'sam-audio'))

from tools.human_voice_enhance import DenoiseTool


def main() -> None:
    print('DenoiseTool available:', DenoiseTool.name())

    audio_path = Path('./example.wav')
    if not audio_path.exists():
        print('Audio file not found:', audio_path)
        return

    params = {
        'audio_path': str(audio_path),
        # 'audio_begin': '00:00:00.000',
        # 'audio_end': '00:00:03.000',
        # 'algorithm': 'spectral_subtraction',
        # 'noise_factor': 2.0,
        # 'sensitivity': 0.5,
    }

    try:
        results = DenoiseTool.execute_batch([params])
        result = results[0] if results else None
        # result = DenoiseTool.execute(params)
        print('Result:')
        print(result)
    except Exception as exc:
        print('DenoiseTool execution failed:', exc)


if __name__ == '__main__':
    main()
