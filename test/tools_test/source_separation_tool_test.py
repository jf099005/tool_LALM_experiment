from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'sam-audio'))

from tools.abstract_tool import ToolValidationError
from tools.source_separation import SourceSeparationTool

def main() -> None:
    print('SourceSeparationTool available:', SourceSeparationTool.name())

    audio_path = './example.wav'  # Replace with a valid WAV file path
    params = {
        'audio_path': audio_path,
        'audio_begin': '00:00:00.000',
        'audio_end': '00:00:03.000',
        'target_description': 'Ratchet and pawl sound',
        'save_residual': True
    }
    # param2 = {
    #     'audio_path': 'example.wav',
    #     'audio_begin': '00:00:00.000',
    #     'audio_end': '00:00:03.000',
    #     'target_description': 'music'
    # }
    
    # try:
    # result = SourceSeparationTool.execute_batch([params, param2])
    result = SourceSeparationTool.execute(params)
    print(result)
    # except Exception as exc:
    #     print('SourceSeparationTool execution failed:', exc)


if __name__ == '__main__':
    main()
