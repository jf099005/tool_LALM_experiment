from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'sam-audio'))

from tools.abstract_tool import ToolValidationError
from tools.extract_remove_target import RemoveTargetTool

def main() -> None:
    # print('RemoveTargetTool available:', RemoveTargetTool.name())

    audio_path = './test.wav'  # Replace with a valid WAV file path
    params = {
        'audio_path': audio_path,
        'target_description': 'noise',
        # 'save_residual': True
    }
    # param2 = {
    #     'audio_path': 'example.wav',
    #     'audio_begin': '00:00:00.000',
    #     'audio_end': '00:00:03.000',
    #     'target_description': 'music'
    # }
    
    # try:
    # result = RemoveTargetTool.execute_batch([params, param2])
    result = RemoveTargetTool.execute(params)
    print(result)
    # except Exception as exc:
    #     print('RemoveTargetTool execution failed:', exc)


if __name__ == '__main__':
    main()
