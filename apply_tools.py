# %% [markdown]
# # Apply Tool Chains with Batch Acceleration
# 
# This notebook reads `tool_schedules.json`, merges all tool calls across problems by tool name, then runs each tool in batch using `tools/tool_batch_execute.py`.
# 
# The output is written in the same per-problem structure as `apply_tool.ipynb`:
# - `./apply_tool_results_accelerated/{problem_id}/...`
# - `./apply_tool_results_accelerated/apply_tool_summary.json`
# 
# Each execution result is annotated with the original problem id and problem metadata.
from __future__ import annotations

from argparse import ArgumentParser
import re
import subprocess
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

repo_root = Path('.').resolve()
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'sam-audio'))

# tools/-package tool names where `Tool.requires_output_path()` is True (see
# tools/abstract_tool.py) -- output_path isn't a schema parameter for these, so the
# harness (this script) must generate and inject one before batch dispatch.
TOOLS_REQUIRING_OUTPUT_PATH: set[str] = {
    "denoise", "pitch_shift", "time_stretch", "human_voice_enhance", "super_resolution",
    "amplitude_normalize", "loudness_normalize", "remove_dc_offset", "spectral_normalize",
    "trim_silence", "pre_emphasis",
}

TOOL_ENV: dict[str, str] = {
    'asr': '/home/u1501463/miniconda3/envs/Whisper/bin/python',
    'clipping': '/home/u1501463/miniconda3/envs/Whisper/bin/python',
    'denoise': '/home/u1501463/miniconda3/envs/deepfilternet/bin/python',
    'source_separation': '/home/u1501463/miniconda3/envs/sam_audio/bin/python',
    'extract_target': '/home/u1501463/miniconda3/envs/sam_audio/bin/python',
    'remove_target': '/home/u1501463/miniconda3/envs/sam_audio/bin/python',
    'human_voice_enhance': '/home/u1501463/miniconda3/envs/deepfilternet/bin/python',
    'super_resolution': '/home/u1501463/miniconda3/envs/audiosr/bin/python',
    'pitch_shift': '/home/u1501463/miniconda3/envs/Whisper/bin/python',
    'time_stretch': '/home/u1501463/miniconda3/envs/Whisper/bin/python',
}


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def find_audio_file(problem: Dict[str, Any]) -> Path:
    audio_path = problem.get('audio_path') or problem.get('audio_url')
    if not audio_path:
        raise ValueError(f'Missing audio_path or audio_url for problem id={problem.get("id")!r}')

    path = Path(audio_path)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def normalize_parameters(parameters: dict[str, Any], audio_path: Path) -> dict[str, Any]:
    normalized = dict(parameters)
    normalized['audio_path'] = str(audio_path)
    return normalized


def params_suffix(parameters: dict[str, Any]) -> str:
    parts = [str(v) for k, v in parameters.items() if k != 'audio_path']
    return re.sub(r'[^\w\-]', '_', '_'.join(parts))


def save_text_result(out_dir: Path, tool_name: str, text: str) -> Path:
    output_path = out_dir / f'tool_result_{tool_name}.txt'
    output_path.write_text(text, encoding='utf-8')
    return output_path


def save_audio_result(out_dir: Path, tool_name: str, source_path: Path, suffix: str | None = None) -> Path:
    name = f'tool_result_{tool_name}' + (f'_{suffix}' if suffix else '') + source_path.suffix
    destination = out_dir / name
    shutil.copy2(source_path, destination)
    return destination


def save_json_result(out_dir: Path, tool_name: str, data: dict[str, Any], problem: dict[str, Any]) -> Path:
    data['_problem'] = problem
    output_path = out_dir / f'tool_result_{tool_name}.json'
    output_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    return output_path


def maybe_save_result(out_dir: Path, tool_name: str, result: dict[str, Any], problem: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    if 'transcript' in result and isinstance(result['transcript'], str):
        paths.append(save_text_result(out_dir, tool_name, result['transcript']))
    elif 'clip_path' in result:
        source_path = Path(result['clip_path'])
        if not source_path.is_absolute():
            source_path = (repo_root / source_path).resolve()
        paths.append(save_audio_result(out_dir, tool_name, source_path, None))
    elif 'output_path' in result and isinstance(result['output_path'], str):
        source_path = Path(result['output_path'])
        if not source_path.is_absolute():
            source_path = (repo_root / source_path).resolve()
        if source_path.suffix.lower() == '.wav' and source_path.exists():
            paths.append(save_audio_result(out_dir, tool_name, source_path, None))
        else:
            paths.append(save_json_result(out_dir, tool_name, result, problem))
    elif 'separated_files' in result and isinstance(result['separated_files'], dict):
        for stem, path_str in result['separated_files'].items():
            source_path = Path(path_str)
            if not source_path.is_absolute():
                source_path = (repo_root / source_path).resolve()
            paths.append(save_audio_result(out_dir, tool_name, source_path, stem))
        paths.append(save_json_result(out_dir, tool_name, result, problem))
    else:
        paths.append(save_json_result(out_dir, tool_name, result, problem))
    return paths


def load_schedule(path: Path) -> list[Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def build_global_batches(schedule: list[Any], output_root: Path, restrict_tool: str | None = None) -> tuple[Dict[str, List[dict[str, Any]]], Dict[str, List[dict[str, Any]]], dict[str, dict[str, Any]]]:
    task_parameters: dict[str, list[dict[str, Any]]] = {}
    task_metadata: dict[str, list[dict[str, Any]]] = {}
    problems: dict[str, dict[str, Any]] = {}

    for item_index, entry in enumerate(schedule, start=1):
        if not isinstance(entry, list) or len(entry) != 2:
            print(f'Skipping malformed schedule entry at index {item_index}: {entry}')
            continue

        problem, tool_chain = entry
        problem_id = problem.get('id') or f'item_{item_index:03d}'
        audio_path = find_audio_file(problem)

        output_dir = output_root / 'tool_outputs' / problem_id
        output_dir.mkdir(parents=True, exist_ok=True)
        if audio_path.exists():
            shutil.copy(audio_path, output_dir / audio_path.name)
        else:
            print(f'  WARNING: audio file not found for {problem_id}: {audio_path}')

        problems[problem_id] = {
            'problem': problem,
            'tool_chain': tool_chain,
            'results': [None] * len(tool_chain),
            'output_dir': output_dir,
        }

        for step_index, step in enumerate(tool_chain, start=1):
            if not isinstance(step, dict):
                print(f'  Skipping malformed step for {problem_id}: {step}')
                problems[problem_id]['results'][step_index - 1] = {
                    'tool': None,
                    'error': 'Malformed step',
                    'raw_step': step,
                }
                continue

            tool_name = step.get('tool')
            if restrict_tool is not None and tool_name != restrict_tool:
                problems[problem_id]['results'][step_index - 1] = {
                    'tool': tool_name,
                    'parameters': step.get('parameters', {}),
                    'status': 'skipped',
                    'message': f'Skipped tool because restrict_tool={restrict_tool}',
                }
                continue

            if tool_name is None:
                problems[problem_id]['results'][step_index - 1] = {
                    'tool': None,
                    'parameters': step.get('parameters', {}),
                    'status': 'error',
                    'message': 'Missing tool name',
                }
                continue

            parameters = normalize_parameters(step.get('parameters', {}), audio_path)

            # Save per-problem input config before batch execution
            write_json(
                output_dir / f'tool_input_{step_index}.json',
                # output_dir / f'tool_inputs_{tool_name}_{params_suffix(parameters)}.json',
                {'tool': tool_name, 'parameters': parameters},
            )

            dispatch_parameters = dict(parameters)
            if tool_name in TOOLS_REQUIRING_OUTPUT_PATH:
                dispatch_parameters['output_path'] = str(output_dir / f'step{step_index}_{tool_name}.wav')

            task_parameters.setdefault(tool_name, []).append(dispatch_parameters)
            task_metadata.setdefault(tool_name, []).append({
                'problem_id': problem_id,
                'step_index': step_index,
                'tool_name': tool_name,
                'origin': {
                    'question': problem.get('question'),
                    'choice': problem.get('choice'),
                    'answer': problem.get('answer'),
                    'question_type': problem.get('question_type'),
                    '_json_path': problem.get('_json_path'),
                    'audio_path': str(audio_path),
                },
            })

    return task_parameters, task_metadata, problems


def execute_batches(task_parameters: dict[str, list[dict[str, Any]]], task_metadata: dict[str, list[dict[str, Any]]], problems: dict[str, dict[str, Any]], output_root: Path) -> None:
    for tool_name, parameter_list in task_parameters.items():
        python_executable = TOOL_ENV.get(tool_name, sys.executable)
        input_path = output_root / f'batch_{tool_name}_input.json'
        output_path = output_root / f'batch_{tool_name}_output.json'

        write_json(input_path, parameter_list)
        print(f'python env: {python_executable}')
        cmd = [
            python_executable,
            str(repo_root / 'tools' / 'tool_batch_execute.py'),
            '--tool-name', tool_name,
            '--input-file', str(input_path),
            '--output-file', str(output_path),
        ]

        print(f'Running batch for tool {tool_name} ({len(parameter_list)} calls) using {python_executable}')
        subprocess.run(cmd, check=True)

        batch_results = read_json(output_path)
        if not isinstance(batch_results, list):
            print(batch_results)
            raise ValueError(f'Unexpected batch result for {tool_name}: expected list, got {type(batch_results).__name__}')

        metadata_list = task_metadata[tool_name]
        if len(batch_results) != len(metadata_list):
            raise ValueError(
                f'Batch result count mismatch for {tool_name}: {len(batch_results)} results vs {len(metadata_list)} metadata entries'
            )

        for result_index, result in enumerate(batch_results):
            metadata = metadata_list[result_index]
            problem_id = metadata['problem_id']
            step_index = metadata['step_index']
            output_dir = problems[problem_id]['output_dir']

            # file_tag = f'{tool_name}_{params_suffix(parameter_list[result_index])}'
            file_tag = f'{step_index}' #{params_suffix(parameter_list[result_index])}'
            saved_paths = maybe_save_result(output_dir, file_tag, result, problems[problem_id]['problem'])
            recorded_parameters = {k: v for k, v in parameter_list[result_index].items() if k != 'output_path'}
            problems[problem_id]['results'][step_index - 1] = {
                'tool': tool_name,
                'parameters': recorded_parameters,
                'result': result,
                'saved_files': [str(path) for path in saved_paths],
                'origin': metadata['origin'],
            }


def run_tool_chain_accelerated(schedule_path: Path, output_root: Path, restrict_tool: str | None = None) -> None:
    schedule = load_schedule(schedule_path)
    output_root.mkdir(parents=True, exist_ok=True)

    task_parameters, task_metadata, problems = build_global_batches(schedule, output_root, restrict_tool=restrict_tool)
    execute_batches(task_parameters, task_metadata, problems, output_root)

    summary: list[dict[str, Any]] = []
    for problem_id, data in problems.items():
        summary.append({
            'id': problem_id,
            'question': data['problem'].get('question'),
            'tool_chain': [step.get('tool') for step in data['tool_chain'] if isinstance(step, dict)],
            'results': data['results'],
        })

    summary_path = output_root / 'apply_tool_summary.json'
    write_json(summary_path, summary)
    print(f'Done. Summary written to {summary_path}')


if __name__ == '__main__':
    args = ArgumentParser(description='Apply tool chains with batch acceleration')
    args.add_argument('--schedule_path', type=Path, default=repo_root / 'tool_schedules.json', help='Path to the tool schedule JSON file')
    args.add_argument('--output_root', type=Path, default=repo_root / 'apply_tool_results_accelerated', help='Directory to save the tool execution results')
    # args.add_argument('--restrict-tool', type=str, default=None, help='If set, only execute this specific tool and skip others')
    parsed_args = args.parse_args()
    run_tool_chain_accelerated(
        parsed_args.schedule_path,
        parsed_args.output_root,
        # restrict_tool=parsed_args.restrict_tool,
    )


