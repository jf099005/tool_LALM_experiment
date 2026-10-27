"""GRPO reward functions for training a tool-using LALM (ms-swift backbone).

The policy is prompted with a source/target audio pair plus a tool catalogue
(see `build_rl_dataset.py`, which converts the raw
`gen_1st_stage_data/build_dataset.py` samples into this single-turn shape)
and must emit a pure tool-call JSON trace of the form::

    {"tool_calls": [{"tool_name": "...", "parameters": {...}}, ...]}

that reproduces the target audio from the source. Ground truth for each
sample is the exact same structure (the `answer` field of the synthetic
dataset), passed to the reward functions as the `solution` column -- either
as a JSON string or an already-parsed dict/list, both are accepted.

Note this is a single-shot approximation: the model emits the whole chain at
once without ever seeing a real intermediate tool-execution result, unlike
the multi-turn dialogue `gen_1st_stage_data/build_dataset.py` renders for
SFT. Real multi-turn tool-use RL (rollout driving `interface/agent.py`'s
real tool executor turn by turn) is future work.

Rewards are exposed as `swift.rewards.orm.ORM` subclasses (matching the GRPO
plugin convention: `class MyReward(ORM): def __call__(self, completions,
**kwargs) -> List[float]`) so they can be combined with `--reward_funcs` /
`--reward_weights` like::

    swift rlhf --rlhf_type grpo \\
        --external_plugins tool_use_training/tool_use_RL/reward.py \\
        --reward_funcs tool_format tool_accuracy \\
        --reward_weights 0.2 0.8 \\
        ...

`ToolAudioClosenessReward` goes one step further than the symbolic rewards
above: it actually executes the model's predicted tool-call chain against the
real source audio (`interface.executor.run_tool_call`, the same executor
`testing_tool_use_benchmark/run_eval.py` drives) and rewards how much closer
the resulting audio gets to the real target versus doing nothing --
`compare_audio(chain_output, target).closeness_score -
compare_audio(source, target).closeness_score` (`audio_metrics.compare_audio`,
also shared with that benchmark). This is exactly the signal
`run_eval.py`'s `summary.mean_audio_{log_mel_cosine,mfcc_cosine,closeness}_diff`
report, so training against it directly targets those numbers rather than
only the symbolic chain match.

Run this file directly (`python reward.py`) for a self-contained smoke test
that doesn't require ms-swift to be importable.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from swift.rewards.orm import ORM
except ImportError:  # allow `python reward.py` / static analysis outside the ms-swift env

    class ORM:  # type: ignore[no-redef]

        def __init__(self, args: Optional[Any] = None, **kwargs):
            self.args = args

        def __call__(self, **kwargs) -> List[float]:
            raise NotImplementedError


# ---------------------------------------------------------------------------
# Real tool execution, for ToolAudioClosenessReward -- shares its executor
# (`interface.executor`) and audio-similarity metrics (`audio_metrics`, from
# `testing_tool_use_benchmark/`) with the evaluation-time benchmark so the
# reward and the eventual eval report the exact same numbers.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARK_DIR = _REPO_ROOT / 'testing_tool_use_benchmark'
for _path in (_REPO_ROOT, _BENCHMARK_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from interface.executor import ToolExecutionError, UnknownToolError, extract_audio_outputs, run_tool_call
    from audio_metrics import compare_audio

    _AUDIO_EXECUTION_AVAILABLE = True
except ImportError:  # tools/audio deps (librosa, soundfile, ...) not installed in this env
    _AUDIO_EXECUTION_AVAILABLE = False


# ---------------------------------------------------------------------------
# Parsing: pull a {"tool_calls": [...]} structure out of free-form model text.
# ---------------------------------------------------------------------------

_JSON_OBJECT_RE = re.compile(r'\{.*\}', re.DOTALL)
_CODE_FENCE_RE = re.compile(r'```(?:json)?\s*(.*?)\s*```', re.DOTALL)
_TIMESTAMP_RE = re.compile(r'^(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?$')


def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE_RE.search(text)
    return match.group(1) if match else text


def _largest_balanced_json_object(text: str) -> Optional[str]:
    """Return the first balanced-brace `{...}` substring, if any.

    Models often wrap the JSON answer in reasoning text; a naive greedy regex
    (`\\{.*\\}`) can swallow trailing prose past the real closing brace, so we
    scan for the first properly-balanced span instead.
    """
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_tool_call_payload(value: Union[str, Dict, List, None]) -> Optional[Dict[str, Any]]:
    """Best-effort parse of a completion or ground-truth value into a dict.

    Accepts an already-parsed dict/list (ground truth loaded from JSON
    upstream), a bare JSON string, or free-form text containing a JSON object
    (optionally inside a ``` fence). Returns None if nothing usable is found.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {'tool_calls': value}
    if not isinstance(value, str):
        return None

    text = _strip_code_fence(value.strip())

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        candidate = _largest_balanced_json_object(text)
        if candidate is None:
            return None
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None

    if isinstance(parsed, list):
        return {'tool_calls': parsed}
    if isinstance(parsed, dict):
        return parsed
    return None


def extract_tool_calls(value: Union[str, Dict, List, None]) -> Optional[List[Dict[str, Any]]]:
    """Extract a well-formed `tool_calls` list, or None if the payload is malformed."""
    payload = parse_tool_call_payload(value)
    if payload is None:
        return None
    calls = payload.get('tool_calls')
    if not isinstance(calls, list):
        return None
    cleaned = []
    for call in calls:
        if not isinstance(call, dict):
            return None
        name = call.get('tool_name')
        params = call.get('parameters', {})
        if not isinstance(name, str) or not isinstance(params, dict):
            return None
        cleaned.append({'tool_name': name, 'parameters': params})
    return cleaned


# ---------------------------------------------------------------------------
# Parameter-value comparison.
# ---------------------------------------------------------------------------

def _timestamp_to_seconds(text: str) -> Optional[float]:
    match = _TIMESTAMP_RE.match(text.strip())
    if not match:
        return None
    hours, minutes, secs, millis = match.groups()
    total = int(hours) * 3600 + int(minutes) * 60 + int(secs)
    if millis:
        total += int(millis.ljust(3, '0')) / 1000.0
    return float(total)


def values_close(
    pred: Any,
    gt: Any,
    rel_tol: float = 0.05,
    abs_tol: float = 0.02,
    timestamp_tol_seconds: float = 0.75,
) -> bool:
    """Loose equality between a predicted and ground-truth parameter value.

    Numeric values (including numeric strings) tolerate small deviations
    since the RL policy can only estimate continuous quantities (gain, rate,
    SNR, ...) by ear. Timestamp strings ("HH:MM:SS.mmm") are parsed to
    seconds and compared with a wider tolerance for the same reason. Other
    strings must match exactly (mode/algorithm/label choices are discrete).
    """
    if isinstance(pred, bool) or isinstance(gt, bool):
        return pred == gt

    if isinstance(pred, (int, float)) and isinstance(gt, (int, float)):
        return abs(pred - gt) <= max(abs_tol, rel_tol * abs(gt))

    if isinstance(pred, str) and isinstance(gt, str):
        pred_ts, gt_ts = _timestamp_to_seconds(pred), _timestamp_to_seconds(gt)
        if pred_ts is not None and gt_ts is not None:
            return abs(pred_ts - gt_ts) <= timestamp_tol_seconds

        try:
            return abs(float(pred) - float(gt)) <= max(abs_tol, rel_tol * abs(float(gt)))
        except ValueError:
            pass

        return pred.strip().lower() == gt.strip().lower()

    return pred == gt


# ---------------------------------------------------------------------------
# Scoring a predicted tool-call sequence against ground truth.
# ---------------------------------------------------------------------------

TOOL_NAME_WEIGHT = 0.3
PARAM_WEIGHT = 0.7


def _score_single_call(pred: Dict[str, Any], gt: Dict[str, Any]) -> float:
    if pred.get('tool_name') != gt.get('tool_name'):
        return 0.0

    gt_params = gt.get('parameters', {})
    pred_params = pred.get('parameters', {})
    num_params = max(len(gt_params), len(pred_params))
    if num_params == 0:
        param_score = 1.0
    else:
        matched = sum(
            1 for key, gt_val in gt_params.items()
            if key in pred_params and values_close(pred_params[key], gt_val)
        )
        param_score = matched / num_params

    return TOOL_NAME_WEIGHT + PARAM_WEIGHT * param_score


def score_tool_call_sequence(pred_calls: List[Dict[str, Any]], gt_calls: List[Dict[str, Any]]) -> float:
    """Position-wise score in [0, 1]; order matters since each tool's output
    feeds the next (a transposed or merely-correct-as-a-set chain is still
    a wrong answer for this task)."""
    max_len = max(len(pred_calls), len(gt_calls))
    if max_len == 0:
        return 1.0

    total = 0.0
    for i in range(max_len):
        if i >= len(pred_calls) or i >= len(gt_calls):
            continue  # missing/extra step scores 0
        total += _score_single_call(pred_calls[i], gt_calls[i])
    return total / max_len


def score_tool_name_f1(pred_calls: List[Dict[str, Any]], gt_calls: List[Dict[str, Any]]) -> float:
    """Order-agnostic bag-of-tool-names F1; a softer auxiliary/curriculum signal."""
    pred_names = [c.get('tool_name') for c in pred_calls]
    gt_names = [c.get('tool_name') for c in gt_calls]
    if not pred_names and not gt_names:
        return 1.0
    if not pred_names or not gt_names:
        return 0.0

    remaining = list(gt_names)
    matched = 0
    for name in pred_names:
        if name in remaining:
            remaining.remove(name)
            matched += 1

    precision = matched / len(pred_names)
    recall = matched / len(gt_names)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Real tool-chain execution + audio-closeness scoring.
# ---------------------------------------------------------------------------

def execute_tool_chain(pred_calls: List[Dict[str, Any]], source_audio: str, work_dir: Path) -> str:
    """Actually run a predicted tool-call chain against real audio, step by
    step, mirroring `testing_tool_use_benchmark/run_eval.py`'s `run_sample`
    loop -- except the whole chain is already generated (single-shot GRPO),
    so this simply replays it rather than driving a live multi-turn engine.

    `run_tool_call` always overwrites `parameters["audio_path"]` with the
    real current audio path, so the model's own placeholder value there
    (`<AUDIO_A>`, `<OUTPUT_OF_STEP_i>`, ...) never needs resolving here.
    Stops at the first tool that fails to execute (unknown tool, invalid
    parameters, or a runtime error) and returns whatever audio the chain
    produced up to that point -- a downstream tool fed the wrong audio is no
    more informative than not running it, so there is nothing to gain by
    forcing the chain onward.
    """
    current_audio = source_audio
    for step_idx, call in enumerate(pred_calls, start=1):
        tool_name = call.get('tool_name')
        parameters = call.get('parameters')
        if not isinstance(tool_name, str) or not isinstance(parameters, dict):
            break
        try:
            result = run_tool_call(tool_name, dict(parameters), current_audio, work_dir, step_idx)
        except (UnknownToolError, ToolExecutionError):
            break
        audio_outputs = extract_audio_outputs(result)
        if not audio_outputs:
            continue  # e.g. a mid-chain `asr` call produces no new audio
        current_audio = audio_outputs.get('target') or audio_outputs.get('') or next(iter(audio_outputs.values()))
    return current_audio


def score_audio_closeness_diff(pred_calls: List[Dict[str, Any]], source_audio: str, target_audio: str) -> float:
    """Execute `pred_calls` for real and return
    `closeness_score(chain_output, target) - closeness_score(source, target)`:
    how much closer to the ground-truth target the model's own predicted
    chain actually gets the audio, versus doing nothing. Positive means the
    chain helped; ~0 means it made no difference (including "failed on step
    one"); negative means it moved the audio away from the target.
    """
    with tempfile.TemporaryDirectory(prefix='grpo_tool_reward_') as tmp_dir:
        final_audio = execute_tool_chain(pred_calls, source_audio, Path(tmp_dir))
        final_metrics = compare_audio(final_audio, target_audio)
    baseline_metrics = compare_audio(source_audio, target_audio)
    return final_metrics['closeness_score'] - baseline_metrics['closeness_score']


# ---------------------------------------------------------------------------
# ORM (reward) classes.
# ---------------------------------------------------------------------------

def _solution_calls(solution: Any) -> Optional[List[Dict[str, Any]]]:
    """Ground truth is normally well-formed already, but route it through the
    same extractor as completions for robustness (e.g. plain JSON-string
    columns loaded straight from `tool_usage_qa.json`)."""
    return extract_tool_calls(solution)


class ToolCallFormatReward(ORM):
    """Gate reward: 1.0 if the completion parses into a well-formed
    `{"tool_calls": [{"tool_name": str, "parameters": dict}, ...]}` payload,
    0.0 otherwise. Use this to bootstrap valid-JSON output before the
    accuracy reward has any signal to work with.
    """

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        return [1.0 if extract_tool_calls(c) is not None else 0.0 for c in completions]


class ToolCallAccuracyReward(ORM):
    """Position-wise tool-name + parameter-value match against `solution`.

    Malformed completions (or malformed ground truth) score 0.0 rather than
    raising, so this can be combined with `ToolCallFormatReward` instead of
    depending on it for correctness.
    """

    def __call__(self, completions: List[str], solution: List[Any], **kwargs) -> List[float]:
        rewards = []
        for completion, sol in zip(completions, solution):
            pred_calls = extract_tool_calls(completion)
            gt_calls = _solution_calls(sol)
            if pred_calls is None or gt_calls is None:
                rewards.append(0.0)
                continue
            rewards.append(score_tool_call_sequence(pred_calls, gt_calls))
        return rewards


class ToolNameF1Reward(ORM):
    """Order-agnostic bag-of-tool-names F1 against `solution`."""

    def __call__(self, completions: List[str], solution: List[Any], **kwargs) -> List[float]:
        rewards = []
        for completion, sol in zip(completions, solution):
            pred_calls = extract_tool_calls(completion)
            gt_calls = _solution_calls(sol)
            if pred_calls is None or gt_calls is None:
                rewards.append(0.0)
                continue
            rewards.append(score_tool_name_f1(pred_calls, gt_calls))
        return rewards


class ToolAudioClosenessReward(ORM):
    """Executes the model's predicted tool-call chain against the real source
    audio and rewards how much closer (in log-mel/MFCC cosine similarity --
    `audio_metrics.compare_audio`'s `closeness_score`) the resulting audio
    gets to the real target than doing nothing does.

    Unlike `ToolCallAccuracyReward`/`ToolNameF1Reward` (which only compare the
    predicted call *structure* to one synthetic ground-truth chain), this
    scores the actual acoustic result -- rewarding any chain that reproduces
    the target, including equally-valid alternate chains the ground truth
    doesn't happen to contain -- and is exactly the metric
    `testing_tool_use_benchmark/run_eval.py` reports as
    `mean_audio_{log_mel_cosine,mfcc_cosine,closeness}_diff`.

    Requires the `audios` dataset column (`[source_audio_path,
    target_audio_path]`, as produced by `build_rl_dataset.py`'s
    `to_swift_rl_sample`) and a real tool-executable environment (librosa,
    soundfile, ...); scores 0.0 per-sample when either is unavailable or a
    sample's own audio fails to load/process, rather than raising, so it
    degrades gracefully when combined with the symbolic rewards above.
    """

    def __call__(self, completions: List[str], audios: Optional[List[List[str]]] = None, **kwargs) -> List[float]:
        if not _AUDIO_EXECUTION_AVAILABLE or audios is None:
            return [0.0] * len(completions)
        rewards = []
        for completion, audio_pair in zip(completions, audios):
            pred_calls = extract_tool_calls(completion)
            if pred_calls is None or not audio_pair or len(audio_pair) < 2:
                rewards.append(0.0)
                continue
            source_audio, target_audio = audio_pair[0], audio_pair[1]
            try:
                rewards.append(score_audio_closeness_diff(pred_calls, source_audio, target_audio))
            except Exception:  # noqa: BLE001 - a bad/missing audio file must not kill the whole batch
                rewards.append(0.0)
        return rewards


class ToolUseReward(ORM):
    """Composite single-function reward: format-gated blend of sequence
    accuracy, name-F1, and (when `audios` is available) real audio-closeness
    improvement. Convenient when the training script only supports a single
    `--reward_funcs` entry; prefer combining the individual rewards above
    with `--reward_weights` when the trainer allows multiple.
    """

    def __init__(
        self,
        args: Optional[Any] = None,
        sequence_weight: float = 0.5,
        name_f1_weight: float = 0.2,
        audio_closeness_weight: float = 0.3,
        **kwargs,
    ):
        super().__init__(args, **kwargs)
        self.sequence_weight = sequence_weight
        self.name_f1_weight = name_f1_weight
        self.audio_closeness_weight = audio_closeness_weight
        self._audio_closeness_reward = ToolAudioClosenessReward(args)

    def __call__(
        self,
        completions: List[str],
        solution: List[Any],
        audios: Optional[List[List[str]]] = None,
        **kwargs,
    ) -> List[float]:
        rewards = []
        for completion, sol in zip(completions, solution):
            pred_calls = extract_tool_calls(completion)
            gt_calls = _solution_calls(sol)
            if pred_calls is None or gt_calls is None:
                rewards.append(0.0)
                continue
            sequence_score = score_tool_call_sequence(pred_calls, gt_calls)
            name_f1_score = score_tool_name_f1(pred_calls, gt_calls)
            rewards.append(self.sequence_weight * sequence_score + self.name_f1_weight * name_f1_score)

        if self.audio_closeness_weight and audios is not None:
            audio_rewards = self._audio_closeness_reward(completions=completions, audios=audios, **kwargs)
            rewards = [r + self.audio_closeness_weight * a for r, a in zip(rewards, audio_rewards)]
        return rewards


orms: Dict[str, type] = {
    'tool_format': ToolCallFormatReward,
    'tool_accuracy': ToolCallAccuracyReward,
    'tool_name_f1': ToolNameF1Reward,
    'tool_audio_closeness': ToolAudioClosenessReward,
    'tool_use': ToolUseReward,
}

try:
    # `swift.rewards.orm.orms` is the registry the GRPO trainer actually looks
    # up `--reward_funcs` names in (`if reward_func in orms: ...`). Loading
    # this file via `--external_plugins` just `importlib.import_module`s it,
    # so defining a same-named local `orms` dict here would only shadow the
    # framework's copy -- it must be mutated in place to actually register.
    from swift.rewards import orms as _swift_orms

    _swift_orms.update(orms)
except ImportError:
    pass


if __name__ == '__main__':
    gt = json.dumps({
        'tool_calls': [
            {'tool_name': 'denoise', 'parameters': {'audio_path': '<AUDIO_A>', 'algorithm': 'wiener'}},
            {
                'tool_name': 'pitch_shift',
                'parameters': {'audio_path': '<OUTPUT_OF_STEP_1>', 'n_steps': 2},
            },
        ]
    })

    completions = [
        gt,  # exact match
        '```json\n' + json.dumps({
            'tool_calls': [
                {'tool_name': 'denoise', 'parameters': {'audio_path': '<AUDIO_A>', 'algorithm': 'wiener'}},
                {'tool_name': 'pitch_shift', 'parameters': {'audio_path': '<OUTPUT_OF_STEP_1>', 'n_steps': 3}},
            ]
        }) + '\n```',  # right tools, one param off
        json.dumps({'tool_calls': [{'tool_name': 'denoise', 'parameters': {'audio_path': '<AUDIO_A>'}}]}),  # missing step
        'I think the answer is not valid json at all',  # malformed
    ]
    solution = [gt] * len(completions)

    for name, cls in orms.items():
        reward_fn = cls()
        print(name, reward_fn(completions=completions, solution=solution))

    # ToolAudioClosenessReward needs real, executable audio -- demonstrate it
    # separately against a synthesized source/target pair rather than the
    # placeholder paths above.
    if _AUDIO_EXECUTION_AVAILABLE:
        import wave as _wave

        import numpy as _np

        from tools.pitch_time import PitchShiftTool

        def _write_sine_wav(path: Path, freq: float, seconds: float = 1.0, sr: int = 16000) -> None:
            t = _np.linspace(0, seconds, int(sr * seconds), endpoint=False)
            samples = (0.5 * _np.sin(2 * _np.pi * freq * t) * _np.iinfo(_np.int16).max).astype(_np.int16)
            with _wave.open(str(path), 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(sr)
                f.writeframes(samples.tobytes())

        with tempfile.TemporaryDirectory(prefix='reward_audio_smoke_') as demo_dir:
            audio_a = Path(demo_dir) / 'a.wav'
            audio_b = Path(demo_dir) / 'b.wav'
            _write_sine_wav(audio_a, freq=220.0)
            # Ground truth: B is A shifted up 4 semitones.
            PitchShiftTool.execute({'audio_path': str(audio_a), 'n_steps': 4}, str(audio_b))

            audio_completions = [
                json.dumps({'tool_calls': [{'tool_name': 'pitch_shift', 'parameters': {'audio_path': '<AUDIO_A>', 'n_steps': 4}}]}),  # correct
                json.dumps({'tool_calls': [{'tool_name': 'pitch_shift', 'parameters': {'audio_path': '<AUDIO_A>', 'n_steps': -4}}]}),  # wrong direction
                json.dumps({'tool_calls': []}),  # does nothing
            ]
            audios = [[str(audio_a), str(audio_b)]] * len(audio_completions)
            reward_fn = ToolAudioClosenessReward()
            print('tool_audio_closeness (real execution)', reward_fn(completions=audio_completions, audios=audios))
    else:
        print('tool_audio_closeness (real execution): skipped, audio tooling not importable in this env')
