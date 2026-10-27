"""Stage-3 reward: MCQ correctness of the final turn (verl `custom_reward_function`).

The answer is read from the *last assistant turn only* (`extra_info["final_text"]`, set by
`audio_tool_agent_loop.py`), and only if the episode ended because the model stopped calling
tools (`stop_reason == "answered"`). Reading it from `solution_str` instead would also scan
tool turns and earlier assistant turns.

    score = ACC_W * correct
          + FORMAT_W * well_formed_final_answer
          + TOOL_BONUS * (correct and >=1 successful tool call)
          - TOOL_ERROR_PENALTY * min(num_tool_errors, 3)

Defaults (override via env vars STAGE3_ACC_W, ...): 1.0 / 0.1 / 0.0 / 0.05. TOOL_BONUS is off
by default on purpose: accuracy alone rewards a tool call exactly when it helps. Raise it if
the policy collapses to never calling tools before it has learned when they help.

Every returned key is numeric and present for every sample so verl can average it into the
logged metrics (val: per data_source, e.g. val-aux/mmau_test_mini/acc/mean@1).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcq_protocol import extract_answer_letter  # noqa: E402

ACC_W = float(os.environ.get("STAGE3_ACC_W", "1.0"))
FORMAT_W = float(os.environ.get("STAGE3_FORMAT_W", "0.1"))
TOOL_BONUS = float(os.environ.get("STAGE3_TOOL_BONUS", "0.0"))
TOOL_ERROR_PENALTY = float(os.environ.get("STAGE3_TOOL_ERROR_PENALTY", "0.05"))


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, float]:
    extra_info = extra_info or {}
    if "stop_reason" in extra_info:
        answered = extra_info["stop_reason"] == "answered"
        final_text = extra_info.get("final_text", "") if answered else ""
    else:  # single-turn fallback (e.g. a no-tool baseline run with the default agent loop)
        answered, final_text = True, solution_str

    pred = extract_answer_letter(final_text)
    correct = float(pred is not None and pred == str(ground_truth).strip().upper())
    well_formed = float(answered and pred is not None)
    n_success = int(extra_info.get("num_tool_success", 0) or 0)
    n_errors = int(extra_info.get("num_tool_errors", 0) or 0)

    score = (
        ACC_W * correct
        + FORMAT_W * well_formed
        + TOOL_BONUS * float(correct and n_success > 0)
        - TOOL_ERROR_PENALTY * min(n_errors, 3)
    )
    return {
        "score": score,
        "acc": correct,
        "format_ok": well_formed,
        "num_tool_calls": float(extra_info.get("num_tool_calls", 0) or 0),
        "num_tool_success": float(n_success),
        "num_tool_errors": float(n_errors),
        "used_tool": float(n_success > 0),
        # acc among tool-using episodes = mean(acc_and_used_tool) / mean(used_tool)
        "acc_and_used_tool": correct * float(n_success > 0),
    }
