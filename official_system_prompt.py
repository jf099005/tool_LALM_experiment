"""Compose the system prompt used by stage-1 SFT / stage-2 GRPO data generation
(and, for consistency, `testing_tool_use_benchmark/run_eval.py`).

The system message is: <the target model's own official default system text>
+ <the tool catalogue>. The model's own default is read straight out of its
`chat_template.json` / `tokenizer_config.json` (the same text its chat
template injects when a conversation has no system turn, e.g. Qwen's
"You are a helpful assistant.") rather than hard-coded, so pointing this at a
different model's directory picks up *that* model's own convention -- the
whole point being that swapping the base model for training later doesn't
require guessing at (or copy-pasting) a system prompt by hand.

This is deliberately separate from `interface.protocol.build_system_prompt`:
that one is a hand-written protocol explanation for the *interactive* agent
(arbitrary instruction, real tool set, spells out the JSON turn format in
English). The prompt here is for models that *learn* the JSON format from
SFT examples, so it carries no protocol explanation -- just the model's own
persona/greeting plus the tool catalogue.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

# Fallback per known model_type, used only when a model directory isn't given
# (or its chat_template can't be parsed) -- e.g. `--model` is a bare HF hub id
# with no local snapshot to inspect yet. Extend this as new model families are
# trained; it's a fallback, not the source of truth (the model's own config
# is), so it's fine for this to lag behind.
KNOWN_DEFAULT_SYSTEM_PROMPTS = {
    "qwen2_5_omni": "You are a helpful assistant.",
    "qwen2_omni": "You are a helpful assistant.",
    "qwen2_5_vl": "You are a helpful assistant.",
    "qwen2_audio": "You are a helpful assistant.",
}
GENERIC_FALLBACK_SYSTEM_PROMPT = "You are a helpful assistant."

# Matches the common HF convention (Qwen and several other chat-template
# families) of a literal `<|im_start|>system\n...<|im_end|>` block the
# template injects by default when no system message is supplied.
_IM_SYSTEM_RE = re.compile(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", re.DOTALL)


def _extract_default_from_template(template: str) -> Optional[str]:
    match = _IM_SYSTEM_RE.search(template)
    if not match:
        return None
    text = match.group(1).strip()
    return text or None


def load_official_system_prompt(
    model_dir: Optional[str | Path],
    model_type: Optional[str] = None,
) -> str:
    """Best-effort read of the target model's own default system message.

    Tries, in order: `<model_dir>/chat_template.json`, the `chat_template`
    field embedded in `<model_dir>/tokenizer_config.json`, a known-model
    fallback keyed by `model_type`, then a generic default. `model_dir` not
    existing locally (e.g. a bare hub id that hasn't been downloaded) is not
    an error -- it just skips straight to the fallbacks.
    """
    if model_dir is not None:
        model_dir = Path(model_dir)
        for filename in ("chat_template.json", "tokenizer_config.json"):
            path = model_dir / filename
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            template = data.get("chat_template") if isinstance(data, dict) else None
            if isinstance(template, list):
                # Some tokenizer_config.json store a list of {"name": ..., "template": ...}.
                template = next(
                    (t.get("template") for t in template if isinstance(t, dict) and t.get("template")),
                    None,
                )
            if isinstance(template, str):
                found = _extract_default_from_template(template)
                if found:
                    return found

    if model_type and model_type in KNOWN_DEFAULT_SYSTEM_PROMPTS:
        return KNOWN_DEFAULT_SYSTEM_PROMPTS[model_type]

    return GENERIC_FALLBACK_SYSTEM_PROMPT


def compose_system_prompt(
    tools_block: str,
    base_system_prompt: Optional[str] = None,
    model_dir: Optional[str | Path] = None,
    model_type: Optional[str] = None,
) -> str:
    """Compose the full system message: base greeting + tool catalogue.

    `base_system_prompt` (an explicit override) wins if given; otherwise the
    base is auto-detected from `model_dir`/`model_type` via
    `load_official_system_prompt`.

    `tools_block` is expected to already be a complete, self-labeled
    preamble section -- see `tools.tools_registry.describe_available_tools`,
    which renders it via a `tool_call_formats.ToolCallFormat` (Qwen's own
    "# Tools" heading by default). This function just concatenates; it
    doesn't add its own label on top.
    """
    if base_system_prompt is None:
        base_system_prompt = load_official_system_prompt(model_dir, model_type)
    return f"{base_system_prompt}\n\n{tools_block}"
