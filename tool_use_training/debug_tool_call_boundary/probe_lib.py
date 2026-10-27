"""Shared plumbing for probing why the first token of an assistant turn
(should be `<tool_call>`, id 151657) sometimes comes out as a lone
byte-fallback token (observed: id 124, `'À'`) that can't decode to valid
UTF-8 on its own.

Both `probe_first_token.py` (CLI, used by `run_all.sh`) and
`probe_first_token.ipynb` (interactive) import this module rather than
duplicating the load/patch/analyze logic, so the two stay consistent.

Findings so far (see chat history that produced this directory):
- The raw training data and ms-swift's actual training-time `template.encode()`
  tokenize `<tool_call>` correctly (id 151657, no corruption) -- ruled out.
- Live reproduction confirms the bug deterministically (temperature=0): the
  first generated token is id 124, matching `decode_warnings` from the full
  benchmark run.
- At that position, the top-~15 candidate logits (including id 151657 itself)
  are all within bf16's rounding resolution of each other (~0.125 near
  logit 17) -- a suspiciously flat/degenerate distribution for what should be
  a confidently memorized structural token. This module's three probes
  (no-audio, no-adapter, fp32) exist to figure out which factor drives that
  flatness.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOOL_CALL_TOKEN = "<tool_call>"
TOOL_CALL_TOKEN_ID = 151657

DEFAULT_MODEL = "/work/u1501463/model_qwen_25"
DEFAULT_ADAPTER = str(
    REPO_ROOT / "tool_use_training" / "stage1_training" / "v8-20260805-005451" / "checkpoint-400"
)
DEFAULT_BENCHMARK_JSON = REPO_ROOT / "testing_tool_use_benchmark" / "benchmark.json"
DEFAULT_RESULTS_JSON = REPO_ROOT / "testing_tool_use_benchmark" / "results_v5_checkpoint2000.2.json"
DEFAULT_SAMPLE_ID = "bench_a06eb33e5157"

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_sample(
    sample_id: str = DEFAULT_SAMPLE_ID,
    benchmark_json: Path = DEFAULT_BENCHMARK_JSON,
    results_json: Path = DEFAULT_RESULTS_JSON,
) -> Dict[str, Any]:
    """Reconstruct the exact (entry, system_prompt) pair a prior benchmark run
    used for `sample_id` -- the system prompt is pulled from the saved results
    file rather than rebuilt, so it's byte-identical across every probe
    variant (only audio/adapter/dtype vary between them)."""
    bench = json.loads(Path(benchmark_json).read_text(encoding="utf-8"))
    entries = bench if isinstance(bench, list) else bench.get("entries") or bench.get("samples")
    entry = next(e for e in entries if e["id"] == sample_id)

    results = json.loads(Path(results_json).read_text(encoding="utf-8"))
    r = next(x for x in results["results"] if x["id"] == sample_id)

    return {"entry": entry, "system_prompt": r["prompt"]}


def build_messages(entry: Dict[str, Any], system_prompt: str, use_audio: bool):
    """`use_audio=False` strips the `<audio_N><audio>` tags from the question
    text and returns no audio paths -- isolates whether audio-embedding
    injection is what destabilizes the first generated token, while keeping
    the system prompt (tool catalogue, instructions) byte-identical."""
    from interface.protocol import AUDIO_TOKEN

    question = entry["question"]
    if not use_audio:
        import re

        question = re.sub(r"<audio_\d+>" + re.escape(AUDIO_TOKEN), "(audio omitted for text-only probe)", question)
        audios: List[str] = []
    else:
        audios = [entry["source_audio"], entry["target_audio"]]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    return messages, audios


def load_engine(model: str, adapter_dir: Optional[str], dtype: torch.dtype, max_new_tokens: int = 64):
    from interface.engine import SwiftEngine

    return SwiftEngine(model, adapter_dir=adapter_dir, max_new_tokens=max_new_tokens, temperature=0.0, torch_dtype=dtype)


def patch_generate_to_capture_scores(model) -> Dict[str, Any]:
    """Monkeypatches `model.generate` so the next call also returns per-step
    logits, without passing `output_scores`/`return_dict_in_generate` as
    separate kwargs alongside an existing `generation_config` object (that
    combination triggers a `transformers` deprecation path and may not
    reflect the true unmodified logits) -- instead mutates the
    `generation_config` object ms-swift already threads through in place.
    Returns the dict `captured` gets written into (`captured['out']` after
    the call)."""
    captured: Dict[str, Any] = {}
    orig_generate = model.generate

    def patched(*args, **kwargs):
        gen_cfg = kwargs.get("generation_config")
        if gen_cfg is not None:
            gen_cfg.output_scores = True
            gen_cfg.return_dict_in_generate = True
        else:
            kwargs["output_scores"] = True
            kwargs["return_dict_in_generate"] = True
        out = orig_generate(*args, **kwargs)
        captured["out"] = out
        return out

    model.generate = patched
    return captured


def run_probe(
    model: str = DEFAULT_MODEL,
    adapter_dir: Optional[str] = DEFAULT_ADAPTER,
    dtype: str = "bf16",
    use_audio: bool = True,
    sample_id: str = DEFAULT_SAMPLE_ID,
    topk: int = 20,
) -> Dict[str, Any]:
    """Runs one (adapter x dtype x audio) variant and returns a JSON-able
    dict: the generated text, raw token ids, and a top-k breakdown of the
    first generated position's logits (including where `<tool_call>` ranks)."""
    sample = load_sample(sample_id)
    messages, audios = build_messages(sample["entry"], sample["system_prompt"], use_audio)

    engine = load_engine(model, adapter_dir, DTYPE_MAP[dtype])
    captured = patch_generate_to_capture_scores(engine.engine.model)

    request = engine._InferRequest(messages=messages, audios=audios)
    response = engine.engine.infer([request], engine.request_config)[0]
    content = response.choices[0].message.content
    token_ids = list(response.choices[0].token_ids or [])

    result: Dict[str, Any] = {
        "variant": {"adapter": bool(adapter_dir), "dtype": dtype, "use_audio": use_audio},
        "content": content,
        "token_ids_head": token_ids[:10],
        "has_replacement_char": "�" in content,
    }

    out = captured.get("out")
    if out is not None and getattr(out, "scores", None):
        first_logits = out.scores[0][0].float()  # step 0, batch 0; upcast for a stable comparison across dtypes
        tokenizer = engine.engine.tokenizer
        values, indices = torch.topk(first_logits, topk)

        top_list = []
        for val, idx in zip(values.tolist(), indices.tolist()):
            top_list.append({"token_id": idx, "piece": tokenizer.convert_ids_to_tokens(idx), "logit": val})

        tool_call_logit = first_logits[TOOL_CALL_TOKEN_ID].item()
        tool_call_rank = int((first_logits > tool_call_logit).sum().item())

        result["first_token_logits"] = {
            "top_k": top_list,
            "spread_top_k": values[0].item() - values[-1].item(),
            "tool_call_token": {
                "token_id": TOOL_CALL_TOKEN_ID,
                "logit": tool_call_logit,
                "rank": tool_call_rank,
            },
        }

    del engine
    torch.cuda.empty_cache()
    return result


def print_probe_result(result: Dict[str, Any]) -> None:
    v = result["variant"]
    print(f"--- variant: adapter={v['adapter']} dtype={v['dtype']} use_audio={v['use_audio']} ---")
    print("content:", repr(result["content"])[:160])
    print("first 10 token ids:", result["token_ids_head"])
    print("contains U+FFFD:", result["has_replacement_char"])
    fl = result.get("first_token_logits")
    if fl:
        tc = fl["tool_call_token"]
        print(f"<tool_call> (id={tc['token_id']}) logit={tc['logit']:.4f} rank={tc['rank']}")
        print(f"top-{len(fl['top_k'])} logit spread: {fl['spread_top_k']:.4f}")
        for entry in fl["top_k"][:10]:
            print(f"  id={entry['token_id']:<7} logit={entry['logit']:.4f}  {entry['piece']!r}")
    print()
