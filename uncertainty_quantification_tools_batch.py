"""Batched P(True) self-verification, for use when no per-item stochastic
sampling is needed (i.e. `--uq_num_samples 0` and only `--uq_p_true` is
requested).

This is kept as a standalone module rather than a change inside the
`uncertainty_quantification_tools` package: it only imports from that
package's public API (`build_self_verification_prompt`,
`p_true_from_first_token_logprobs`) and does not modify it. When any
sampling-based metric is also requested, callers should fall back to
`uncertainty_quantification_tools.compute_uncertainty`, which already
batches the K stochastic samples of a single item via
`SamplingParams(n=K, ...)` but issues one `llm.generate()` call per item.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from uncertainty_quantification_tools import (
    build_self_verification_prompt,
    p_true_from_first_token_logprobs,
)


def compute_p_true_batch(
    llm,
    items: Sequence[Dict],
    system_prompt: str,
    max_tokens: int = 5,
) -> List[Dict]:
    """Compute P(True) for a whole batch of items in a single `llm.generate()` call.

    Each entry of `items` must have: `prompt`, `multi_modal_data`,
    `choices`, `question`, `greedy_prediction`. Returns a list of result
    dicts, in the same order as `items` and with the same shape
    `compute_uncertainty()` produces for its p_true fields
    (`num_samples`, `p_true`, `p_true_raw_response`).
    """
    from vllm import SamplingParams

    verification_requests = []
    for item in items:
        num_audio_tokens = item["prompt"].count("<|AUDIO|>")
        verification_prompt = build_self_verification_prompt(
            system_prompt=system_prompt,
            question=item["question"],
            choices=item["choices"],
            candidate_answer=item["greedy_prediction"],
            num_audio_tokens=num_audio_tokens,
        )
        verification_requests.append({
            "prompt": verification_prompt,
            "multi_modal_data": item["multi_modal_data"],
        })

    verification_params = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=20)
    outputs = llm.generate(verification_requests, sampling_params=verification_params, use_tqdm=False)

    results = []
    for output in outputs:
        verification_output = output.outputs[0]
        first_token_logprobs = verification_output.logprobs[0] if verification_output.logprobs else None
        results.append({
            "num_samples": 0,
            "p_true": p_true_from_first_token_logprobs(first_token_logprobs),
            "p_true_raw_response": verification_output.text.strip(),
        })
    return results
