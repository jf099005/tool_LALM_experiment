"""
End-to-end uncertainty quantification (UQ) for a single audio-QA item,
to be plugged into qwen25_with_tool_chain_evaluation.py.

Implements the protocol from "Walking Through Uncertainty: An Empirical
Study of Uncertainty Estimation for Audio-Aware Large Language Models"
(Kuan, Huang & Lee, arXiv:2604.25591, 2026), which found that for
closed-form (MCQ) audio benchmarks (MMAU, MMAR, MMSU, SAKURA),
semantic/discrete-semantic entropy and P(True) self-verification
consistently outperform raw predictive entropy and length-normalized
entropy.

Per item this module:
  1. Draws K stochastic samples (temperature ~1.0) of the *same*
     prompt/audio via vLLM's `SamplingParams(n=K, ...)`.
  2. Computes predictive entropy and length-normalized entropy from
     the cumulative log-probabilities of those samples.
  3. Computes discrete semantic entropy by mapping each sample onto
     one of the MCQ `choices`.
  4. Optionally computes "true" semantic entropy via NLI-based
     bidirectional-entailment clustering of the raw sample texts.
  5. Computes P(True): a single self-verification generation that asks
     the model (conditioned on the audio) whether the greedy answer is
     correct, scored via the True/False token log-probabilities.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .entropy_methods import (
    discrete_semantic_entropy,
    length_normalized_entropy,
    p_true_from_first_token_logprobs,
    predictive_entropy,
)
from .nli import NLIEntailmentModel, semantic_entropy

AUDIO_TOKEN = "<|audio_bos|><|AUDIO|><|audio_eos|>"          # Qwen2-Audio / Qwen2.5-Omni
QWEN3_OMNI_AUDIO_TOKEN = "<|audio_start|><|audio_pad|><|audio_end|>"


def count_audio_blocks(prompt: str, audio_token: str = AUDIO_TOKEN) -> int:
    """Number of audio placeholders in a prompt, for the given model's audio token."""
    if audio_token == AUDIO_TOKEN:
        return prompt.count("<|AUDIO|>")  # unchanged behaviour for the Qwen2.5 prompts
    return prompt.count(audio_token)


def _sequence_logprob(completion) -> Optional[float]:
    """Summed log-prob of a sampled completion. Uses vLLM's
    `cumulative_logprob` when present, else sums the per-token logprobs."""
    if completion.cumulative_logprob is not None:
        return completion.cumulative_logprob
    if not completion.logprobs:
        return None
    total = 0.0
    for token_id, step in zip(completion.token_ids, completion.logprobs):
        lp = step.get(token_id) if step else None
        if lp is None:
            return None
        total += lp.logprob
    return total


def build_self_verification_prompt(
    system_prompt: str,
    question: str,
    choices: Sequence[str],
    candidate_answer: str,
    num_audio_tokens: int = 1,
    audio_token: str = AUDIO_TOKEN,
) -> str:
    """Build a self-verification ("P(True)") prompt that asks the model,
    conditioned on the audio, whether `candidate_answer` is correct.
    The model is constrained to answer with a single True/False token.
    """
    choice_text = "\n".join(choices) if choices else ""
    audio_block = audio_token * max(num_audio_tokens, 1)
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n"
        f"{audio_block}\nQuestion: {question}\nChoices:\n{choice_text}"
        f"\nProposed answer: {candidate_answer}\nConsidering the audio, is the "
        f"proposed answer correct? Respond with exactly one word: True or False.\n"
        f"<|im_end|>\n<|im_start|>assistant\n"
    )


def compute_uncertainty(
    llm,
    prompt: str,
    multi_modal_data: dict,
    choices: Sequence[str],
    greedy_prediction: str,
    system_prompt: str,
    question: str,
    num_samples: int = 10,
    sample_temperature: float = 1.0,
    max_tokens: int = 256,
    compute_semantic: bool = False,
    compute_p_true: bool = True,
    nli_model: Optional[NLIEntailmentModel] = None,
) -> Dict:
    """Run the full UQ protocol for one item and return a JSON-serializable dict.

    `multi_modal_data` is the same dict passed to `llm.generate` for the
    greedy prediction (e.g. `{"audio": [(array, sr), ...]}`).
    """
    item = {
        "prompt": prompt,
        "multi_modal_data": multi_modal_data,
        "choices": choices,
        "greedy_prediction": greedy_prediction,
        "question": question,
    }
    return compute_uncertainty_batch(
        llm, [item], system_prompt=system_prompt, num_samples=num_samples,
        sample_temperature=sample_temperature, max_tokens=max_tokens,
        compute_semantic=compute_semantic, compute_p_true=compute_p_true, nli_model=nli_model,
    )[0]


def compute_uncertainty_batch(
    llm,
    items: Sequence[Dict],
    system_prompt: str,
    num_samples: int = 10,
    sample_temperature: float = 1.0,
    max_tokens: int = 256,
    compute_semantic: bool = False,
    compute_p_true: bool = True,
    nli_model: Optional[NLIEntailmentModel] = None,
    audio_token: str = AUDIO_TOKEN,
) -> List[Dict]:
    """Same protocol as `compute_uncertainty`, for several items at once: one
    `llm.generate` call for all items' K samples and one for all P(True)
    checks, so vLLM can batch across items. Each item needs `prompt`,
    `multi_modal_data`, `choices`, `greedy_prediction`, `question`.
    Results are identical in shape (and, for greedy P(True), in value) to
    calling `compute_uncertainty` item by item.

    `audio_token` is the model's audio placeholder, used to build the P(True) prompt
    (default: Qwen2.5-Omni's; pass QWEN3_OMNI_AUDIO_TOKEN for Qwen3-Omni).
    """
    from vllm import SamplingParams

    results = [{"num_samples": num_samples} for _ in items]

    if num_samples > 0:
        # logprobs=0: the vLLM V1 engine only fills `cumulative_logprob` when
        # logprobs are requested; 0 means "just the sampled token's logprob".
        sampling_params = SamplingParams(n=num_samples, temperature=sample_temperature, max_tokens=max_tokens, logprobs=0)
        outputs = llm.generate(
            [{"prompt": it["prompt"], "multi_modal_data": it["multi_modal_data"]} for it in items],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        for it, result, output in zip(items, results, outputs):
            completions = output.outputs
            texts = [c.text.strip() for c in completions]
            logprob_sums = [_sequence_logprob(c) for c in completions]
            lengths = [len(c.token_ids) for c in completions]

            result["sampled_predictions"] = texts
            result["predictive_entropy"] = predictive_entropy(logprob_sums)
            result["length_normalized_entropy"] = length_normalized_entropy(logprob_sums, lengths)

            disc_entropy, distribution = discrete_semantic_entropy(texts, it["choices"])
            result["discrete_semantic_entropy"] = disc_entropy
            result["answer_distribution"] = distribution

            if compute_semantic:
                sem_entropy, clusters = semantic_entropy(texts, question=it["question"], nli_model=nli_model)
                result["semantic_entropy"] = sem_entropy
                result["semantic_clusters"] = clusters

    if compute_p_true:
        requests = []
        for it in items:
            verification_prompt = build_self_verification_prompt(
                system_prompt=system_prompt,
                question=it["question"],
                choices=it["choices"],
                candidate_answer=it["greedy_prediction"],
                num_audio_tokens=count_audio_blocks(it["prompt"], audio_token),
                audio_token=audio_token,
            )
            requests.append({"prompt": verification_prompt, "multi_modal_data": it["multi_modal_data"]})
        verification_params = SamplingParams(temperature=0.0, max_tokens=5, logprobs=20)
        verification_outputs = llm.generate(requests, sampling_params=verification_params, use_tqdm=False)
        for result, output in zip(results, verification_outputs):
            verification_output = output.outputs[0]
            first_token_logprobs = verification_output.logprobs[0] if verification_output.logprobs else None
            result["p_true"] = p_true_from_first_token_logprobs(first_token_logprobs)
            result["p_true_raw_response"] = verification_output.text.strip()

    return results
