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

AUDIO_TOKEN = "<|audio_bos|><|AUDIO|><|audio_eos|>"


def build_self_verification_prompt(
    system_prompt: str,
    question: str,
    choices: Sequence[str],
    candidate_answer: str,
    num_audio_tokens: int = 1,
) -> str:
    """Build a self-verification ("P(True)") prompt that asks the model,
    conditioned on the audio, whether `candidate_answer` is correct.
    The model is constrained to answer with a single True/False token.
    """
    choice_text = "\n".join(choices) if choices else ""
    audio_block = AUDIO_TOKEN * max(num_audio_tokens, 1)
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
    from vllm import SamplingParams

    result = {"num_samples": num_samples}

    if num_samples > 0:
        sampling_params = SamplingParams(n=num_samples, temperature=sample_temperature, max_tokens=max_tokens)
        outputs = llm.generate(
            [{"prompt": prompt, "multi_modal_data": multi_modal_data}],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        completions = outputs[0].outputs

        texts = [c.text.strip() for c in completions]
        logprob_sums = [c.cumulative_logprob for c in completions]
        lengths = [len(c.token_ids) for c in completions]

        result["sampled_predictions"] = texts
        result["predictive_entropy"] = predictive_entropy(logprob_sums)
        result["length_normalized_entropy"] = length_normalized_entropy(logprob_sums, lengths)

        disc_entropy, distribution = discrete_semantic_entropy(texts, choices)
        result["discrete_semantic_entropy"] = disc_entropy
        result["answer_distribution"] = distribution

        if compute_semantic:
            sem_entropy, clusters = semantic_entropy(texts, question=question, nli_model=nli_model)
            result["semantic_entropy"] = sem_entropy
            result["semantic_clusters"] = clusters

    if compute_p_true:
        num_audio_tokens = prompt.count("<|AUDIO|>")
        verification_prompt = build_self_verification_prompt(
            system_prompt=system_prompt,
            question=question,
            choices=choices,
            candidate_answer=greedy_prediction,
            num_audio_tokens=num_audio_tokens,
        )
        verification_params = SamplingParams(temperature=0.0, max_tokens=5, logprobs=20)
        verification_outputs = llm.generate(
            [{"prompt": verification_prompt, "multi_modal_data": multi_modal_data}],
            sampling_params=verification_params,
            use_tqdm=False,
        )
        verification_output = verification_outputs[0].outputs[0]
        first_token_logprobs = verification_output.logprobs[0] if verification_output.logprobs else None
        result["p_true"] = p_true_from_first_token_logprobs(first_token_logprobs)
        result["p_true_raw_response"] = verification_output.text.strip()

    return result
