"""
Pure-python building blocks for uncertainty quantification (UQ) of
audio-aware LLM (ALLM) generations.

Implements the four "free-form sampling" metrics from:
    "Walking Through Uncertainty: An Empirical Study of Uncertainty
    Estimation for Audio-Aware Large Language Models"
    (Kuan, Huang & Lee, arXiv:2604.25591, 2026)

  - predictive_entropy            (H_pred)
  - length_normalized_entropy     (H_norm_tok, ratio-of-expectations)
  - discrete_semantic_entropy     (H_disc, for closed-form MCQ answers)
  - semantic_entropy clustering   (bidirectional-entailment clustering,
                                   implemented in nli.py)

None of the functions here require torch/vllm so they can be unit
tested in isolation.
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple, Union


def predictive_entropy(logprob_sums: Sequence[Optional[float]]) -> Optional[float]:
    """H_pred(x) ~= (1/K) * sum_i [ -log p(y_i | x) ]

    `logprob_sums` are the cumulative (summed) log-probabilities of K
    sampled sequences. Higher values indicate the model assigns lower
    probability to its own samples on average, i.e. higher uncertainty.
    """
    nlls = [-lp for lp in logprob_sums if lp is not None]
    if not nlls:
        return None
    return sum(nlls) / len(nlls)


def length_normalized_entropy(logprob_sums: Sequence[Optional[float]], lengths: Sequence[int]) -> Optional[float]:
    """Token-level normalized entropy using a ratio-of-expectations:

        H_norm_tok(x) = sum_i [ -log p(y_i|x) ] / sum_i |y_i|

    This is more stable than averaging per-sequence ratios because it
    is not dominated by very short sequences.
    """
    pairs = [(-lp, max(int(length), 1)) for lp, length in zip(logprob_sums, lengths) if lp is not None]
    if not pairs:
        return None
    total_nll = sum(p[0] for p in pairs)
    total_len = sum(p[1] for p in pairs)
    if total_len == 0:
        return None
    return total_nll / total_len


_LEADING_LETTER_RE = re.compile(r'^\s*\(?([A-Za-z])\)?\s*[\.\):,-]')


def match_choice(text: str, choices: Sequence[str]) -> Optional[int]:
    """Map a free-form model response onto the index of one of `choices`.

    Resolution order:
      1. A leading "A." / "(B)" / "C:" style letter, mapped positionally
         onto `choices` (the model is told the options in this order).
      2. Exact (case-insensitive) substring match of a choice's text.
         If several choices match, the one occurring earliest in the
         text wins.

    Returns None if no choice can be confidently identified.
    """
    if not choices:
        return None
    stripped = text.strip()
    m = _LEADING_LETTER_RE.match(stripped)
    if m:
        idx = ord(m.group(1).upper()) - ord('A')
        if 0 <= idx < len(choices):
            return idx
    lower_text = stripped.lower()
    found = []
    for i, choice in enumerate(choices):
        choice_lower = str(choice).strip().lower()
        if not choice_lower:
            continue
        pos = lower_text.find(choice_lower)
        if pos != -1:
            found.append((pos, i))
    if not found:
        return None
    found.sort()
    return found[0][1]


def discrete_semantic_entropy(texts: Sequence[str], choices: Sequence[str]) -> Tuple[Optional[float], Dict[str, int]]:
    """H_disc(x) = -sum_a p(a|x) log p(a|x), p(a|x) ~= count(a) / K

    `a` ranges over the closed answer set `choices`, plus a fallback
    "unresolved" bucket for samples that could not be mapped onto any
    choice. Returns (entropy, distribution) where `distribution` maps
    choice text (or "unresolved") -> raw count.
    """
    if not texts:
        return None, {}
    counts = Counter()
    for t in texts:
        idx = match_choice(t, choices)
        key = choices[idx] if idx is not None else 'unresolved'
        counts[key] += 1
    n = len(texts)
    entropy = -sum((c / n) * math.log(c / n) for c in counts.values() if c > 0)
    return entropy, dict(counts)


_TRUE_TOKENS = {'yes', 'true', 'correct'}
_FALSE_TOKENS = {'no', 'false', 'incorrect'}


def p_true_from_first_token_logprobs(first_token_logprobs: Optional[Dict[Union[str, int], object]]) -> Optional[float]:
    """Given vLLM's per-token logprob dict for the *first* generated
    token of a self-verification response, return the normalized
    probability mass assigned to "true"-like tokens vs "false"-like
    tokens:

        P(true | x, y_hat) = mass(true) / (mass(true) + mass(false))
    """
    if not first_token_logprobs:
        return None
    true_mass = 0.0
    false_mass = 0.0
    for token, lp in first_token_logprobs.items():
        logprob_value = getattr(lp, 'logprob', lp)
        try:
            prob = math.exp(float(logprob_value))
        except (TypeError, ValueError):
            continue
        decoded = getattr(lp, 'decoded_token', None)
        token_str = decoded if decoded is not None else token
        normalized = str(token_str).strip().lower().strip(string.punctuation)
        if normalized in _TRUE_TOKENS:
            true_mass += prob
            continue
        if normalized in _FALSE_TOKENS:
            false_mass += prob
    total = true_mass + false_mass
    if total == 0:
        return None
    return true_mass / total
