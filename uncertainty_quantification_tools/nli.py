"""
Optional NLI-based bidirectional-entailment clustering for "true"
semantic entropy (H_sem), as described in arXiv:2604.25591.

Two sampled answers s, s' are placed in the same semantic cluster iff
an NLI model judges (question + s) entails (question + s') AND
(question + s') entails (question + s).

This is heavier than `discrete_semantic_entropy` (it loads a small
cross-encoder NLI model) and is only used when `--semantic_entropy`
is passed to the evaluation script. The model is loaded lazily and on
failure (e.g. missing `sentence-transformers`) clustering falls back
to None so callers can skip semantic entropy gracefully.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple


class NLIEntailmentModel:
    """Thin wrapper around a sentence-transformers CrossEncoder NLI model."""

    def __init__(self, model_name: str = 'cross-encoder/nli-deberta-v3-small', device: Optional[str] = None):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=device)
        self.label2idx = {'contradiction': 0, 'entailment': 1, 'neutral': 2}

    def entails(self, premise: str, hypothesis: str) -> bool:
        scores = self.model.predict([(premise, hypothesis)])
        label_idx = int(scores[0].argmax())
        return label_idx == self.label2idx['entailment']

    def bidirectional_entailment(self, a: str, b: str) -> bool:
        return self.entails(a, b) and self.entails(b, a)


_DEFAULT_MODEL: Optional[object] = None


def get_default_nli_model() -> Optional[NLIEntailmentModel]:
    """Lazily instantiate (and cache) a default NLI model.

    Returns None if the model cannot be loaded (e.g. missing
    dependency), in which case semantic entropy should be skipped.
    """
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        try:
            _DEFAULT_MODEL = NLIEntailmentModel()
        except Exception as exc:
            print(f"[uncertainty_quantification_tools] Could not load NLI model for semantic entropy ({exc}). Semantic entropy will be skipped.")
            _DEFAULT_MODEL = False
    return _DEFAULT_MODEL or None


def cluster_by_entailment(
    texts: Sequence[str],
    question: Optional[str] = None,
    nli_model: Optional[NLIEntailmentModel] = None,
) -> Optional[List[List[int]]]:
    """Cluster `texts` (indices into the input sequence) by bidirectional
    entailment. Returns a list of clusters, each a list of indices into
    `texts`, or None if no NLI model is available.
    """
    if nli_model is None:
        nli_model = get_default_nli_model()
    if nli_model is None:
        return None

    def wrap(t: str) -> str:
        return f"{t} {question}" if question else t

    clusters = []
    for i, text in enumerate(texts):
        placed = False
        for cluster in clusters:
            rep = texts[cluster[0]]
            if nli_model.bidirectional_entailment(wrap(rep), wrap(text)):
                cluster.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    return clusters


def semantic_entropy(
    texts: Sequence[str],
    question: Optional[str] = None,
    nli_model: Optional[NLIEntailmentModel] = None,
) -> Tuple[Optional[float], Optional[List[List[str]]]]:
    """H_sem(x) = -sum_k p(c_k|x) log p(c_k|x), p(c_k|x) ~= |c_k| / K

    Returns (entropy, clusters_as_text) or (None, None) if no NLI
    model is available.
    """
    if not texts:
        return None, None
    clusters = cluster_by_entailment(texts, question=question, nli_model=nli_model)
    if clusters is None:
        return None, None
    n = len(texts)
    entropy = -sum((len(c) / n) * math.log(len(c) / n) for c in clusters)
    cluster_texts = [[texts[i] for i in c] for c in clusters]
    return entropy, cluster_texts
