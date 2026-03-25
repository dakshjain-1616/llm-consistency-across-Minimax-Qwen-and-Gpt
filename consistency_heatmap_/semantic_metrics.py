"""
semantic_metrics.py — Semantic similarity metrics (complement to Levenshtein distance).

Uses Jaccard token-overlap to measure vocabulary divergence between responses.
This catches cases where very different words convey similar meaning, which
character-level Levenshtein misses.
"""

import re
import itertools
from typing import List, Dict


def tokenize(text: str) -> set:
    """Extract lowercase alphanumeric tokens from text."""
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def jaccard_similarity(s1: str, s2: str) -> float:
    """Jaccard token-overlap similarity in [0, 1]. 1 = identical vocabulary."""
    t1, t2 = tokenize(s1), tokenize(s2)
    if not t1 and not t2:
        return 1.0
    union = len(t1 | t2)
    return len(t1 & t2) / union if union > 0 else 1.0


def semantic_distance(s1: str, s2: str) -> float:
    """Semantic distance = 1 - Jaccard similarity. 0 = same vocab, 1 = no overlap."""
    return 1.0 - jaccard_similarity(s1, s2)


def semantic_variance(responses: List[str]) -> float:
    """
    Mean pairwise semantic distance across all response pairs.

    0 = all responses share identical vocabulary,
    1 = completely disjoint vocabularies across runs.
    """
    if len(responses) < 2:
        return 0.0
    pairs = list(itertools.combinations(responses, 2))
    distances = [semantic_distance(a, b) for a, b in pairs]
    return sum(distances) / len(distances)


def compute_semantic_variance_data(raw_responses: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, float]]:
    """
    Compute semantic variance for every (model, prompt) pair.

    Args:
        raw_responses: {model_name: {prompt_idx_str: [responses]}}

    Returns:
        {model_name: {prompt_idx_str: semantic_variance_float}}
    """
    result: Dict[str, Dict[str, float]] = {}
    for model_name, prompt_map in raw_responses.items():
        result[model_name] = {}
        for p_idx_str, responses in prompt_map.items():
            result[model_name][p_idx_str] = semantic_variance(responses)
    return result


def cross_model_divergence(raw_responses: Dict[str, Dict[str, List[str]]]) -> Dict[str, float]:
    """
    For each prompt, compute mean pairwise semantic distance between models'
    representative responses (first run per model).  High divergence → models
    disagree on vocabulary; low divergence → models agree.

    Args:
        raw_responses: {model_name: {prompt_idx_str: [responses]}}

    Returns:
        {prompt_idx_str: float} — cross-model semantic divergence per prompt
    """
    if not raw_responses:
        return {}

    # Collect all prompt indices
    prompt_indices: set = set()
    for model_data in raw_responses.values():
        prompt_indices.update(model_data.keys())

    divergence: Dict[str, float] = {}
    for p_idx in sorted(prompt_indices, key=lambda x: int(x)):
        # Use first response from each model as representative
        reps = [
            model_data[p_idx][0]
            for model_data in raw_responses.values()
            if p_idx in model_data and model_data[p_idx]
        ]
        if len(reps) < 2:
            divergence[p_idx] = 0.0
        else:
            pairs = list(itertools.combinations(reps, 2))
            distances = [semantic_distance(a, b) for a, b in pairs]
            divergence[p_idx] = sum(distances) / len(distances)

    return divergence
