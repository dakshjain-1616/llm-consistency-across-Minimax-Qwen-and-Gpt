"""Variance calculation utilities using normalized Levenshtein distance."""

import itertools
from typing import List

try:
    import Levenshtein as lev

    def _levenshtein_distance(s1: str, s2: str) -> int:
        return lev.distance(s1, s2)

except ImportError:
    # Pure-Python fallback
    def _levenshtein_distance(s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if s1[i - 1] == s2[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return dp[n]


def normalized_levenshtein(s1: str, s2: str) -> float:
    """Return normalized Levenshtein distance in [0, 1]."""
    if not s1 and not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    return _levenshtein_distance(s1, s2) / max_len


def compute_variance(responses: List[str]) -> float:
    """
    Compute mean pairwise normalized Levenshtein distance across all runs.
    This is our variance metric: 0 = perfectly consistent, 1 = maximally different.
    """
    if len(responses) < 2:
        return 0.0
    pairs = list(itertools.combinations(responses, 2))
    distances = [normalized_levenshtein(a, b) for a, b in pairs]
    return sum(distances) / len(distances)
