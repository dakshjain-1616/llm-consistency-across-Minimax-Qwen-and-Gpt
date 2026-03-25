"""
stats.py — Descriptive statistics with 95% confidence intervals.

Used to enrich experiment reports with more than just mean/min/max.
"""

import math
from typing import List, Dict


def compute_stats(values: List[float]) -> Dict[str, float]:
    """
    Compute descriptive statistics for a list of variance values.

    Returns a dict with: n, mean, std, min, max, median, ci_95_low, ci_95_high
    """
    if not values:
        return {
            "n": 0, "mean": 0.0, "std": 0.0,
            "min": 0.0, "max": 0.0, "median": 0.0,
            "ci_95_low": 0.0, "ci_95_high": 0.0,
        }

    n = len(values)
    mean = sum(values) / n

    if n > 1:
        std = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
    else:
        std = 0.0

    sorted_v = sorted(values)
    mid = n // 2
    median = sorted_v[mid] if n % 2 == 1 else (sorted_v[mid - 1] + sorted_v[mid]) / 2.0

    # 95% CI via normal approximation (z = 1.96)
    se = std / math.sqrt(n) if n > 1 else 0.0
    ci_low = max(0.0, mean - 1.96 * se)
    ci_high = min(1.0, mean + 1.96 * se)

    return {
        "n": n,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "median": round(median, 6),
        "ci_95_low": round(ci_low, 6),
        "ci_95_high": round(ci_high, 6),
    }


def verdict(mean_variance: float) -> str:
    """Human-readable consistency verdict from a mean variance score."""
    if mean_variance < 0.1:
        return "Very consistent"
    if mean_variance < 0.2:
        return "Consistent"
    if mean_variance < 0.35:
        return "Moderate variance"
    return "High variance"
