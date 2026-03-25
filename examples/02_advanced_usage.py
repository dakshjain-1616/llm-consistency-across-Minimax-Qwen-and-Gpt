"""
02_advanced_usage.py — Levenshtein + semantic metrics side-by-side.

Shows how to combine character-level Levenshtein variance with
vocabulary-level Jaccard (semantic) variance for richer analysis.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from consistency_heatmap_ import (
    compute_variance,
    semantic_variance,
    compute_stats,
    verdict,
    SEED_PROMPTS,
)

# Two contrasting response sets for the same prompt
consistent_responses = [
    "The Pythagorean theorem states that a² + b² = c².",
    "The Pythagorean theorem: a² + b² = c².",
    "Pythagorean theorem: a squared plus b squared equals c squared.",
    "According to the Pythagorean theorem, a² + b² = c².",
    "The Pythagorean theorem holds that a² + b² = c².",
]

inconsistent_responses = [
    "Right triangles obey a² + b² = c² where c is the hypotenuse.",
    "In geometry, Pythagoras showed that side lengths of triangles relate.",
    "Triangles have angles summing to 180 degrees.",
    "The theorem describes perpendicular sides and their relationship.",
    "Euclid formalized many geometric relationships including this one.",
]

for label, responses in [("Consistent set", consistent_responses),
                          ("Inconsistent set", inconsistent_responses)]:
    lev_v = compute_variance(responses)
    sem_v = semantic_variance(responses)
    stats = compute_stats([lev_v] * len(responses))  # illustrative
    print(f"\n{label}")
    print(f"  Levenshtein variance : {lev_v:.4f}  ({verdict(lev_v)})")
    print(f"  Semantic variance    : {sem_v:.4f}")
    print(f"  Prompt               : {SEED_PROMPTS[8]!r}")
