"""
01_quick_start.py — Minimal working example.

Computes Levenshtein variance for a small set of mock responses
and prints the result. No API key or external dependencies required.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from consistency_heatmap_ import compute_variance, SEED_PROMPTS

# Simulate 5 slightly different responses to the same prompt
responses = [
    "Paris is the capital of France.",
    "The capital of France is Paris.",
    "France's capital city is Paris.",
    "Paris, the capital of France.",
    "Paris is France's capital.",
]

variance = compute_variance(responses)
print(f"Prompt : {SEED_PROMPTS[0]!r}")
print(f"Runs   : {len(responses)}")
print(f"Variance (0=identical, 1=maximally different): {variance:.4f}")

if variance < 0.1:
    print("Verdict: Very consistent")
elif variance < 0.2:
    print("Verdict: Consistent")
elif variance < 0.35:
    print("Verdict: Moderate variance")
else:
    print("Verdict: High variance")
