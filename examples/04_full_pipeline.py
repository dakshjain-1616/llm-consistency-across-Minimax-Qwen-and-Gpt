"""
04_full_pipeline.py — End-to-end mock pipeline.

Demonstrates the complete workflow:
  1. Generate mock responses for 3 models × 50 prompts × 10 runs
  2. Compute Levenshtein + semantic variance
  3. Compute per-model statistics with 95% confidence intervals
  4. Compute cross-model divergence
  5. Export variance.json, results.csv, and experiment_meta.json
  6. Generate an interactive HTML report

No API key required — all responses are synthetic.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import random
import tempfile
from pathlib import Path

from consistency_heatmap_ import (
    SEED_PROMPTS,
    MODELS,
    compute_variance,
    compute_semantic_variance_data,
    cross_model_divergence,
    compute_stats,
    verdict,
    export_variance_csv,
    save_experiment_meta,
    generate_html_report,
)

# ---------------------------------------------------------------------------
# Step 1 — Generate mock responses
# ---------------------------------------------------------------------------
NUM_RUNS    = int(os.getenv("NUM_RUNS", "10"))
NUM_PROMPTS = int(os.getenv("NUM_PROMPTS", "50"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

_TEMPLATES = [
    "The answer is {noun}. This is a well-known fact in {domain}.",
    "{noun} refers to {domain} in scientific literature.",
    "In essence, {noun} is defined as {domain}.",
    "Simply put, {noun} describes {domain}.",
    "{noun}: {domain}.",
]
_NOUNS   = ["entropy", "recursion", "gravity", "DNA", "an algorithm", "momentum"]
_DOMAINS = ["physics", "biology", "computer science", "chemistry", "mathematics"]


def _mock_response(seed: int) -> str:
    rng = random.Random(seed)
    return rng.choice(_TEMPLATES).format(
        noun=rng.choice(_NOUNS),
        domain=rng.choice(_DOMAINS),
    )


rng = random.Random(RANDOM_SEED)
raw_responses: dict = {}
variance_data: dict = {}

print(f"Generating mock data: {len(MODELS)} models × {NUM_PROMPTS} prompts × {NUM_RUNS} runs")
for model_name in MODELS:
    raw_responses[model_name] = {}
    variance_data[model_name] = {}
    bias = rng.uniform(0.05, 0.35)
    for p_idx in range(NUM_PROMPTS):
        responses = []
        for run in range(NUM_RUNS):
            stability = rng.uniform(0.0, 1.0)
            seed = rng.randint(0, 9999) if stability < bias else p_idx * 100 + int(stability * 10)
            responses.append(_mock_response(seed))
        raw_responses[model_name][str(p_idx)] = responses
        variance_data[model_name][str(p_idx)] = compute_variance(responses)

print("  Done.")

# ---------------------------------------------------------------------------
# Step 2 — Compute semantic variance + cross-model divergence
# ---------------------------------------------------------------------------
semantic_data = compute_semantic_variance_data(raw_responses)
divergence    = cross_model_divergence(raw_responses)

# ---------------------------------------------------------------------------
# Step 3 — Per-model statistics
# ---------------------------------------------------------------------------
model_stats = {
    name: compute_stats(list(variance_data[name].values()))
    for name in MODELS
}

print("\nPer-model summary:")
for name, st in model_stats.items():
    print(f"  {name:<10}  mean={st['mean']:.4f}  std={st['std']:.4f}  "
          f"CI=[{st['ci_95_low']:.3f}, {st['ci_95_high']:.3f}]  "
          f"[{verdict(st['mean'])}]")

# ---------------------------------------------------------------------------
# Step 4 — Export outputs
# ---------------------------------------------------------------------------
out_dir = Path(os.getenv("OUTPUTS_DIR", "outputs")) / "pipeline_example"
out_dir.mkdir(parents=True, exist_ok=True)

variance_path = out_dir / "variance.json"
variance_path.write_text(json.dumps(variance_data, indent=2))

csv_path = export_variance_csv(
    variance_data, SEED_PROMPTS[:NUM_PROMPTS],
    out_dir / "results.csv",
    semantic_data=semantic_data,
)

meta_path = save_experiment_meta(
    out_dir / "experiment_meta.json",
    mode="mock",
    models=MODELS,
    num_runs=NUM_RUNS,
    num_prompts=NUM_PROMPTS,
    variance_data=variance_data,
    semantic_data=semantic_data,
    cross_model_divergence=divergence,
    model_stats=model_stats,
)

html_path = generate_html_report(
    variance_data,
    SEED_PROMPTS[:NUM_PROMPTS],
    out_dir / "report.html",
    mode="mock",
    num_runs=NUM_RUNS,
    models=MODELS,
    semantic_data=semantic_data,
    model_stats=model_stats,
    cross_model_divergence=divergence,
)

# ---------------------------------------------------------------------------
# Step 5 — Summary
# ---------------------------------------------------------------------------
print(f"\nOutputs written to {out_dir}/")
for p in [variance_path, csv_path, meta_path, html_path]:
    print(f"  {p.name:<30} ({p.stat().st_size / 1024:.1f} KB)")

print("\nTop 3 most volatile prompts (cross-model average):")
import itertools
prompt_scores: dict = {}
for m, scores in variance_data.items():
    for p_str, v in scores.items():
        prompt_scores.setdefault(int(p_str), []).append(v)
top3 = sorted(
    {idx: sum(vs)/len(vs) for idx, vs in prompt_scores.items()}.items(),
    key=lambda x: -x[1]
)[:3]
for rank, (p_idx, avg_v) in enumerate(top3, 1):
    print(f"  {rank}. [{avg_v:.4f}] {SEED_PROMPTS[p_idx]}")
