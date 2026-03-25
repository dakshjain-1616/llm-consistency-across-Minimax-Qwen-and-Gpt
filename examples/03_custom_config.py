"""
03_custom_config.py — Customising behaviour via environment variables.

Demonstrates how every tunable parameter can be overridden through env vars,
and shows how to read the effective configuration before running.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Override settings before importing the package so the module-level
# constants pick up the custom values.
os.environ.setdefault("NUM_RUNS", "5")
os.environ.setdefault("NUM_PROMPTS", "10")
os.environ.setdefault("TEMPERATURE", "0.5")
os.environ.setdefault("MAX_TOKENS", "128")
os.environ.setdefault("OUTPUTS_DIR", "outputs/custom_run")

from consistency_heatmap_ import MODELS, SEED_PROMPTS, compute_variance, compute_stats, verdict

num_runs    = int(os.getenv("NUM_RUNS", "10"))
num_prompts = int(os.getenv("NUM_PROMPTS", "50"))
temperature = float(os.getenv("TEMPERATURE", "0.7"))
max_tokens  = int(os.getenv("MAX_TOKENS", "256"))
outputs_dir = os.getenv("OUTPUTS_DIR", "outputs")

print("Effective configuration")
print("=" * 40)
print(f"  Models      : {', '.join(MODELS.keys())}")
print(f"  Prompts     : {num_prompts} (of {len(SEED_PROMPTS)} available)")
print(f"  Runs        : {num_runs} per prompt per model")
print(f"  Temperature : {temperature}")
print(f"  Max tokens  : {max_tokens}")
print(f"  Output dir  : {outputs_dir}")
print("=" * 40)

# Demo: compute variance on the configured number of prompts with fake data
import random
rng = random.Random(42)

print("\nMock variance (first model, first 5 prompts):")
for i in range(min(5, num_prompts)):
    fake_responses = [
        f"Answer {j} to: {SEED_PROMPTS[i][:30]}" for j in range(num_runs)
    ]
    v = compute_variance(fake_responses)
    st = compute_stats([v])
    print(f"  Prompt {i+1:02d}: variance={v:.4f}  [{verdict(v)}]")

print("\nTo run a full live experiment:")
print("  export OPENROUTER_API_KEY=your_key")
print("  python run_experiment.py --prompts 10 --runs 5")
