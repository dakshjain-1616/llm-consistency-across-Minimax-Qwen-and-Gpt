# Examples

Runnable example scripts for the Consistency Heatmap project.
Each script adds `sys.path.insert` so it works from any directory.

| Script | What it demonstrates |
|--------|----------------------|
| [01_quick_start.py](01_quick_start.py) | Minimal example — compute Levenshtein variance for a handful of responses and print a verdict. ~15 lines, no API key needed. |
| [02_advanced_usage.py](02_advanced_usage.py) | Levenshtein **and** semantic (Jaccard) variance side-by-side; contrasts a consistent vs. inconsistent response set to show how both metrics behave. |
| [03_custom_config.py](03_custom_config.py) | Shows how to customise every tunable parameter (`NUM_RUNS`, `TEMPERATURE`, `MAX_TOKENS`, `OUTPUTS_DIR`, …) via environment variables; prints the effective configuration before running. |
| [04_full_pipeline.py](04_full_pipeline.py) | End-to-end mock workflow: generates synthetic responses for 3 models × 50 prompts × 10 runs, computes all metrics, exports `variance.json`, `results.csv`, `experiment_meta.json`, and an interactive HTML report. |

## Running an example

```bash
# from the project root
python examples/01_quick_start.py

# or from anywhere
cd /tmp
python /path/to/project/examples/04_full_pipeline.py
```

All examples run without an API key (mock data only).
To run a live experiment against real models see `run_experiment.py`.
