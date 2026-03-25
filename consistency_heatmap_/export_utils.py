"""
export_utils.py — CSV and enriched JSON export utilities.

Provides functions to persist experiment results in formats suitable for
downstream analysis (spreadsheets, Pandas, BI tools).
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def export_variance_csv(
    variance_data: Dict[str, Dict[str, float]],
    prompts: List[str],
    output_path: Path,
    semantic_data: Optional[Dict[str, Dict[str, float]]] = None,
) -> Path:
    """
    Export per-(model, prompt) variance scores to a CSV file.

    Columns: model, prompt_idx, prompt_text, levenshtein_variance[, semantic_variance]

    Returns the path to the written file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["model", "prompt_idx", "prompt_text", "levenshtein_variance"]
    if semantic_data is not None:
        fieldnames.append("semantic_variance")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model_name in sorted(variance_data):
            scores = variance_data[model_name]
            for p_idx_str in sorted(scores, key=lambda x: int(x)):
                p_idx = int(p_idx_str)
                row: dict = {
                    "model": model_name,
                    "prompt_idx": p_idx + 1,  # 1-based for humans
                    "prompt_text": prompts[p_idx] if p_idx < len(prompts) else "",
                    "levenshtein_variance": round(scores[p_idx_str], 6),
                }
                if semantic_data is not None:
                    sem = semantic_data.get(model_name, {}).get(p_idx_str, 0.0)
                    row["semantic_variance"] = round(sem, 6)
                writer.writerow(row)

    return output_path


def save_experiment_meta(
    output_path: Path,
    *,
    mode: str,
    models: Dict[str, str],
    num_runs: int,
    num_prompts: int,
    variance_data: Dict[str, Dict[str, float]],
    semantic_data: Optional[Dict[str, Dict[str, float]]] = None,
    cross_model_divergence: Optional[Dict[str, float]] = None,
    model_stats: Optional[Dict[str, dict]] = None,
    elapsed_seconds: Optional[float] = None,
) -> Path:
    """
    Save an enriched JSON metadata file alongside the standard variance.json.

    This file captures everything that variance.json doesn't: timestamps,
    per-model statistics, semantic variance, and cross-model divergence.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "num_runs": num_runs,
        "num_prompts": num_prompts,
        "models": models,
        "model_stats": model_stats or {},
        "semantic_variance": semantic_data or {},
        "cross_model_divergence": cross_model_divergence or {},
    }
    if elapsed_seconds is not None:
        meta["elapsed_seconds"] = round(elapsed_seconds, 2)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return output_path
