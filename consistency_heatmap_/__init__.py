"""
consistency_heatmap_ — LLM consistency measurement toolkit.

50 prompts × 10 runs × 3 models, variance visualized in 30s.
Built autonomously by NEO · your autonomous AI Agent.
"""

# Prompts
from .prompts import SEED_PROMPTS

# Variance / Levenshtein metrics
from .variance import (
    compute_variance,
    normalized_levenshtein,
    _levenshtein_distance,
)

# Semantic (Jaccard) metrics
from .semantic_metrics import (
    tokenize,
    jaccard_similarity,
    semantic_distance,
    semantic_variance,
    compute_semantic_variance_data,
    cross_model_divergence,
)

# Descriptive statistics
from .stats import compute_stats, verdict

# OpenRouter API client
from .api_client import MODELS, query_model

# Export utilities
from .export_utils import export_variance_csv, save_experiment_meta

# HTML report generator
from .html_report import generate_html_report

__all__ = [
    # prompts
    "SEED_PROMPTS",
    # variance
    "compute_variance",
    "normalized_levenshtein",
    "_levenshtein_distance",
    # semantic metrics
    "tokenize",
    "jaccard_similarity",
    "semantic_distance",
    "semantic_variance",
    "compute_semantic_variance_data",
    "cross_model_divergence",
    # stats
    "compute_stats",
    "verdict",
    # api client
    "MODELS",
    "query_model",
    # export
    "export_variance_csv",
    "save_experiment_meta",
    # html
    "generate_html_report",
]
