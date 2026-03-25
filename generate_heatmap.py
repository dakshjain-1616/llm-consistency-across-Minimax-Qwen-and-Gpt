"""
generate_heatmap.py — Reads variance.json and produces a PNG heatmap.

The heatmap shows normalized Levenshtein variance for each (model, prompt) pair:
  - X axis: prompt index (1–50)
  - Y axis: model name
  - Color:  variance score (0 = perfectly consistent, 1 = maximally different)

Output: results/heatmap.png  (or RESULTS_DIR/heatmap.png)
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works on headless servers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results"))
HEATMAP_DPI = int(os.getenv("HEATMAP_DPI", "150"))
HEATMAP_COLORMAP = os.getenv("HEATMAP_COLORMAP", "YlOrRd")
FIGURE_WIDTH = float(os.getenv("FIGURE_WIDTH", "20"))
FIGURE_HEIGHT = float(os.getenv("FIGURE_HEIGHT", "5"))


def load_variance(variance_path: Path) -> dict:
    with open(variance_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_matrix(variance_data: dict, num_prompts: int = 50) -> tuple:
    """
    Convert variance dict to a 2-D numpy array.

    Returns (matrix, model_names) where matrix shape is (n_models, num_prompts).
    """
    model_names = sorted(variance_data.keys())
    matrix = np.zeros((len(model_names), num_prompts), dtype=float)
    for row, model in enumerate(model_names):
        scores = variance_data[model]
        for col in range(num_prompts):
            matrix[row, col] = scores.get(str(col), 0.0)
    return matrix, model_names


def generate_heatmap(
    variance_path: Path = None,
    output_path: Path = None,
    title: str = "LLM Output Consistency Heatmap — Normalized Levenshtein Variance",
) -> Path:
    """
    Generate and save the heatmap PNG.

    Returns the path to the saved file.
    """
    if variance_path is None:
        variance_path = RESULTS_DIR / "variance.json"
    if output_path is None:
        output_path = RESULTS_DIR / "heatmap.png"

    variance_data = load_variance(variance_path)
    num_prompts = int(os.getenv("NUM_PROMPTS", "50"))
    matrix, model_names = build_matrix(variance_data, num_prompts=num_prompts)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=HEATMAP_COLORMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    # Axes labels
    ax.set_xticks(range(num_prompts))
    ax.set_xticklabels([str(i + 1) for i in range(num_prompts)], fontsize=7, rotation=90)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=11, fontweight="bold")

    ax.set_xlabel("Prompt Index (1 – 50)", fontsize=12, labelpad=8)
    ax.set_ylabel("Model", fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Variance (0 = consistent, 1 = inconsistent)", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # Annotate each cell with the numeric value
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            val = matrix[row, col]
            text_color = "white" if val > 0.55 else "black"
            ax.text(
                col, row, f"{val:.2f}",
                ha="center", va="center",
                fontsize=5.5,
                color=text_color,
            )

    # Add per-model mean annotation on the right margin
    for row, model in enumerate(model_names):
        mean_val = matrix[row].mean()
        ax.text(
            num_prompts + 0.5, row,
            f"μ={mean_val:.3f}",
            ha="left", va="center",
            fontsize=9, color="#333333",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=HEATMAP_DPI, bbox_inches="tight")
    plt.close(fig)

    file_size = output_path.stat().st_size
    print(f"Heatmap saved: {output_path}  ({file_size / 1024:.1f} KB)")
    return output_path


if __name__ == "__main__":
    variance_path = RESULTS_DIR / "variance.json"
    if not variance_path.exists():
        print(f"ERROR: {variance_path} not found.")
        print("Run run_experiment.py first, or run demo.py for a mock run.")
        sys.exit(1)

    generate_heatmap()
