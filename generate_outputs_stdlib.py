"""
Standalone output generator using only Python stdlib.
Generates variance.json, raw_responses.json, report.md, and heatmap.png
without requiring matplotlib or python-Levenshtein.
"""
import itertools
import json
import os
import random
import struct
import zlib
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", "outputs"))
NUM_RUNS = int(os.getenv("NUM_RUNS", "10"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
NUM_PROMPTS = int(os.getenv("NUM_PROMPTS", "50"))

SEED_PROMPTS = [
    "What is the capital of France?",
    "Explain the concept of entropy in thermodynamics.",
    "Write a haiku about the ocean.",
    "What is 17 multiplied by 23?",
    "Describe the water cycle in two sentences.",
    "What is the difference between a virus and a bacterium?",
    "Name three primary colors.",
    "Explain recursion to a 10-year-old.",
    "What is the Pythagorean theorem?",
    "Who wrote the play Hamlet?",
    "What is photosynthesis?",
    "Define machine learning in one sentence.",
    "What is the speed of light in a vacuum?",
    "List the first five prime numbers.",
    "What is the boiling point of water at sea level?",
    "Explain what DNA stands for.",
    "What year did World War II end?",
    "What is Newton's first law of motion?",
    "Describe the function of the mitochondria.",
    "What is the largest planet in our solar system?",
    "What is the chemical formula for water?",
    "Explain the concept of gravity briefly.",
    "What is the square root of 144?",
    "Who painted the Mona Lisa?",
    "What does CPU stand for?",
    "Explain what a black hole is.",
    "What is the Fibonacci sequence?",
    "What is the difference between weather and climate?",
    "Name the four fundamental forces of nature.",
    "What is Ohm's law?",
    "Explain what an algorithm is.",
    "What is the distance from Earth to the Moon approximately?",
    "Who discovered penicillin?",
    "What is the greenhouse effect?",
    "Explain what inflation means in economics.",
    "What is a neural network?",
    "What are the three states of matter?",
    "Who wrote 'A Brief History of Time'?",
    "What is the human genome?",
    "Explain what blockchain technology is.",
    "What is the theory of relativity?",
    "What is a prime number?",
    "Describe what an API is.",
    "What is the periodic table?",
    "Who invented the telephone?",
    "What is quantum mechanics?",
    "Explain what a derivative is in calculus.",
    "What is the Big Bang theory?",
    "What does RAM stand for in computing?",
    "Explain what open source software means.",
]

MODELS = {
    "gpt": os.getenv("MODEL_GPT", "openai/gpt-4.1-mini"),
    "mistral": os.getenv("MODEL_MISTRAL", "mistralai/mistral-small-3.1-24b-instruct"),
    "qwen": os.getenv("MODEL_QWEN", "qwen/qwen3-72b"),
}

_MOCK_TEMPLATES = [
    "The answer is {noun}. This is a well-known fact in {domain}.",
    "{noun} refers to the concept of {domain} in scientific literature.",
    "In essence, {noun} is defined as {domain} by most authorities.",
    "Simply put, {noun} describes {domain}.",
    "{noun}: {domain}.",
    "According to standard definitions, {noun} is {domain}.",
    "The term {noun} is broadly understood as {domain}.",
    "{noun} can be described as {domain} in the context of science.",
    "Most textbooks define {noun} as {domain}.",
    "{domain} — that is what {noun} fundamentally represents.",
]

_NOUNS = [
    "entropy", "photosynthesis", "recursion", "gravity", "DNA",
    "an algorithm", "quantum entanglement", "thermodynamics", "the nucleus",
    "momentum", "velocity", "frequency", "amplitude", "resonance",
]

_DOMAINS = [
    "physics", "biology", "computer science", "chemistry", "mathematics",
    "information theory", "quantum mechanics", "classical mechanics",
    "molecular biology", "number theory", "calculus", "statistics",
]


def mock_response(prompt: str, seed: int) -> str:
    rng = random.Random(seed)
    template = rng.choice(_MOCK_TEMPLATES)
    noun = rng.choice(_NOUNS)
    domain = rng.choice(_DOMAINS)
    base = template.format(noun=noun, domain=domain)
    extras = [
        "", " See also related concepts.", " This is widely accepted.",
        f" Discovered in {rng.randint(1850, 2000)}.", " Further reading available.",
        " Multiple interpretations exist.",
    ]
    return base + rng.choice(extras)


# ── Pure Python Levenshtein ───────────────────────────────────────────────────
def levenshtein(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def norm_lev(s1: str, s2: str) -> float:
    if not s1 and not s2: return 0.0
    mx = max(len(s1), len(s2))
    return levenshtein(s1, s2) / mx if mx else 0.0


def compute_variance(responses: list) -> float:
    if len(responses) < 2: return 0.0
    pairs = list(itertools.combinations(responses, 2))
    return sum(norm_lev(a, b) for a, b in pairs) / len(pairs)


# ── Mock experiment ──────────────────────────────────────────────────────────
def run_mock():
    rng = random.Random(RANDOM_SEED)
    model_variance_bias = {name: rng.uniform(0.05, 0.35) for name in MODELS}
    raw_responses = {name: {} for name in MODELS}
    variance_data = {name: {} for name in MODELS}

    for model_name in MODELS:
        bias = model_variance_bias[model_name]
        print(f"[MOCK] Model: {model_name}  (bias={bias:.3f})")
        for p_idx, prompt in enumerate(SEED_PROMPTS):
            responses = []
            for run_idx in range(NUM_RUNS):
                prompt_stability = rng.uniform(0.0, 1.0)
                if prompt_stability < bias:
                    seed = rng.randint(0, 10_000)
                else:
                    seed = p_idx * 100 + int(prompt_stability * 10)
                responses.append(mock_response(prompt, seed))
            raw_responses[model_name][str(p_idx)] = responses
            variance_data[model_name][str(p_idx)] = compute_variance(responses)
            if p_idx % 10 == 9:
                print(f"  prompt {p_idx+1}/50 done")

    return variance_data, raw_responses


# ── Minimal PNG writer (stdlib only) ─────────────────────────────────────────
def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    chunk = chunk_type + data
    return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)


def write_heatmap_png(variance_data: dict, output_path: Path) -> None:
    """Write a real PNG heatmap using only stdlib (struct + zlib)."""
    model_names = sorted(variance_data.keys())
    n_models = len(model_names)
    n_prompts = NUM_PROMPTS

    # Build variance matrix
    matrix = []
    for model in model_names:
        row = [variance_data[model].get(str(i), 0.0) for i in range(n_prompts)]
        matrix.append(row)

    # Image dimensions
    cell_w = 18   # pixels per prompt cell
    cell_h = 40   # pixels per model row
    label_w = 80  # left label area
    colorbar_w = 30
    top_pad = 40
    bottom_pad = 30

    img_w = label_w + n_prompts * cell_w + colorbar_w + 20
    img_h = top_pad + n_models * cell_h + bottom_pad

    # Create RGB pixel array (list of rows, each row is list of (R,G,B))
    pixels = [[(240, 240, 240)] * img_w for _ in range(img_h)]

    def yor_color(v: float):
        """YlOrRd colormap approximation."""
        v = max(0.0, min(1.0, v))
        if v < 0.25:
            t = v / 0.25
            return (int(255), int(255 - t * 27), int(204 - t * 204))
        elif v < 0.5:
            t = (v - 0.25) / 0.25
            return (int(255), int(228 - t * 102), int(0))
        elif v < 0.75:
            t = (v - 0.5) / 0.25
            return (int(255 - t * 55), int(126 - t * 126), int(0))
        else:
            t = (v - 0.75) / 0.25
            return (int(200 - t * 55), int(0), int(0))

    # Draw heatmap cells
    for row_idx, model in enumerate(model_names):
        y0 = top_pad + row_idx * cell_h
        for p_idx in range(n_prompts):
            x0 = label_w + p_idx * cell_w
            v = matrix[row_idx][p_idx]
            r, g, b = yor_color(v)
            for y in range(y0, y0 + cell_h):
                for x in range(x0, x0 + cell_w - 1):
                    pixels[y][x] = (r, g, b)
            # Cell border
            for y in range(y0, y0 + cell_h):
                if x0 + cell_w - 1 < img_w:
                    pixels[y][x0 + cell_w - 1] = (200, 200, 200)

        # Row border
        if y0 + cell_h < img_h:
            for x in range(label_w, label_w + n_prompts * cell_w):
                pixels[y0 + cell_h - 1][x] = (200, 200, 200)

    # Draw colorbar
    bar_x = label_w + n_prompts * cell_w + 10
    bar_y0 = top_pad
    bar_h = n_models * cell_h
    for y in range(bar_y0, bar_y0 + bar_h):
        v = 1.0 - (y - bar_y0) / bar_h
        r, g, b = yor_color(v)
        for x in range(bar_x, min(bar_x + 15, img_w)):
            pixels[y][x] = (r, g, b)

    # Encode as PNG
    def pack_row(row):
        return b'\x00' + bytes([c for px in row for c in px])

    raw = b''.join(pack_row(pixels[y]) for y in range(img_h))
    compressed = zlib.compress(raw, 9)

    png_data = b'\x89PNG\r\n\x1a\n'
    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", img_w, img_h, 8, 2, 0, 0, 0)
    png_data += _png_chunk(b'IHDR', ihdr_data)
    # IDAT
    png_data += _png_chunk(b'IDAT', compressed)
    # IEND
    png_data += _png_chunk(b'IEND', b'')

    with open(output_path, 'wb') as f:
        f.write(png_data)

    print(f"Heatmap saved: {output_path}  ({len(png_data)/1024:.1f} KB)")


def build_report(variance_data: dict, output_path: Path) -> None:
    lines = [
        "# Consistency Heatmap — Experiment Report",
        "",
        "**Mode:** mock (RANDOM_SEED=42, no API key required)",
        f"**Prompts:** {NUM_PROMPTS}",
        f"**Runs per prompt per model:** {NUM_RUNS}",
        "",
        "## Variance Summary",
        "",
        "| Model | Mean Variance | Min | Max | Verdict |",
        "|-------|--------------|-----|-----|---------|",
    ]
    for model_name, scores in sorted(variance_data.items()):
        vals = list(scores.values())
        mean_v = sum(vals) / len(vals)
        verdict = (
            "Very consistent" if mean_v < 0.1
            else "Consistent" if mean_v < 0.2
            else "Moderate variance" if mean_v < 0.35
            else "High variance"
        )
        lines.append(
            f"| {model_name} | {mean_v:.4f} | {min(vals):.4f} | {max(vals):.4f} | {verdict} |"
        )

    lines += ["", "## Most Volatile Prompts (top 5 across all models)", ""]
    prompt_scores = {}
    for model_name, scores in variance_data.items():
        for p_idx_str, v in scores.items():
            prompt_scores.setdefault(int(p_idx_str), []).append(v)
    avg = {idx: sum(vs) / len(vs) for idx, vs in prompt_scores.items()}
    top5 = sorted(avg.items(), key=lambda x: -x[1])[:5]

    lines.append("| Rank | Prompt # | Prompt (truncated) | Mean Variance |")
    lines.append("|------|----------|--------------------|--------------|")
    for rank, (p_idx, v) in enumerate(top5, 1):
        pt = SEED_PROMPTS[p_idx][:60] + ("..." if len(SEED_PROMPTS[p_idx]) > 60 else "")
        lines.append(f"| {rank} | {p_idx + 1} | {pt} | {v:.4f} |")

    lines += ["", "---", "*Generated by Consistency Heatmap / NEO*", ""]
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print(" Consistency Heatmap — Stdlib Output Generator")
    print("=" * 60)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nRunning mock experiment...")
    variance_data, raw_responses = run_mock()

    variance_path = OUTPUTS_DIR / "variance.json"
    raw_path = OUTPUTS_DIR / "raw_responses.json"
    with open(variance_path, "w") as f:
        json.dump(variance_data, f, indent=2)
    print(f"Variance saved: {variance_path}")
    with open(raw_path, "w") as f:
        json.dump(raw_responses, f, indent=2)
    print(f"Raw responses saved: {raw_path}")

    print("\nGenerating PNG heatmap...")
    write_heatmap_png(variance_data, OUTPUTS_DIR / "heatmap.png")

    print("\nGenerating report...")
    build_report(variance_data, OUTPUTS_DIR / "report.md")

    print("\n" + "=" * 60)
    print(" Done!")
    for fname in ["variance.json", "raw_responses.json", "heatmap.png", "report.md"]:
        p = OUTPUTS_DIR / fname
        if p.exists():
            print(f"   {p}  ({p.stat().st_size / 1024:.1f} KB)")
    print("=" * 60)
