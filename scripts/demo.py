"""
demo.py — Runnable demo for the Consistency Heatmap project.

Auto-detects mock mode when OPENROUTER_API_KEY is not set.
Always saves real output files to outputs/.

Usage:
  python demo.py                          # mock mode (no API key needed)
  python demo.py --live                   # live mode (requires OPENROUTER_API_KEY)
  python demo.py --models qwen mistral    # test only specific models
  python demo.py --prompts 10 --runs 3   # smaller experiment
  python demo.py --dry-run               # show config without running
  python demo.py --version               # print version and exit
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from consistency_heatmap_ import SEED_PROMPTS
from consistency_heatmap_ import compute_variance
from generate_heatmap import generate_heatmap
from consistency_heatmap_ import compute_semantic_variance_data, cross_model_divergence
from consistency_heatmap_ import compute_stats, verdict
from consistency_heatmap_ import export_variance_csv, save_experiment_meta
from consistency_heatmap_ import generate_html_report

_VERSION = "2.0.0"

# ---------------------------------------------------------------------------
# Rich — graceful fallback if not installed
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
    from rich.table import Table
    from rich import box
    _RICH = True
    console = Console()
except ImportError:
    _RICH = False
    console = None  # type: ignore


def _print(msg: str, style: str = "") -> None:
    """Print with Rich styling when available, plain text otherwise."""
    if _RICH and console:
        console.print(msg, style=style)
    else:
        # Strip Rich markup for plain output
        import re
        plain = re.sub(r"\[/?[^\[\]]*\]", "", msg)
        print(plain)


def _print_banner(mode: str, num_prompts: int, num_runs: int, outputs_dir: Path) -> None:
    """Print startup banner."""
    if _RICH and console:
        console.print(Panel.fit(
            f"[bold cyan]Consistency Heatmap[/bold cyan]  [dim]v{_VERSION}[/dim]\n"
            f"[dim]50 prompts × 10 runs × 3 models — variance visualized in 30s[/dim]\n"
            f"[dim]Built autonomously by [link=https://heyneo.so]NEO[/link] · your autonomous AI Agent[/dim]",
            border_style="cyan",
            padding=(0, 2),
        ))
        mode_color = "green" if mode == "live" else "yellow"
        console.print(
            f"  Mode     [dim]:[/dim] [{mode_color}]{mode.upper()} MODE[/{mode_color}]\n"
            f"  Prompts  [dim]:[/dim] {num_prompts}\n"
            f"  Runs     [dim]:[/dim] {num_runs} per prompt per model\n"
            f"  Output   [dim]:[/dim] {outputs_dir}/\n"
        )
    else:
        print("=" * 60)
        print(f" Consistency Heatmap Demo  [{mode.upper()} MODE]")
        print("=" * 60)
        print(f" Prompts : {num_prompts}")
        print(f" Runs    : {num_runs} per prompt per model")
        print(f" Output  : {outputs_dir}/")
        print("=" * 60)
        print()


# ---------------------------------------------------------------------------
# Mock response generator
# ---------------------------------------------------------------------------

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


def _mock_response(prompt: str, seed: int) -> str:
    """Generate a plausible but slightly varied mock LLM response."""
    rng = random.Random(seed)
    template = rng.choice(_MOCK_TEMPLATES)
    noun = rng.choice(_NOUNS)
    domain = rng.choice(_DOMAINS)
    base = template.format(noun=noun, domain=domain)
    extras = [
        "",
        " See also related concepts.",
        " This is widely accepted.",
        f" Discovered in {rng.randint(1850, 2000)}.",
        " Further reading available.",
        " Multiple interpretations exist.",
    ]
    return base + rng.choice(extras)


def run_mock_experiment(
    models_cfg: dict,
    prompts: list,
    num_runs: int,
    results_dir: Path,
    verbose: bool = True,
    random_seed: int = 42,
) -> dict:
    """Run experiment with mock responses (no API calls)."""
    rng = random.Random(random_seed)

    raw_responses: dict = {name: {} for name in models_cfg}
    variance_data: dict = {name: {} for name in models_cfg}

    model_variance_bias = {
        name: rng.uniform(0.05, 0.35) for name in models_cfg
    }

    total = len(models_cfg) * len(prompts) * num_runs

    if _RICH and verbose:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Running mock experiment…", total=total)
            for model_name in models_cfg:
                bias = model_variance_bias[model_name]
                for p_idx, prompt in enumerate(prompts):
                    responses = []
                    for run_idx in range(num_runs):
                        prompt_stability = rng.uniform(0.0, 1.0)
                        if prompt_stability < bias:
                            seed = rng.randint(0, 10_000)
                        else:
                            seed = p_idx * 100 + int(prompt_stability * 10)
                        responses.append(_mock_response(prompt, seed))
                        progress.advance(task)
                    raw_responses[model_name][str(p_idx)] = responses
                    variance_data[model_name][str(p_idx)] = compute_variance(responses)
    else:
        done = 0
        for model_name in models_cfg:
            bias = model_variance_bias[model_name]
            if verbose:
                print(f"\n[MOCK] Model: {model_name}")
            for p_idx, prompt in enumerate(prompts):
                responses = []
                for run_idx in range(num_runs):
                    done += 1
                    prompt_stability = rng.uniform(0.0, 1.0)
                    if prompt_stability < bias:
                        seed = rng.randint(0, 10_000)
                    else:
                        seed = p_idx * 100 + int(prompt_stability * 10)
                    responses.append(_mock_response(prompt, seed))
                    if verbose and done % 100 == 0:
                        pct = done / total * 100
                        print(
                            f"  [{pct:5.1f}%] {model_name} prompt {p_idx+1:02d} run {run_idx+1}",
                            end="\r",
                        )
                raw_responses[model_name][str(p_idx)] = responses
                variance_data[model_name][str(p_idx)] = compute_variance(responses)
            if verbose:
                print()

    results_dir.mkdir(parents=True, exist_ok=True)

    variance_path = results_dir / "variance.json"
    raw_path = results_dir / "raw_responses.json"
    with open(variance_path, "w") as f:
        json.dump(variance_data, f, indent=2)
    with open(raw_path, "w") as f:
        json.dump(raw_responses, f, indent=2)

    semantic_data = compute_semantic_variance_data(raw_responses)
    model_stats = {
        name: compute_stats(list(variance_data[name].values()))
        for name in models_cfg
    }
    divergence = cross_model_divergence(raw_responses)

    save_experiment_meta(
        results_dir / "experiment_meta.json",
        mode="mock",
        models=models_cfg,
        num_runs=num_runs,
        num_prompts=len(prompts),
        variance_data=variance_data,
        semantic_data=semantic_data,
        cross_model_divergence=divergence,
        model_stats=model_stats,
    )

    variance_data["__semantic__"] = semantic_data
    variance_data["__divergence__"] = divergence
    variance_data["__model_stats__"] = model_stats
    return variance_data


def build_summary_report(variance_data: dict, output_path: Path, mode: str, num_runs: int) -> None:
    """Build a human-readable Markdown summary report."""
    clean = {k: v for k, v in variance_data.items() if not k.startswith("__")}

    lines = [
        "# Consistency Heatmap — Experiment Report",
        "",
        f"**Mode:** {mode}",
        f"**Prompts:** {len(SEED_PROMPTS)}",
        f"**Runs per prompt per model:** {num_runs}",
        "",
        "## Variance Summary",
        "",
        "| Model | Mean | Std Dev | Min | Max | Median | 95% CI | Verdict |",
        "|-------|------|---------|-----|-----|--------|--------|---------|",
    ]
    for model_name, scores in sorted(clean.items()):
        vals = list(scores.values())
        if not vals:
            continue
        st = compute_stats(vals)
        ci = f"[{st['ci_95_low']:.3f}, {st['ci_95_high']:.3f}]"
        v = verdict(st["mean"])
        lines.append(
            f"| {model_name} | {st['mean']:.4f} | {st['std']:.4f} | "
            f"{st['min']:.4f} | {st['max']:.4f} | {st['median']:.4f} | {ci} | {v} |"
        )

    lines += [
        "",
        "## Most Volatile Prompts (top 5 across all models)",
        "",
    ]

    prompt_scores: dict = {}
    for model_name, scores in clean.items():
        for p_idx_str, v in scores.items():
            prompt_scores.setdefault(int(p_idx_str), []).append(v)

    avg_prompt_scores = {
        idx: sum(vs) / len(vs) for idx, vs in prompt_scores.items()
    }
    top5 = sorted(avg_prompt_scores.items(), key=lambda x: -x[1])[:5]

    lines.append("| Rank | Prompt # | Prompt (truncated) | Mean Variance |")
    lines.append("|------|----------|--------------------|--------------|")
    for rank, (p_idx, v) in enumerate(top5, 1):
        prompt_text = SEED_PROMPTS[p_idx][:60] + ("..." if len(SEED_PROMPTS[p_idx]) > 60 else "")
        lines.append(f"| {rank} | {p_idx + 1} | {prompt_text} | {v:.4f} |")

    lines += ["", "---", "*Generated by Consistency Heatmap / NEO*", ""]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def _dry_run_info(models_cfg: dict, prompts: list, num_runs: int, outputs_dir: Path, mode: str) -> None:
    """Print dry-run configuration summary."""
    total = len(models_cfg) * len(prompts) * num_runs
    if _RICH and console:
        table = Table(title="Dry Run — nothing will be written", box=box.ROUNDED)
        table.add_column("Setting", style="dim")
        table.add_column("Value", style="cyan")
        table.add_row("Mode", f"[yellow]{mode.upper()}[/yellow]")
        table.add_row("Models", ", ".join(models_cfg.keys()))
        table.add_row("Prompts", str(len(prompts)))
        table.add_row("Runs", f"{num_runs} per prompt per model")
        table.add_row("API calls", f"{total} {'(all mocked)' if mode == 'mock' else ''}")
        table.add_row("Output dir", str(outputs_dir))
        console.print(table)
    else:
        print("=" * 60)
        print(f" DRY RUN — [{mode.upper()} MODE] — nothing will be written")
        print("=" * 60)
        print(f"  Models    : {', '.join(models_cfg.keys())}")
        print(f"  Prompts   : {len(prompts)}")
        print(f"  Runs      : {num_runs} per prompt per model")
        print(f"  API calls : {total} {'(all mocked)' if mode == 'mock' else ''}")
        print(f"  Output    : {outputs_dir}/")
        print("=" * 60)


def _print_file_summary(outputs_dir: Path) -> None:
    """Print summary of generated output files."""
    fnames = [
        "variance.json", "raw_responses.json", "experiment_meta.json",
        "heatmap.png", "report.md", "report.html", "results.csv",
    ]
    if _RICH and console:
        table = Table(title="[bold green]Done! Output files[/bold green]", box=box.ROUNDED)
        table.add_column("File", style="cyan")
        table.add_column("Size", justify="right", style="dim")
        for fname in fnames:
            p = outputs_dir / fname
            if p.exists():
                size = f"{p.stat().st_size / 1024:.1f} KB"
                table.add_row(str(p), size)
        console.print(table)
    else:
        print(f"\n{'='*60}")
        print(" Done! Output files:")
        for fname in fnames:
            p = outputs_dir / fname
            if p.exists():
                print(f"   {p}  ({p.stat().st_size / 1024:.1f} KB)")
        print("=" * 60)


def main(live: bool = False) -> None:
    """Run the consistency heatmap demo (mock or live mode)."""
    outputs_dir = Path(os.getenv("OUTPUTS_DIR", "outputs"))
    num_runs = int(os.getenv("NUM_RUNS", "10"))
    random_seed = int(os.getenv("RANDOM_SEED", "42"))

    has_api_key = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
    use_live = live and has_api_key

    if live and not has_api_key:
        _print("[yellow]WARNING:[/yellow] --live requested but OPENROUTER_API_KEY is not set. Falling back to mock mode.")

    mode = "live" if use_live else "mock"
    _print_banner(mode, len(SEED_PROMPTS), num_runs, outputs_dir)

    outputs_dir.mkdir(parents=True, exist_ok=True)

    from consistency_heatmap_ import MODELS as LIVE_MODELS

    variance_data: dict

    if use_live:
        import requests
        from run_experiment import run_experiment
        with requests.Session() as session:
            variance_data = run_experiment(
                models=LIVE_MODELS,
                prompts=SEED_PROMPTS,
                num_runs=num_runs,
                results_dir=outputs_dir,
                session=session,
                verbose=True,
            )
        raw_path = outputs_dir / "raw_responses.json"
        if raw_path.exists():
            with open(raw_path) as f:
                raw = json.load(f)
            semantic_data = compute_semantic_variance_data(raw)
            divergence = cross_model_divergence(raw)
        else:
            semantic_data = {}
            divergence = {}
        model_stats = {
            name: compute_stats(list(variance_data[name].values()))
            for name in LIVE_MODELS
        }
        variance_data["__semantic__"] = semantic_data
        variance_data["__divergence__"] = divergence
        variance_data["__model_stats__"] = model_stats
    else:
        variance_data = run_mock_experiment(
            models_cfg=LIVE_MODELS,
            prompts=SEED_PROMPTS,
            num_runs=num_runs,
            results_dir=outputs_dir,
            verbose=True,
            random_seed=random_seed,
        )

    semantic_data = variance_data.pop("__semantic__", {})
    divergence = variance_data.pop("__divergence__", {})
    model_stats = variance_data.pop("__model_stats__", {})

    _print("\n[cyan]Generating heatmap…[/cyan]")
    generate_heatmap(
        variance_path=outputs_dir / "variance.json",
        output_path=outputs_dir / "heatmap.png",
        title=f"LLM Consistency Heatmap [{mode.upper()}] — Normalized Levenshtein Variance",
    )
    _print("[green]✓[/green] Heatmap saved")

    _print("[cyan]Generating Markdown report…[/cyan]")
    report_path = outputs_dir / "report.md"
    build_summary_report(variance_data, report_path, mode, num_runs)
    _print(f"[green]✓[/green] Report saved: {report_path}")

    _print("[cyan]Generating HTML report…[/cyan]")
    from consistency_heatmap_ import MODELS as LIVE_MODELS  # noqa: F811
    html_path = generate_html_report(
        variance_data,
        SEED_PROMPTS,
        outputs_dir / "report.html",
        mode=mode,
        num_runs=num_runs,
        models=LIVE_MODELS,
        semantic_data=semantic_data,
        model_stats=model_stats,
        cross_model_divergence=divergence,
    )
    _print(f"[green]✓[/green] HTML report saved: {html_path}")

    _print("[cyan]Exporting CSV…[/cyan]")
    csv_path = export_variance_csv(
        variance_data,
        SEED_PROMPTS,
        outputs_dir / "results.csv",
        semantic_data=semantic_data,
    )
    _print(f"[green]✓[/green] CSV saved: {csv_path}")

    _print_file_summary(outputs_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consistency Heatmap Demo — 50 prompts × 10 runs × 3 models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version",
        version=f"Consistency Heatmap v{_VERSION}",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Use live OpenRouter API (requires OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="MODEL",
        help="Restrict to these model names (e.g. --models qwen gpt).",
    )
    parser.add_argument(
        "--prompts", type=int, default=None,
        help="Use only the first N prompts (default: all 50).",
    )
    parser.add_argument(
        "--runs", type=int, default=None,
        help="Number of runs per prompt (overrides NUM_RUNS env var).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be run without executing anything.",
    )
    args = parser.parse_args()

    if args.prompts is not None:
        os.environ["NUM_PROMPTS"] = str(args.prompts)
    if args.runs is not None:
        os.environ["NUM_RUNS"] = str(args.runs)

    if args.dry_run:
        from consistency_heatmap_ import MODELS as LIVE_MODELS
        has_key = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
        use_live = args.live and has_key
        n_runs = int(os.getenv("NUM_RUNS", "10"))
        n_prompts = int(os.getenv("NUM_PROMPTS", "50"))
        selected = (
            {k: v for k, v in LIVE_MODELS.items() if k in args.models}
            if args.models else LIVE_MODELS
        )
        _dry_run_info(selected, SEED_PROMPTS[:n_prompts], n_runs,
                      Path(os.getenv("OUTPUTS_DIR", "outputs")),
                      "live" if use_live else "mock")
        sys.exit(0)

    main(live=args.live)
