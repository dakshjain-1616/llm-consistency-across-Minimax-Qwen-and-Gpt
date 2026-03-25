"""
run_experiment.py — Main experiment runner.

Sends fixed prompts to 3 models (Qwen, Mistral, GPT) via OpenRouter,
multiple runs each, then computes normalized Levenshtein variance and
semantic (Jaccard) variance per (model, prompt).

Outputs:
  results/variance.json        — raw Levenshtein variance scores (backward-compat)
  results/raw_responses.json   — all individual responses
  results/experiment_meta.json — enriched metadata (timestamps, stats, semantic variance)
  results/results.csv          — flat CSV for spreadsheet/Pandas analysis (if --export-csv)

Usage:
  python run_experiment.py [--models qwen mistral gpt] [--prompts 50] [--runs 10]
                           [--export-csv] [--dry-run] [--version]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from consistency_heatmap_ import MODELS, query_model
from consistency_heatmap_ import SEED_PROMPTS
from consistency_heatmap_ import compute_semantic_variance_data, cross_model_divergence
from consistency_heatmap_ import compute_stats
from consistency_heatmap_ import compute_variance

_VERSION = "2.0.0"

NUM_RUNS = int(os.getenv("NUM_RUNS", "10"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results"))
INTER_REQUEST_DELAY = float(os.getenv("INTER_REQUEST_DELAY_SECONDS", "0.5"))

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
        import re
        plain = re.sub(r"\[/?[^\[\]]*\]", "", msg)
        print(plain)


def _print_startup_banner(models: dict, num_prompts: int, num_runs: int, results_dir: Path) -> None:
    """Print the startup banner for a live experiment run."""
    if _RICH and console:
        console.print(Panel.fit(
            f"[bold cyan]Consistency Heatmap[/bold cyan]  [dim]v{_VERSION}[/dim]\n"
            f"[dim]50 prompts × 10 runs × 3 models — variance visualized in 30s[/dim]\n"
            f"[dim]Built autonomously by [link=https://heyneo.so]NEO[/link] · your autonomous AI Agent[/dim]",
            border_style="cyan",
            padding=(0, 2),
        ))
        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("Key", style="dim")
        table.add_column("Value", style="cyan")
        table.add_row("Models", ", ".join(f"{n} ({mid})" for n, mid in models.items()))
        table.add_row("Prompts", str(num_prompts))
        table.add_row("Runs", f"{num_runs} per prompt per model")
        table.add_row("Output", str(results_dir))
        console.print(table)
    else:
        print("=" * 60)
        print(" Consistency Heatmap — Live Experiment")
        print("=" * 60)
        for n, mid in models.items():
            print(f"  {n}: {mid}")
        print(f"  Prompts : {num_prompts}")
        print(f"  Runs    : {num_runs} per prompt per model")
        print(f"  Output  : {results_dir}/")
        print("=" * 60)


def run_experiment(
    models: dict = None,
    prompts: list = None,
    num_runs: int = NUM_RUNS,
    results_dir: Path = RESULTS_DIR,
    session=None,
    verbose: bool = True,
    export_csv: bool = False,
) -> dict:
    """
    Execute the consistency experiment.

    Returns a dict with structure:
      {model_name: {prompt_index (str): variance_score (float), ...}, ...}
    """
    if models is None:
        models = MODELS
    if prompts is None:
        prompts = SEED_PROMPTS

    results_dir.mkdir(parents=True, exist_ok=True)

    raw_responses: dict = {name: {} for name in models}
    variance_data: dict = {name: {} for name in models}

    total_calls = len(models) * len(prompts) * num_runs
    call_count = 0
    t_start = time.monotonic()

    if _RICH and verbose:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            for model_name, model_id in models.items():
                task = progress.add_task(
                    f"[cyan]{model_name}[/cyan] ({model_id})",
                    total=len(prompts) * num_runs,
                )
                for p_idx, prompt in enumerate(prompts):
                    responses = []
                    for run_idx in range(num_runs):
                        call_count += 1
                        try:
                            response = query_model(model_id, prompt, session=session)
                            responses.append(response)
                        except RuntimeError as exc:
                            console.print(f"[yellow]  WARNING:[/yellow] {exc}")
                            responses.append("")
                        progress.advance(task)
                        inter_delay = float(os.getenv("INTER_REQUEST_DELAY_SECONDS", str(INTER_REQUEST_DELAY)))
                        if inter_delay > 0 and (run_idx < num_runs - 1 or p_idx < len(prompts) - 1):
                            time.sleep(inter_delay)
                    raw_responses[model_name][str(p_idx)] = responses
                    variance_data[model_name][str(p_idx)] = compute_variance(responses)
    else:
        for model_name, model_id in models.items():
            if verbose:
                print(f"\n{'='*60}")
                print(f"Model: {model_name} ({model_id})")
                print(f"{'='*60}")

            for p_idx, prompt in enumerate(prompts):
                responses = []
                for run_idx in range(num_runs):
                    call_count += 1
                    progress_pct = call_count / total_calls * 100
                    if verbose:
                        print(
                            f"  [{progress_pct:5.1f}%] {model_name} | prompt {p_idx+1:02d}/{len(prompts)} | run {run_idx+1}/{num_runs}",
                            end="\r",
                        )
                    try:
                        response = query_model(model_id, prompt, session=session)
                        responses.append(response)
                    except RuntimeError as exc:
                        if verbose:
                            print(f"\n  WARNING: {exc}")
                        responses.append("")

                    inter_delay = float(os.getenv("INTER_REQUEST_DELAY_SECONDS", str(INTER_REQUEST_DELAY)))
                    if inter_delay > 0 and (run_idx < num_runs - 1 or p_idx < len(prompts) - 1):
                        time.sleep(inter_delay)

                raw_responses[model_name][str(p_idx)] = responses
                variance_data[model_name][str(p_idx)] = compute_variance(responses)

            if verbose:
                print()

    elapsed = time.monotonic() - t_start

    variance_path = results_dir / "variance.json"
    raw_path = results_dir / "raw_responses.json"

    with open(variance_path, "w", encoding="utf-8") as f:
        json.dump(variance_data, f, indent=2)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_responses, f, indent=2)

    semantic_data = compute_semantic_variance_data(raw_responses)
    model_stats = {
        name: compute_stats(list(variance_data[name].values()))
        for name in models
    }
    divergence = cross_model_divergence(raw_responses)

    from consistency_heatmap_ import save_experiment_meta
    save_experiment_meta(
        results_dir / "experiment_meta.json",
        mode="live",
        models=models,
        num_runs=num_runs,
        num_prompts=len(prompts),
        variance_data=variance_data,
        semantic_data=semantic_data,
        cross_model_divergence=divergence,
        model_stats=model_stats,
        elapsed_seconds=elapsed,
    )

    if export_csv:
        from consistency_heatmap_ import export_variance_csv
        csv_path = export_variance_csv(
            variance_data, prompts,
            results_dir / "results.csv",
            semantic_data=semantic_data,
        )
        if verbose:
            _print(f"[green]✓[/green] CSV saved: {csv_path}")

    if verbose:
        _print(f"\n[green]Results saved to {results_dir}/[/green]")
        _print_summary(variance_data)

    return variance_data


def _print_summary(variance_data: dict) -> None:
    """Print a brief summary of mean variance per model."""
    if _RICH and console:
        table = Table(title="Variance Summary", box=box.ROUNDED)
        table.add_column("Model", style="cyan")
        table.add_column("Mean Variance", justify="right")
        table.add_column("Std Dev", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Verdict", style="dim")
        for model_name, scores in variance_data.items():
            vals = list(scores.values())
            if not vals:
                continue
            st = compute_stats(vals)
            from consistency_heatmap_ import verdict
            v = verdict(st["mean"])
            color = "green" if st["mean"] < 0.1 else ("yellow" if st["mean"] < 0.2 else "red")
            table.add_row(
                model_name,
                f"[{color}]{st['mean']:.4f}[/{color}]",
                f"{st['std']:.4f}",
                f"{st['min']:.4f}",
                f"{st['max']:.4f}",
                v,
            )
        console.print(table)
    else:
        print("\nVariance Summary (mean across all prompts):")
        print(f"  {'Model':<14} {'Mean Variance':>14} {'Std Dev':>8} {'Min':>8} {'Max':>8}")
        print(f"  {'-'*54}")
        for model_name, scores in variance_data.items():
            vals = list(scores.values())
            if not vals:
                continue
            st = compute_stats(vals)
            print(
                f"  {model_name:<14} {st['mean']:>14.4f} {st['std']:>8.4f} "
                f"{st['min']:>8.4f} {st['max']:>8.4f}"
            )


def _dry_run(models: dict, prompts: list, num_runs: int, results_dir: Path, export_csv: bool) -> None:
    """Print what would happen without making any API calls."""
    total = len(models) * len(prompts) * num_runs
    delay = float(os.getenv("INTER_REQUEST_DELAY_SECONDS", str(INTER_REQUEST_DELAY)))
    est_seconds = total * delay

    if _RICH and console:
        table = Table(title="Dry Run — no API calls will be made", box=box.ROUNDED)
        table.add_column("Setting", style="dim")
        table.add_column("Value", style="cyan")
        for n, mid in models.items():
            table.add_row(f"Model: {n}", mid)
        table.add_row("Prompts", str(len(prompts)))
        table.add_row("Runs", f"{num_runs} per prompt per model")
        table.add_row("API calls", str(total))
        table.add_row("Est. time", f"{est_seconds/60:.1f} min (at {delay}s delay/call)")
        table.add_row("Output", str(results_dir))
        files = "variance.json, raw_responses.json, experiment_meta.json"
        if export_csv:
            files += ", results.csv"
        table.add_row("Files", files)
        console.print(table)
    else:
        print("=" * 60)
        print(" DRY RUN — no API calls will be made")
        print("=" * 60)
        print(f"  Models    : {', '.join(f'{n} ({id_})' for n, id_ in models.items())}")
        print(f"  Prompts   : {len(prompts)}")
        print(f"  Runs      : {num_runs} per prompt per model")
        print(f"  API calls : {total}")
        print(f"  Est. time : {est_seconds/60:.1f} min (at {delay}s delay/call)")
        print(f"  Output    : {results_dir}/")
        print(f"    variance.json, raw_responses.json, experiment_meta.json")
        if export_csv:
            print(f"    results.csv")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consistency Heatmap — live experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version",
        version=f"Consistency Heatmap v{_VERSION}",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="MODEL",
        help="Model names to test (subset of: qwen mistral gpt). Default: all.",
    )
    parser.add_argument(
        "--prompts", type=int, default=int(os.getenv("NUM_PROMPTS", "50")),
        help="Number of prompts to use (1–50).",
    )
    parser.add_argument(
        "--runs", type=int, default=int(os.getenv("NUM_RUNS", "10")),
        help="Number of runs per prompt per model.",
    )
    parser.add_argument(
        "--export-csv", action="store_true",
        help="Also export results.csv alongside the JSON files.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be run without making any API calls.",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help="Directory to write output files.",
    )
    args = parser.parse_args()

    selected_models = dict(MODELS)
    if args.models:
        unknown = [m for m in args.models if m not in MODELS]
        if unknown:
            _print(f"[red]ERROR:[/red] unknown model(s): {unknown}. Available: {list(MODELS.keys())}")
            sys.exit(1)
        selected_models = {k: v for k, v in MODELS.items() if k in args.models}

    selected_prompts = SEED_PROMPTS[: max(1, min(args.prompts, len(SEED_PROMPTS)))]

    if args.dry_run:
        _dry_run(selected_models, selected_prompts, args.runs, args.results_dir, args.export_csv)
        sys.exit(0)

    if not os.getenv("OPENROUTER_API_KEY"):
        _print("[red]ERROR:[/red] OPENROUTER_API_KEY environment variable is not set.")
        _print("Set it with: [cyan]export OPENROUTER_API_KEY=your_key[/cyan]")
        _print("Or run [cyan]demo.py[/cyan] for a mock demonstration without an API key.")
        sys.exit(1)

    _print_startup_banner(selected_models, len(selected_prompts), args.runs, args.results_dir)

    with requests.Session() as session:
        run_experiment(
            models=selected_models,
            prompts=selected_prompts,
            num_runs=args.runs,
            results_dir=args.results_dir,
            session=session,
            verbose=True,
            export_csv=args.export_csv,
        )
