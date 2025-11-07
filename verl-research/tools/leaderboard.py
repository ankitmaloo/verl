#!/usr/bin/env python3
"""
Show leaderboard of all experiments.

Usage:
    python tools/leaderboard.py
    python tools/leaderboard.py --metric accuracy
"""

import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table


def load_results(experiments_dir: Path):
    """Load results from all experiments"""
    results = []

    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name == 'template':
            continue

        results_file = exp_dir / "results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
                data['experiment_id'] = exp_dir.name
                results.append(data)

    return results


def show_leaderboard(results, metric='accuracy', baseline_id='00_baseline'):
    """Display leaderboard table"""

    console = Console()

    # Find baseline value
    baseline_value = None
    for r in results:
        if r['experiment_id'] == baseline_id:
            baseline_value = r.get('results', {}).get('full_train', {}).get(metric)
            break

    # Sort by metric (descending)
    results_sorted = sorted(
        results,
        key=lambda x: x.get('results', {}).get('full_train', {}).get(metric, 0) or 0,
        reverse=True
    )

    # Create table
    table = Table(title="verl-research Leaderboard", show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Experiment", style="white", width=30)
    table.add_column(metric.title(), style="green", width=12)
    table.add_column("Δ vs Baseline", style="yellow", width=15)
    table.add_column("Status", style="blue", width=10)

    for i, result in enumerate(results_sorted, 1):
        exp_id = result['experiment_id']
        value = result.get('results', {}).get('full_train', {}).get(metric)
        status = result.get('status', 'unknown')

        # Rank emoji
        if i == 1:
            rank = "🥇"
        elif i == 2:
            rank = "🥈"
        elif i == 3:
            rank = "🥉"
        else:
            rank = str(i)

        # Format value
        if value is not None:
            value_str = f"{value:.2%}" if isinstance(value, float) and value < 1 else f"{value:.2f}"

            # Compute delta
            if baseline_value is not None and exp_id != baseline_id:
                delta = value - baseline_value
                delta_str = f"{delta:+.2%}" if isinstance(delta, float) and abs(delta) < 1 else f"{delta:+.2f}"
            else:
                delta_str = "—"
        else:
            value_str = "N/A"
            delta_str = "—"

        # Status emoji
        status_emoji = {
            'completed': '✅',
            'running': '🏃',
            'failed': '❌',
            'not_started': '🔵',
        }.get(status, '❓')

        table.add_row(rank, exp_id, value_str, delta_str, status_emoji)

    console.print(table)

    # Show baseline
    if baseline_value is not None:
        console.print(f"\n📍 Baseline ({baseline_id}): {baseline_value:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Show experiment leaderboard")
    parser.add_argument(
        "--metric",
        default="accuracy",
        help="Metric to rank by (default: accuracy)"
    )
    parser.add_argument(
        "--baseline",
        default="00_baseline",
        help="Baseline experiment ID"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Experiments directory"
    )

    args = parser.parse_args()

    # Default to ../experiments
    if args.dir is None:
        script_dir = Path(__file__).parent
        experiments_dir = script_dir.parent / "experiments"
    else:
        experiments_dir = args.dir

    # Load results
    results = load_results(experiments_dir)

    if not results:
        print("No experiments found.")
        return 1

    # Show leaderboard
    show_leaderboard(results, metric=args.metric, baseline_id=args.baseline)

    return 0


if __name__ == "__main__":
    exit(main())
