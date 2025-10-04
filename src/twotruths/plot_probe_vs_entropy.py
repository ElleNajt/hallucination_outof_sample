#!/usr/bin/env python3
"""
Plot comparison of probe vs entropy baseline for two truths experiments.

Creates bar charts comparing accuracy of probe vs entropy across different aggregation methods.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_results(log_file: Path) -> List[Dict[str, Any]]:
    """Load results from a JSONL file, filtering out failed trials."""
    results = []
    with open(log_file) as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                if result.get("success", True):
                    results.append(result)
    return results


def calculate_accuracies(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Calculate accuracy for probe and entropy across avg/sum/max."""
    n = len(results)

    accuracies = {
        "probe": {},
        "entropy": {}
    }

    # Probe accuracies
    if any("probe_correct_avg" in r for r in results):
        accuracies["probe"]["avg"] = sum(1 for r in results if r.get("probe_correct_avg", False)) / n * 100
        accuracies["probe"]["sum"] = sum(1 for r in results if r.get("probe_correct_sum", False)) / n * 100
        accuracies["probe"]["max"] = sum(1 for r in results if r.get("probe_correct_max", False)) / n * 100

    # Entropy accuracies
    if any("entropy_correct_avg" in r for r in results):
        accuracies["entropy"]["avg"] = sum(1 for r in results if r.get("entropy_correct_avg", False)) / n * 100
        accuracies["entropy"]["sum"] = sum(1 for r in results if r.get("entropy_correct_sum", False)) / n * 100
        accuracies["entropy"]["max"] = sum(1 for r in results if r.get("entropy_correct_max", False)) / n * 100

    return accuracies


def plot_comparison(
    results: List[Dict[str, Any]],
    title: str,
    output_path: Path = None,
):
    """Create bar chart comparing probe vs entropy accuracy."""
    accuracies = calculate_accuracies(results)

    if not accuracies["probe"] or not accuracies["entropy"]:
        print("Missing probe or entropy data")
        return

    # Setup data
    methods = ["avg", "sum", "max"]
    probe_acc = [accuracies["probe"][m] for m in methods]
    entropy_acc = [accuracies["entropy"][m] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, probe_acc, width, label='Hallucination Probe', color='#2E86AB')
    bars2 = ax.bar(x + width/2, entropy_acc, width, label='Token Entropy (baseline)', color='#A23B72')

    # Add random chance line
    ax.axhline(y=100/3, color='gray', linestyle='--', alpha=0.5, label='Random chance (33.3%)')

    # Formatting
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Average', 'Sum', 'Max'])
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def print_summary(results: List[Dict[str, Any]], experiment_name: str):
    """Print summary statistics."""
    accuracies = calculate_accuracies(results)
    n = len(results)

    print(f"\n{'=' * 80}")
    print(f"SUMMARY: {experiment_name}")
    print(f"{'=' * 80}")
    print(f"Total trials: {n}")

    if accuracies["probe"]:
        print(f"\nProbe accuracy:")
        print(f"  avg: {accuracies['probe']['avg']:.1f}%")
        print(f"  sum: {accuracies['probe']['sum']:.1f}%")
        print(f"  max: {accuracies['probe']['max']:.1f}%")

    if accuracies["entropy"]:
        print(f"\nEntropy accuracy:")
        print(f"  avg: {accuracies['entropy']['avg']:.1f}%")
        print(f"  sum: {accuracies['entropy']['sum']:.1f}%")
        print(f"  max: {accuracies['entropy']['max']:.1f}%")

    if accuracies["probe"] and accuracies["entropy"]:
        print(f"\nProbe advantage (percentage points):")
        print(f"  avg: +{accuracies['probe']['avg'] - accuracies['entropy']['avg']:.1f}")
        print(f"  sum: +{accuracies['probe']['sum'] - accuracies['entropy']['sum']:.1f}")
        print(f"  max: +{accuracies['probe']['max'] - accuracies['entropy']['max']:.1f}")

    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot probe vs entropy comparison for two truths experiments"
    )
    parser.add_argument("log_file", type=str, help="Path to JSONL log file")
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title (default: derived from filename)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for plot (default: auto-generate)",
    )

    args = parser.parse_args()

    log_path = Path(args.log_file)

    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return

    # Load results
    print(f"Loading results from {log_path}...")
    results = load_results(log_path)

    if not results:
        print("No results found in log file")
        return

    # Determine title
    if args.title:
        title = args.title
    else:
        experiment_type = "Two Truths About Me" if "about_me" in str(log_path) else "Two Truths and a Lie"
        title = f"Probe vs Entropy Baseline\n{experiment_type} (N={len(results)})"

    # Print summary
    print_summary(results, title)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = log_path.parent / f"{log_path.stem}_probe_vs_entropy.png"

    # Create plot
    plot_comparison(results, title, output_path)


if __name__ == "__main__":
    main()
