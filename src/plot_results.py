#!/usr/bin/env python3
"""
Plot results from two truths and a lie experiments.

This script analyzes JSONL logs and plots:
1. Overall accuracy of probe predictions
2. Accuracy vs confidence (ratio of max score to second-highest score)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_results(log_file: Path) -> List[Dict[str, Any]]:
    """Load results from a JSONL file, filtering out failed trials."""
    results = []
    with open(log_file) as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                # Only include successful trials with required fields
                if result.get("success", True) and "statement_scores" in result:
                    results.append(result)
    return results


def calculate_confidence_metrics(result: Dict[str, Any]) -> Tuple[float, bool]:
    """
    Calculate confidence metric for a result.

    Returns:
        - confidence_ratio: ratio of max score to second-highest score
        - correct: whether probe was correct
    """
    scores = result["statement_scores"]

    # Convert string keys to int and get sorted scores
    score_list = [(int(k), v) for k, v in scores.items()]
    score_list.sort(key=lambda x: x[1], reverse=True)

    max_score = score_list[0][1]
    second_max_score = score_list[1][1] if len(score_list) > 1 else 0.0

    # Avoid division by zero
    if second_max_score == 0.0:
        confidence_ratio = float("inf") if max_score > 0 else 1.0
    else:
        confidence_ratio = max_score / second_max_score

    correct = result["probe_correct"]

    return confidence_ratio, correct


def plot_accuracy_by_confidence(
    results: List[Dict[str, Any]],
    title: str,
    output_path: Path = None,
    num_bins: int = 10,
):
    """
    Plot accuracy vs confidence ratio.

    Bins results by confidence ratio and plots accuracy for each bin.
    """
    # Calculate confidence metrics for all results
    data = [calculate_confidence_metrics(r) for r in results]
    confidence_ratios = [d[0] for d in data]
    correct = [d[1] for d in data]

    # Filter out infinite values for binning
    finite_data = [
        (c, corr) for c, corr in zip(confidence_ratios, correct) if c != float("inf")
    ]

    if not finite_data:
        print("No finite confidence ratios to plot")
        return

    finite_confidence = [d[0] for d in finite_data]
    finite_correct = [d[1] for d in finite_data]

    # Create bins based on confidence ratio
    min_conf = min(finite_confidence)
    max_conf = max(finite_confidence)

    bins = np.linspace(min_conf, max_conf, num_bins + 1)
    bin_indices = np.digitize(finite_confidence, bins)

    # Calculate accuracy for each bin
    bin_accuracies = []
    bin_centers = []
    bin_counts = []

    for i in range(1, num_bins + 1):
        bin_mask = bin_indices == i
        bin_results = [
            finite_correct[j] for j in range(len(finite_correct)) if bin_mask[j]
        ]

        if bin_results:
            accuracy = sum(bin_results) / len(bin_results)
            bin_accuracies.append(accuracy)
            bin_centers.append((bins[i - 1] + bins[i]) / 2)
            bin_counts.append(len(bin_results))

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot accuracy vs confidence
    ax1.plot(bin_centers, bin_accuracies, "o-", linewidth=2, markersize=8)
    ax1.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="Random chance")
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot histogram of confidence ratios
    ax2.bar(bin_centers, bin_counts, width=(bins[1] - bins[0]) * 0.8, alpha=0.6)
    ax2.set_xlabel("Confidence Ratio (max score / 2nd max score)", fontsize=12)
    ax2.set_ylabel("Number of trials", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def print_summary_stats(results: List[Dict[str, Any]], experiment_name: str):
    """Print summary statistics for the experiment."""
    print(f"\n{'=' * 80}")
    print(f"SUMMARY: {experiment_name}")
    print(f"{'=' * 80}")

    total = len(results)
    correct = sum(1 for r in results if r.get("probe_correct", False))
    accuracy = correct / total if total > 0 else 0

    print(f"Total trials: {total}")
    print(f"Probe correct: {correct}/{total} ({accuracy * 100:.1f}%)")

    # Calculate confidence metrics
    data = [calculate_confidence_metrics(r) for r in results]
    confidence_ratios = [d[0] for d in data if d[0] != float("inf")]

    if confidence_ratios:
        print(f"\nConfidence ratio statistics:")
        print(f"  Mean: {np.mean(confidence_ratios):.3f}")
        print(f"  Median: {np.median(confidence_ratios):.3f}")
        print(f"  Min: {np.min(confidence_ratios):.3f}")
        print(f"  Max: {np.max(confidence_ratios):.3f}")

    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot results from two truths and a lie experiments"
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
        help="Output path for plot (default: show plot)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of bins for confidence ratio (default: 10)",
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
        title = f"Probe Accuracy vs Confidence ({log_path.stem})"

    # Print summary statistics
    print_summary_stats(results, title)

    # Determine output path
    output_path = None
    if args.output:
        output_path = Path(args.output)
    elif not args.output:
        # Auto-generate output path next to log file
        output_path = log_path.parent / f"{log_path.stem}_plot.png"

    # Create plot
    plot_accuracy_by_confidence(results, title, output_path, args.bins)


if __name__ == "__main__":
    main()
