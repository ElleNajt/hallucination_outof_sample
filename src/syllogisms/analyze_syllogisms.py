#!/usr/bin/env python3
"""Analyze syllogistic reasoning experiment results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def analyze_results(log_file):
    """Analyze results from syllogistic reasoning log file."""

    # Load all trials (handle both JSONL and JSON array formats)
    with open(log_file, "r") as f:
        content = f.read()
        if content.strip().startswith("["):
            # JSON array format
            trials = json.loads(content)
        else:
            # JSONL format
            trials = []
            for line in content.split("\n"):
                if line.strip():
                    trials.append(json.loads(line))

    # Collect statistics
    valid_correct = 0
    valid_total = 0
    invalid_correct = 0
    invalid_total = 0

    valid_scores = []
    invalid_scores = []

    for trial in trials:
        # Analyze valid syllogisms (should have LOW probe scores)
        for valid in trial["valid_syllogisms"]:
            valid_total += 1
            valid_scores.append(valid["avg_probe_score"])

        # Analyze invalid syllogisms (should have HIGH probe scores)
        for invalid in trial["invalid_syllogisms"]:
            invalid_total += 1
            invalid_scores.append(invalid["avg_probe_score"])

    # Calculate if probe correctly identifies invalid vs valid
    # A threshold approach: find optimal threshold
    all_scores = [(s, "valid") for s in valid_scores] + [
        (s, "invalid") for s in invalid_scores
    ]

    # Try different thresholds
    best_threshold = 0
    best_accuracy = 0

    for threshold in np.linspace(0, max(valid_scores + invalid_scores), 100):
        correct = 0
        total = len(all_scores)

        for score, label in all_scores:
            if label == "valid" and score < threshold:
                correct += 1
            elif label == "invalid" and score >= threshold:
                correct += 1

        accuracy = correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    # Calculate statistics with best threshold
    for score in valid_scores:
        if score < best_threshold:
            valid_correct += 1

    for score in invalid_scores:
        if score >= best_threshold:
            invalid_correct += 1

    print(f"\n=== Syllogistic Reasoning Analysis ===")
    print(f"Log file: {log_file}")
    print(f"Number of trials: {len(trials)}")
    print(f"\nOptimal threshold: {best_threshold:.4f}")
    print(f"\nValid syllogisms (should have LOW scores):")
    print(f"  Total: {valid_total}")
    print(f"  Correctly identified (score < threshold): {valid_correct}")
    print(f"  Accuracy: {valid_correct / valid_total * 100:.1f}%")
    print(f"  Mean score: {np.mean(valid_scores):.4f}")
    print(f"  Median score: {np.median(valid_scores):.4f}")
    print(f"  Std: {np.std(valid_scores):.4f}")

    print(f"\nInvalid syllogisms (should have HIGH scores):")
    print(f"  Total: {invalid_total}")
    print(f"  Correctly identified (score >= threshold): {invalid_correct}")
    print(f"  Accuracy: {invalid_correct / invalid_total * 100:.1f}%")
    print(f"  Mean score: {np.mean(invalid_scores):.4f}")
    print(f"  Median score: {np.median(invalid_scores):.4f}")
    print(f"  Std: {np.std(invalid_scores):.4f}")

    print(f"\nOverall accuracy: {best_accuracy * 100:.1f}%")
    print(
        f"Total correct: {valid_correct + invalid_correct}/{valid_total + invalid_total}"
    )

    return {
        "valid_scores": valid_scores,
        "invalid_scores": invalid_scores,
        "best_threshold": best_threshold,
        "best_accuracy": best_accuracy,
        "valid_correct": valid_correct,
        "valid_total": valid_total,
        "invalid_correct": invalid_correct,
        "invalid_total": invalid_total,
    }


def plot_results(stats, output_file):
    """Create visualization of results."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Distribution of scores
    ax = axes[0, 0]
    ax.hist(
        stats["valid_scores"],
        bins=30,
        alpha=0.6,
        label="Valid syllogisms",
        color="green",
    )
    ax.hist(
        stats["invalid_scores"],
        bins=30,
        alpha=0.6,
        label="Invalid syllogisms",
        color="red",
    )
    ax.axvline(
        stats["best_threshold"],
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Threshold: {stats['best_threshold']:.4f}",
    )
    ax.set_xlabel("Average Probe Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Probe Scores")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Box plot comparison
    ax = axes[0, 1]
    data = [stats["valid_scores"], stats["invalid_scores"]]
    bp = ax.boxplot(data, labels=["Valid", "Invalid"], patch_artist=True)
    bp["boxes"][0].set_facecolor("green")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("red")
    bp["boxes"][1].set_alpha(0.6)
    ax.axhline(
        stats["best_threshold"],
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Threshold: {stats['best_threshold']:.4f}",
    )
    ax.set_ylabel("Average Probe Score")
    ax.set_title("Probe Scores by Syllogism Type")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Accuracy summary
    ax = axes[1, 0]
    categories = ["Valid\nSyllogisms", "Invalid\nSyllogisms", "Overall"]
    accuracies = [
        stats["valid_correct"] / stats["valid_total"] * 100,
        stats["invalid_correct"] / stats["invalid_total"] * 100,
        stats["best_accuracy"] * 100,
    ]
    colors = ["green", "red", "blue"]
    bars = ax.bar(categories, accuracies, color=colors, alpha=0.6)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Classification Accuracy")
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Plot 4: Summary text
    ax = axes[1, 1]
    ax.axis("off")
    summary_text = f"""
    Syllogistic Reasoning Results

    Total Trials: {len(stats["valid_scores"]) + len(stats["invalid_scores"])}

    Valid Syllogisms:
      • Total: {stats["valid_total"]}
      • Correctly identified: {stats["valid_correct"]}
      • Accuracy: {stats["valid_correct"] / stats["valid_total"] * 100:.1f}%
      • Mean score: {np.mean(stats["valid_scores"]):.4f}

    Invalid Syllogisms:
      • Total: {stats["invalid_total"]}
      • Correctly identified: {stats["invalid_correct"]}
      • Accuracy: {stats["invalid_correct"] / stats["invalid_total"] * 100:.1f}%
      • Mean score: {np.mean(stats["invalid_scores"]):.4f}

    Overall Accuracy: {stats["best_accuracy"] * 100:.1f}%

    Optimal Threshold: {stats["best_threshold"]:.4f}

    Random Chance: 50.0%
    """
    ax.text(
        0.1,
        0.5,
        summary_text,
        fontsize=12,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze syllogistic reasoning results"
    )
    parser.add_argument("log_file", type=str, help="Path to log file")
    parser.add_argument(
        "--output",
        type=str,
        help="Output plot file (default: same as log file with _plot.png)",
    )

    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            log_path.with_suffix("")
            .with_suffix(".png")
            .with_name(log_path.stem + "_plot.png")
        )

    stats = analyze_results(log_path)
    plot_results(stats, output_path)


if __name__ == "__main__":
    main()
