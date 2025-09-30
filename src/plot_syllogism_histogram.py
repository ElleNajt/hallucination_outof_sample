#!/usr/bin/env python3
"""Plot histogram of probe scores for valid vs invalid syllogisms."""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(log_file, output_file=None, bins=30, score_type='avg'):
    """Plot histogram of probe scores.

    Args:
        score_type: 'avg', 'sum', or 'max'
    """

    # Load all trials (handle both JSONL and JSON array formats)
    with open(log_file, 'r') as f:
        content = f.read()
        if content.strip().startswith('['):
            # JSON array format
            trials = json.loads(content)
        else:
            # JSONL format
            trials = []
            for line in content.split('\n'):
                if line.strip():
                    trials.append(json.loads(line))

    # Collect scores
    valid_scores = []
    invalid_scores = []

    score_key = f'{score_type}_probe_score'

    for trial in trials:
        for valid in trial['valid_syllogisms']:
            valid_scores.append(valid[score_key])

        for invalid in trial['invalid_syllogisms']:
            invalid_scores.append(invalid[score_key])

    # Create histogram
    plt.figure(figsize=(10, 6))

    plt.hist(valid_scores, bins=bins, alpha=0.7, label='Valid syllogisms', color='blue', edgecolor='black')
    plt.hist(invalid_scores, bins=bins, alpha=0.7, label='Invalid syllogisms', color='red', edgecolor='black')

    score_type_labels = {'avg': 'Average', 'sum': 'Sum', 'max': 'Max'}
    plt.xlabel(f'{score_type_labels[score_type]} Probe Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distribution of Hallucination Probe Scores ({score_type_labels[score_type]})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f'Valid: n={len(valid_scores)}, mean={np.mean(valid_scores):.4f}, std={np.std(valid_scores):.4f}\n'
    stats_text += f'Invalid: n={len(invalid_scores)}, mean={np.mean(invalid_scores):.4f}, std={np.std(invalid_scores):.4f}'
    plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        output_path = Path(log_file).with_suffix('.png').with_name(Path(log_file).stem + f'_histogram_{score_type}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

    plt.close()

    # Print summary
    print(f"\nSummary ({score_type}):")
    print(f"Valid syllogisms (n={len(valid_scores)}): mean={np.mean(valid_scores):.4f}, std={np.std(valid_scores):.4f}")
    print(f"Invalid syllogisms (n={len(invalid_scores)}): mean={np.mean(invalid_scores):.4f}, std={np.std(invalid_scores):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Plot histogram of probe scores')
    parser.add_argument('log_file', type=str, help='Path to log file')
    parser.add_argument('--output', type=str, help='Output plot file')
    parser.add_argument('--bins', type=int, default=30, help='Number of bins (default: 30)')
    parser.add_argument('--score-type', type=str, default='avg', choices=['avg', 'sum', 'max'], help='Score type to plot (default: avg)')

    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return

    plot_histogram(log_path, args.output, args.bins, args.score_type)


if __name__ == '__main__':
    main()
