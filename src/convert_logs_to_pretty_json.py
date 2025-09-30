#!/usr/bin/env python3
"""Convert JSONL logs to pretty JSON format."""

import json
import argparse
from pathlib import Path


def convert_jsonl_to_pretty_json(jsonl_file):
    """Convert JSONL file to pretty JSON."""

    # Read all entries
    entries = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    # Write back as pretty JSON
    with open(jsonl_file, 'w') as f:
        json.dump(entries, f, indent=2)

    print(f"Converted {jsonl_file} to pretty JSON format ({len(entries)} entries)")


def main():
    parser = argparse.ArgumentParser(description='Convert JSONL logs to pretty JSON')
    parser.add_argument('files', nargs='+', type=str, help='Log files to convert')

    args = parser.parse_args()

    for file_path in args.files:
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: File not found: {path}")
            continue

        convert_jsonl_to_pretty_json(path)


if __name__ == '__main__':
    main()
