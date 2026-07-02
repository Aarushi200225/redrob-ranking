#!/usr/bin/env python3
"""
rank.py — Main entry point for submission reproduction.

Usage:
    python rank.py --candidates ./data/candidates.jsonl --out ./outputs/ranked_output.csv
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Redrob Intelligent Candidate Ranking System"
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--jd",
        type=Path,
        default=Path("data/job_description.txt"),
    )
    args = parser.parse_args()

    if not args.candidates.exists():
        print(f"Error: candidates file not found: {args.candidates}")
        sys.exit(1)
    if not args.jd.exists():
        print(f"Error: JD not found: {args.jd}")
        sys.exit(1)

    output = run_pipeline(
        jd_path=args.jd,
        candidates_path=args.candidates,
        output_path=args.out,
    )
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
