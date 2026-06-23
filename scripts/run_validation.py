"""
scripts/run_validation.py
─────────────────────────
Standalone validator for ranked_output.csv against hackathon spec.

Usage:
    python scripts/run_validation.py outputs/ranked_output.csv data/candidates.jsonl.gz
    make validate
"""

import gzip
import sys
from pathlib import Path

import orjson
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils.logger import get_logger

log = get_logger("validator")

REQUIRED_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]


def load_valid_ids(candidates_path: Path) -> set:
    """Stream JSONL to collect all valid candidate_ids."""
    valid_ids = set()
    open_fn = gzip.open if str(candidates_path).endswith(".gz") else open

    with open_fn(candidates_path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                record = orjson.loads(line)
                valid_ids.add(record["candidate_id"])

    return valid_ids


def validate(csv_path: Path, candidates_path: Path) -> bool:
    """Run all format checks. Returns True if valid."""
    passed = 0
    total  = 0

    def check(condition: bool, message: str) -> bool:
        nonlocal passed, total
        total += 1
        if condition:
            log.info(f"  ✓ {message}")
            passed += 1
        else:
            log.error(f"  ✗ {message}")
        return condition

    log.info(f"Validating: {csv_path}")
    log.info("─" * 50)

    check(csv_path.suffix == ".csv", f"File is .csv (got {csv_path.suffix})")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        log.error(f"Cannot read CSV: {e}")
        return False

    check(len(df) == 100,
          f"Exactly 100 rows (got {len(df)})")
    check(list(df.columns) == REQUIRED_COLUMNS,
          f"Columns in order {REQUIRED_COLUMNS}")

    if list(df.columns) != REQUIRED_COLUMNS:
        log.error("Aborting — column mismatch prevents further checks")
        return False

    check(df["rank"].tolist() == list(range(1, 101)),
          "Ranks 1-100, no gaps, no duplicates")
    check(df["candidate_id"].nunique() == 100,
          "No duplicate candidate_ids")

    if candidates_path.exists():
        log.info("Checking candidate_ids against source ...")
        valid_ids = load_valid_ids(candidates_path)
        bad_ids   = set(df["candidate_id"]) - valid_ids
        check(len(bad_ids) == 0,
              f"All candidate_ids valid (bad: {bad_ids or 'none'})")
    else:
        log.warning("Candidates file not found — skipping ID check")

    diffs = df["score"].diff().dropna()
    check((diffs <= 1e-9).all(),
          "Scores monotonically non-increasing")
    check(df["score"].nunique() > 1,
          "Scores are not all the same value")
    check(
        df["reasoning"].notna().all()
        and (df["reasoning"].str.strip() != "").all(),
        "Reasoning non-empty for all 100 rows",
    )

    log.info("─" * 50)
    if passed == total:
        log.info(f"✓ ALL {total} CHECKS PASSED — ready to submit")
        return True
    else:
        log.error(f"✗ {total - passed}/{total} CHECKS FAILED")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_validation.py <csv_path> [candidates_path]")
        sys.exit(1)

    csv_path        = Path(sys.argv[1])
    candidates_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/candidates.jsonl")

    sys.exit(0 if validate(csv_path, candidates_path) else 1)