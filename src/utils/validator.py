"""
utils/validator.py
──────────────────
Hard format assertions for ranked_output.csv.

Called as the final step before CSV write in Stage 5.
Raises AssertionError immediately on any violation —
invalid output is worse than no output (auto-validator
rejects on any format failure, scoring zero).

Checks every requirement from the hackathon spec:
  ✓ Exactly 100 rows
  ✓ Columns in correct order
  ✓ Ranks 1-100, unique, no gaps
  ✓ No duplicate candidate_ids
  ✓ All candidate_ids exist in source JSONL
  ✓ Scores monotonically non-increasing
  ✓ Scores not all the same value
  ✓ Reasoning column non-empty for all rows
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.logger import get_logger

log = get_logger(__name__)

REQUIRED_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]


def validate_output(
    df: pd.DataFrame,
    valid_ids: set,
) -> None:
    """
    Run all format checks against the output DataFrame.

    Parameters
    ----------
    df        : Output DataFrame before CSV write.
    valid_ids : Set of all valid candidate_ids from source JSONL.

    Raises
    ------
    AssertionError if any check fails — with a clear message.
    """
    log.info("Running output format validation ...")

    # ── Row count ─────────────────────────────────────────────────────────────
    assert len(df) == 100, (
        f"Expected exactly 100 rows, got {len(df)}"
    )

    # ── Column names and order ────────────────────────────────────────────────
    assert list(df.columns) == REQUIRED_COLUMNS, (
        f"Columns must be {REQUIRED_COLUMNS} in order, got {list(df.columns)}"
    )

    # ── Ranks ─────────────────────────────────────────────────────────────────
    assert df["rank"].tolist() == list(range(1, 101)), (
        "Ranks must be integers 1-100 with no gaps or duplicates"
    )

    # ── Candidate ID uniqueness ───────────────────────────────────────────────
    assert df["candidate_id"].nunique() == 100, (
        f"Duplicate candidate_ids found: "
        f"{df[df['candidate_id'].duplicated()]['candidate_id'].tolist()}"
    )

    # ── Candidate IDs exist in source ────────────────────────────────────────
    bad_ids = set(df["candidate_id"]) - valid_ids
    assert len(bad_ids) == 0, (
        f"candidate_ids not found in source JSONL: {bad_ids}"
    )

    # ── Scores monotonically non-increasing ──────────────────────────────────
    diffs = df["score"].diff().dropna()
    assert (diffs <= 1e-9).all(), (
        "Scores must be monotonically non-increasing "
        f"(violations at ranks: "
        f"{df[df['score'].diff() > 1e-9]['rank'].tolist()})"
    )

    # ── Scores not all same ───────────────────────────────────────────────────
    assert df["score"].nunique() > 1, (
        "All scores are identical — model is not differentiating candidates"
    )

    # ── Reasoning non-empty ───────────────────────────────────────────────────
    empty_reasoning = df[
        df["reasoning"].isna() | (df["reasoning"].str.strip() == "")
    ]
    assert len(empty_reasoning) == 0, (
        f"Empty reasoning at ranks: {empty_reasoning['rank'].tolist()}"
    )

    log.info("✓ All format checks passed — output is valid")