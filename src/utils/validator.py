"""
utils/validator.py
──────────────────
Output format validation — matches validate_submission.py exactly.

Hard assertions — any failure raises ValueError immediately.
Run before writing CSV to catch issues early.
"""

import re
import pandas as pd
from src.utils.logger import get_logger

log = get_logger(__name__)

CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")


def validate_output(df: pd.DataFrame, valid_ids: set) -> None:
    """
    Validate ranked output DataFrame against all submission rules.

    Parameters
    ----------
    df        : DataFrame with columns [candidate_id, rank, score, reasoning]
    valid_ids : Set of valid candidate_ids from source JSONL

    Raises
    ------
    ValueError if any check fails.
    """
    log.info("Running output format validation ...")
    errors = []

    # ── Column check ──────────────────────────────────────────────────────────
    required_cols = ["candidate_id", "rank", "score", "reasoning"]
    if list(df.columns) != required_cols:
        errors.append(
            f"Columns must be exactly {required_cols} in order. "
            f"Got: {list(df.columns)}"
        )

    # ── Row count ─────────────────────────────────────────────────────────────
    if len(df) != 100:
        errors.append(f"Must have exactly 100 rows. Got: {len(df)}")

    # ── Rank checks ───────────────────────────────────────────────────────────
    ranks = df["rank"].tolist()
    if sorted(ranks) != list(range(1, 101)):
        errors.append("Ranks must be exactly 1-100 with no gaps or duplicates.")

    # ── Candidate ID checks ───────────────────────────────────────────────────
    seen_ids = set()
    for i, cid in enumerate(df["candidate_id"]):
        if not CANDIDATE_ID_PATTERN.match(str(cid)):
            errors.append(
                f"Row {i+2}: candidate_id '{cid}' must match CAND_XXXXXXX"
            )
        if cid in seen_ids:
            errors.append(f"Row {i+2}: duplicate candidate_id '{cid}'")
        seen_ids.add(cid)

    if valid_ids:
        invalid = seen_ids - valid_ids
        if invalid:
            errors.append(
                f"{len(invalid)} candidate_ids not found in source: "
                f"{sorted(invalid)[:5]}"
            )

    # ── Score checks ──────────────────────────────────────────────────────────
    df_sorted = df.sort_values("rank").reset_index(drop=True)
    scores = df_sorted["score"].tolist()

    # Scores must be numeric
    try:
        scores_float = [float(s) for s in scores]
    except (ValueError, TypeError):
        errors.append("All scores must be numeric floats.")
        scores_float = []

    if scores_float:
        # Monotonically non-increasing
        for i in range(len(scores_float) - 1):
            if scores_float[i] < scores_float[i + 1]:
                errors.append(
                    f"Score not non-increasing: "
                    f"rank {i+1} ({scores_float[i]:.6f}) < "
                    f"rank {i+2} ({scores_float[i+1]:.6f})"
                )

        # Not all identical
        if len(set(scores_float)) == 1:
            errors.append("All scores are identical — ranking has no signal.")

        # ── Tie-breaking check (matches validate_submission.py exactly) ───────
        by_rank = list(zip(
            df_sorted["rank"].tolist(),
            scores_float,
            df_sorted["candidate_id"].tolist()
        ))
        for i in range(len(by_rank) - 1):
            r1, s1, c1 = by_rank[i]
            r2, s2, c2 = by_rank[i + 1]
            if s1 == s2 and c1 > c2:
                errors.append(
                    f"Tie-breaking violation: equal scores at ranks "
                    f"{r1} and {r2} — candidate_id must be ascending "
                    f"({c1!r} > {c2!r})"
                )

    # ── Reasoning checks ──────────────────────────────────────────────────────
    for i, reasoning in enumerate(df["reasoning"]):
        if not reasoning or str(reasoning).strip() == "":
            errors.append(f"Row {i+2}: reasoning is empty.")

    # ── Result ────────────────────────────────────────────────────────────────
    if errors:
        error_msg = "\n".join(f"  - {e}" for e in errors)
        raise ValueError(
            f"Output validation failed ({len(errors)} issues):\n{error_msg}"
        )

    log.info("✓ All format checks passed — output is valid")