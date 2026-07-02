"""
utils/validator.py
------------------
Output format validation.
In SANDBOX_MODE=1, accepts fewer than 100 rows (for small input demos).
In normal mode, enforces all submission rules strictly.
"""

import re
import os
import pandas as pd
from src.utils.logger import get_logger

log = get_logger(__name__)

CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")


def validate_output(df: pd.DataFrame, valid_ids: set) -> None:
    """
    Validate ranked output DataFrame against submission rules.
    In SANDBOX_MODE, relaxes row count and rank range checks.
    """
    log.info("Running output format validation ...")
    errors = []

    sandbox_mode = os.environ.get("SANDBOX_MODE", "0") == "1"
    n_expected   = len(df) if sandbox_mode else 100

    required_cols = ["candidate_id", "rank", "score", "reasoning"]
    if list(df.columns) != required_cols:
        errors.append(
            f"Columns must be {required_cols} in order. Got: {list(df.columns)}"
        )

    if not sandbox_mode and len(df) != 100:
        errors.append(f"Must have exactly 100 rows. Got: {len(df)}")

    ranks = df["rank"].tolist()
    expected_ranks = list(range(1, n_expected + 1))
    if sorted(ranks) != expected_ranks:
        errors.append(
            f"Ranks must be exactly 1-{n_expected}, no gaps or duplicates. "
            f"Got: {sorted(ranks)[:5]}..."
        )

    seen_ids = set()
    for i, cid in enumerate(df["candidate_id"]):
        if not CANDIDATE_ID_PATTERN.match(str(cid)):
            errors.append(f"Row {i+2}: bad candidate_id {cid!r}")
        if cid in seen_ids:
            errors.append(f"Row {i+2}: duplicate candidate_id {cid!r}")
        seen_ids.add(cid)

    if valid_ids:
        invalid = seen_ids - valid_ids
        if invalid:
            errors.append(
                f"{len(invalid)} candidate_ids not in source: "
                f"{sorted(invalid)[:3]}"
            )

    df_sorted = df.sort_values("rank").reset_index(drop=True)
    scores = df_sorted["score"].tolist()

    try:
        scores_float = [float(s) for s in scores]
    except (ValueError, TypeError):
        errors.append("All scores must be numeric floats.")
        scores_float = []

    if scores_float:
        for i in range(len(scores_float) - 1):
            if scores_float[i] < scores_float[i + 1]:
                errors.append(
                    f"Score not non-increasing: rank {i+1} "
                    f"({scores_float[i]:.6f}) < rank {i+2} ({scores_float[i+1]:.6f})"
                )

        if len(set(scores_float)) == 1:
            errors.append("All scores identical — no ranking signal.")

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
                    f"Tie-break violation ranks {r1},{r2}: "
                    f"equal scores but {c1!r} > {c2!r}"
                )

    for i, reasoning in enumerate(df["reasoning"]):
        if not reasoning or str(reasoning).strip() == "":
            errors.append(f"Row {i+2}: reasoning is empty.")

    if errors:
        msg = "\n".join(f"  - {e}" for e in errors)
        raise ValueError(f"Output validation failed ({len(errors)} issues):\n{msg}")

    log.info("\u2713 All format checks passed -- output is valid")
