"""
features/f8_honeypot_gate.py
─────────────────────────────
Honeypot consistency gate — vectorised numpy on all 100K candidates.

Detects logically impossible profiles before retrieval.
Runs first, eliminates ~80 honeypots from the pool.

Three signals checked:
  1. Timeline impossibility — sum(duration_months) vs years_of_experience
  2. Expert + zero duration — expert proficiency with 0 months used
  3. Skill inflation — 10+ expert skills with < 3 years total experience

Candidates scoring below HONEYPOT_SCORE_GATE are hard-eliminated.
"""

import numpy as np
from src.config import (
    HONEYPOT_TIMELINE_DELTA_MONTHS,
    HONEYPOT_EXPERT_SKILL_THRESHOLD,
    HONEYPOT_YOE_THRESHOLD,
    HONEYPOT_SCORE_GATE,
)
from src.utils.logger import get_logger, log_pool_transition

log = get_logger(__name__)


def compute_honeypot_scores(candidates: list[dict]) -> np.ndarray:
    """
    Compute honeypot consistency score for every candidate.

    Vectorised — no Python loops over candidates.
    Each signal contributes a flag (0 or 1).
    Final score = 1.0 - (weighted flag sum).

    Parameters
    ----------
    candidates : Full list of 100K candidate dicts.

    Returns
    -------
    np.ndarray of float scores [0-1], shape [len(candidates)].
    Score < HONEYPOT_SCORE_GATE → hard-eliminated.
    """
    n = len(candidates)

    # Pre-extract fields into arrays for vectorised ops
    yoe              = np.array([
        c.get("profile", {}).get("years_of_experience", 0)
        for c in candidates
    ], dtype=np.float32)

    total_months     = np.array([
        sum(r.get("duration_months", 0) for r in c.get("career_history", []))
        for c in candidates
    ], dtype=np.float32)

    expert_count     = np.array([
        sum(1 for s in c.get("skills", [])
            if s.get("proficiency") == "expert")
        for c in candidates
    ], dtype=np.int32)

    expert_zero_dur  = np.array([
        any(
            s.get("proficiency") == "expert"
            and s.get("duration_months", 0) == 0
            for s in c.get("skills", [])
        )
        for c in candidates
    ], dtype=np.float32)

    # ── Signal 1: Timeline impossibility ─────────────────────────────────────
    expected_months  = yoe * 12
    timeline_delta   = np.abs(total_months - expected_months)
    timeline_flag    = (timeline_delta > HONEYPOT_TIMELINE_DELTA_MONTHS).astype(np.float32)

    # ── Signal 2: Expert skill with zero duration ─────────────────────────────
    expert_zero_flag = expert_zero_dur  # already 0/1

    # ── Signal 3: Skill inflation ─────────────────────────────────────────────
    inflation_flag   = (
        (expert_count >= HONEYPOT_EXPERT_SKILL_THRESHOLD)
        & (yoe < HONEYPOT_YOE_THRESHOLD)
    ).astype(np.float32)

    # Weighted flag sum — timeline weighted highest
    flag_sum = (
        0.50 * timeline_flag
        + 0.30 * expert_zero_flag
        + 0.20 * inflation_flag
    )

    scores = 1.0 - flag_sum
    scores = np.clip(scores, 0.0, 1.0)

    flagged = int((scores < HONEYPOT_SCORE_GATE).sum())
    log.info(f"Honeypot gate: {flagged} candidates flagged for elimination")

    return scores


def apply_honeypot_gate(
    candidates: list[dict],
    scores: np.ndarray,
) -> list[dict]:
    """
    Hard-eliminate candidates below the honeypot score gate.

    Parameters
    ----------
    candidates : Full candidate list.
    scores     : Honeypot scores from compute_honeypot_scores.

    Returns
    -------
    list[dict] — clean candidates above the gate threshold.
    """
    mask  = scores >= HONEYPOT_SCORE_GATE
    clean = [c for c, keep in zip(candidates, mask) if keep]

    log_pool_transition(
        log, "F8 honeypot gate",
        len(candidates), len(clean),
        note=f"{len(candidates) - len(clean)} eliminated"
    )

    return clean