"""
features/f8_honeypot_gate.py
─────────────────────────────
Honeypot consistency gate — vectorised numpy + hard logical rules.

Detects logically impossible profiles before retrieval.
Runs on all 100K candidates before any embedding or scoring.

Design principle:
  Honeypots are LOGICALLY IMPOSSIBLE, not just statistically unlikely.
  Good candidates with unusual careers should not be flagged.
  Only profiles that violate physical/temporal impossibility are eliminated.

Hard gate signals (any single one eliminates the candidate):
  H1 — Expert skill claimed with zero months of usage
  H2 — Multiple expert skills with zero months (definitive stuffer)
  H3 — Total career duration impossibly exceeds claimed YoE by >3 years
  H4 — Skill inflation: 8+ expert skills AND YoE < 3 years

Soft signals (combined score — must cross threshold):
  S1 — Moderate timeline inconsistency
  S2 — High expert skill count relative to YoE
  S3 — Endorsement inflation (many endorsements, no duration)

Threshold: HONEYPOT_SCORE_GATE — candidates below are hard-eliminated.
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


# ── Hard gate thresholds ──────────────────────────────────────────────────────

# Any candidate with ANY expert skill having duration_months == 0
HARD_EXPERT_ZERO_DURATION = True

# Minimum number of expert+zero-duration skills to trigger hard gate
HARD_MULTI_EXPERT_ZERO_COUNT = 1    # Even ONE expert skill with 0 months = flag

# Maximum ratio of expert skills to total YoE before hard flag
# e.g. 10 expert skills with 2 YoE = 5.0 ratio → impossible
HARD_EXPERT_PER_YEAR_RATIO = 3.0   # > 3 expert skills per year = suspicious

# Hard YoE inflation: if sum(role durations) > stated YoE + this many months
HARD_TIMELINE_DELTA_MONTHS = 18    # Tighter than config default of 24

# Minimum endorsements per relevant skill for inflation flag
HARD_ENDORSEMENT_RATIO = 50        # >50 endorsements per skill with 0 duration

# Expert skill count hard gate — regardless of YoE
HARD_ABSOLUTE_EXPERT_COUNT = 12    # >12 expert skills total = stuffer flag


def _has_expert_zero_duration(candidate: dict) -> bool:
    """
    Hard flag: any skill claimed as 'expert' with 0 months usage.
    This is the primary honeypot pattern — claiming expertise never used.
    """
    return any(
        s.get("proficiency") == "expert"
        and int(s.get("duration_months", 0)) == 0
        for s in candidate.get("skills", [])
    )


def _count_expert_zero_skills(candidate: dict) -> int:
    """Count skills with expert proficiency AND zero duration."""
    return sum(
        1 for s in candidate.get("skills", [])
        if s.get("proficiency") == "expert"
        and int(s.get("duration_months", 0)) == 0
    )


def _expert_per_year_ratio(candidate: dict) -> float:
    """
    Ratio of expert skill count to years of experience.
    A genuine expert typically has 1-2 expert skills per year of focused work.
    >3 expert skills per year is implausible.
    """
    yoe = float(
        candidate.get("profile", {}).get("years_of_experience", 1)
    )
    yoe = max(yoe, 0.5)  # Avoid division by zero

    expert_count = sum(
        1 for s in candidate.get("skills", [])
        if s.get("proficiency") == "expert"
    )
    return expert_count / yoe


def _timeline_delta_months(candidate: dict) -> float:
    """
     Check for timeline impossibilities.

    Direction A — role_months >> YoE: worked more than claimed.
    Impossible regardless of how many roles listed.
    Only flag if delta > 36 months to allow for rounding.

    Direction B — role_months << YoE: only a honeypot signal when
    the candidate has ZERO roles listed entirely, or when the delta
    is extreme AND all listed roles have suspiciously low total months.
    NOT flagged for senior candidates who simply didn't list early roles.
    """
    yoe = float(
        candidate.get("profile", {}).get("years_of_experience", 0)
    )
    career = candidate.get("career_history", [])
    total_months = sum(
        int(r.get("duration_months", 0))
        for r in career
    )
    expected_months = yoe * 12

    # Direction A: worked MORE than claimed YoE — impossible
    if total_months > expected_months + 36:
        return total_months - expected_months
    
    # Direction B: only flag if ZERO roles listed with senior claim
    # A candidate with even 1 real role simply didn't list everything
    if len(career) == 0 and yoe > 3:
        return expected_months  # No roles at all = suspicious

    return 0.0  # Not flagged


def _endorsement_inflation(candidate: dict) -> bool:
    """
    Flag candidates with high endorsements on skills with zero usage.
    Legitimate endorsements come from colleagues who observed the skill.
    Many endorsements on a never-used skill = fabricated profile.
    """
    return any(
        s.get("endorsements", 0) > HARD_ENDORSEMENT_RATIO
        and int(s.get("duration_months", 0)) == 0
        for s in candidate.get("skills", [])
    )


def _absolute_expert_count(candidate: dict) -> int:
    """Total number of expert-level skills."""
    return sum(
        1 for s in candidate.get("skills", [])
        if s.get("proficiency") == "expert"
    )


def compute_honeypot_scores(candidates: list[dict]) -> np.ndarray:
    """
    Compute honeypot consistency score for every candidate.

    Uses both vectorised operations for speed and per-candidate
    logic for precision on subtle patterns.

    Parameters
    ----------
    candidates : Full list of 100K candidate dicts.

    Returns
    -------
    np.ndarray of float scores [0-1], shape [len(candidates)].
    Score < HONEYPOT_SCORE_GATE → hard-eliminated.
    """
    n      = len(candidates)
    scores = np.ones(n, dtype=np.float32)

    # Track flag counts for logging
    h1_count = 0  # Expert + zero duration
    h2_count = 0  # Multiple expert + zero duration
    h3_count = 0  # Timeline impossibility
    h4_count = 0  # Expert/YoE ratio
    h5_count = 0  # Absolute expert count
    h6_count = 0  # Endorsement inflation

    for i, candidate in enumerate(candidates):

        flag_score = 0.0

        # ── H1: Expert skill with zero duration (HARD) ───────────────────────
        if _has_expert_zero_duration(candidate):
            # Each expert+zero skill adds a significant penalty
            zero_count = _count_expert_zero_skills(candidate)
            if zero_count >= 1:
                flag_score += 0.50   # Single expert+zero = strong flag
                h1_count += 1
            if zero_count >= 2:
                flag_score += 0.30   # Two or more = definitive honeypot
                h2_count += 1

        # ── H2: Expert-per-year ratio (HARD) ─────────────────────────────────
        ep_ratio = _expert_per_year_ratio(candidate)
        if ep_ratio > HARD_EXPERT_PER_YEAR_RATIO:
            flag_score += 0.40
            h4_count += 1

        # ── H3: Timeline impossibility ────────────────────────────────────────
        delta = _timeline_delta_months(candidate)
        if delta > HARD_TIMELINE_DELTA_MONTHS:
            # Scale penalty by how impossible the delta is
            penalty = min(0.50, (delta - HARD_TIMELINE_DELTA_MONTHS) / 24)
            flag_score += penalty
            h3_count += 1

        # ── H4: Absolute expert count (HARD) ─────────────────────────────────
        abs_expert = _absolute_expert_count(candidate)
        if abs_expert > HARD_ABSOLUTE_EXPERT_COUNT:
            flag_score += 0.30
            h5_count += 1

        # ── H5: Endorsement inflation ─────────────────────────────────────────
        if _endorsement_inflation(candidate):
            flag_score += 0.40
            h6_count += 1

        # Final score — clamp to [0, 1]
        scores[i] = max(0.0, min(1.0, 1.0 - flag_score))

    # Log flag distribution
    flagged = int((scores < HONEYPOT_SCORE_GATE).sum())
    log.info(
        f"Honeypot gate signals: "
        f"H1(expert+zero)={h1_count}, "
        f"H2(multi-zero)={h2_count}, "
        f"H3(timeline)={h3_count}, "
        f"H4(expert/yr)={h4_count}, "
        f"H5(abs-expert)={h5_count}, "
        f"H6(endorsement)={h6_count}"
    )
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