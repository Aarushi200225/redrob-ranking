"""
features/f5_availability.py
────────────────────────────
F5 — Availability and Reachability scorer.

Signals:
  - Recency: days since last_active_date (exponential decay)
  - open_to_work_flag
  - Recruiter response composite
  - Notice period fit
  - Interview completion rate
  - Offer acceptance rate
"""

import math
from datetime import date, datetime
from src.config import (
    NOTICE_PERIOD_SCORES,
    RECENCY_HARD_CUTOFF_DAYS,
    RECENCY_HARD_MULTIPLIER,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

REFERENCE_DATE = date.today()


def _recency_score(last_active_date: str) -> tuple[float, float]:
    """
    Compute recency score and hard multiplier.

    Returns
    -------
    (score, multiplier) — score in [0,1], multiplier is
    RECENCY_HARD_MULTIPLIER if >180 days inactive, else 1.0.
    """
    if not last_active_date:
        return 0.4, 1.0

    try:
        last_active = datetime.strptime(
            last_active_date, "%Y-%m-%d"
        ).date()
        days_inactive = (REFERENCE_DATE - last_active).days
    except ValueError:
        return 0.4, 1.0

    # Exponential decay — half-life ~60 days
    score      = math.exp(-days_inactive / 60.0)
    multiplier = RECENCY_HARD_MULTIPLIER \
                 if days_inactive > RECENCY_HARD_CUTOFF_DAYS else 1.0

    return float(min(1.0, score)), multiplier


def _notice_period_score(days: int) -> float:
    """Map notice period days to score using config thresholds."""
    if days <= 30:
        return NOTICE_PERIOD_SCORES[30]
    elif days <= 60:
        return NOTICE_PERIOD_SCORES[60]
    elif days <= 90:
        return NOTICE_PERIOD_SCORES[90]
    else:
        return NOTICE_PERIOD_SCORES[180]


def _response_composite(
    response_rate: float,
    avg_response_hours: float,
) -> float:
    """
    Combine recruiter response rate and response time into one score.
    """
    rate_score = float(max(0.0, min(1.0, response_rate)))

    # Response time: lower is better, inverse normalised
    # 0h=1.0, 24h=0.75, 72h=0.5, 168h=0.25
    if avg_response_hours <= 0:
        time_score = 1.0
    elif avg_response_hours <= 24:
        time_score = 0.75 + 0.25 * (1 - avg_response_hours / 24)
    elif avg_response_hours <= 72:
        time_score = 0.5 + 0.25 * (1 - (avg_response_hours - 24) / 48)
    elif avg_response_hours <= 168:
        time_score = 0.25 + 0.25 * (1 - (avg_response_hours - 72) / 96)
    else:
        time_score = 0.1

    return float(0.60 * rate_score + 0.40 * time_score)


def score_f5(candidate: dict, jd_object: dict) -> tuple[float, float]:
    """
    Compute F5 Availability score and recency multiplier.

    Parameters
    ----------
    candidate  : Raw candidate dict.
    jd_object  : Parsed JD object (unused here, for interface consistency).

    Returns
    -------
    tuple[float, float] — (availability_score, recency_multiplier).
    Recency multiplier applied multiplicatively in composite scoring.
    """
    signals = candidate.get("redrob_signals", {})

    # Recency
    recency, recency_mult = _recency_score(
        signals.get("last_active_date", "")
    )

    # Open to work
    open_to_work = 1.0 if signals.get("open_to_work_flag", False) else 0.4

    # Response composite
    response = _response_composite(
        signals.get("recruiter_response_rate", 0.5),
        signals.get("avg_response_time_hours", 48),
    )

    # Notice period
    notice = _notice_period_score(
        int(signals.get("notice_period_days", 60))
    )

    # Interview completion
    interview = float(
        max(0.0, min(1.0, signals.get("interview_completion_rate", 0.5)))
    )

    # Offer acceptance (-1 = no history → neutral 0.5)
    raw_offer = signals.get("offer_acceptance_rate", -1)
    offer     = 0.5 if raw_offer == -1 else float(
        max(0.0, min(1.0, raw_offer))
    )

    score = (
        0.30 * recency
        + 0.20 * open_to_work
        + 0.25 * response
        + 0.15 * notice
        + 0.07 * interview
        + 0.03 * offer
    )

    return float(max(0.0, min(1.0, score))), recency_mult