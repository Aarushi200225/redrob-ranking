"""
features/f6_market_validation.py
──────────────────────────────────
F6 — Market Validation scorer.

Weak signal — carries only 0.03 weight in composite.
Noisy platform metrics that loosely correlate with candidate quality.

Signals:
  - saved_by_recruiters_30d (log-normalised)
  - search_appearance_30d (log-normalised)
  - endorsements on JD-relevant skills only
"""

import math
from src.utils.logger import get_logger

log = get_logger(__name__)


def _log_normalize(value: float, scale: float = 50.0) -> float:
    """
    Log-normalise a count metric to [0, 1].
    Diminishing returns — prevents outliers from dominating.
    """
    if value <= 0:
        return 0.0
    return float(min(1.0, math.log1p(value) / math.log1p(scale)))


def _relevant_endorsements(candidate: dict, jd_object: dict) -> float:
    """
    Score endorsements received on JD-relevant skills only.
    Total endorsements count is ignored — easily gamed.
    """
    hard_reqs = {r.lower() for r in jd_object.get("hard_requirements", [])}
    skills    = candidate.get("skills", [])

    relevant_endorsements = sum(
        s.get("endorsements", 0)
        for s in skills
        if s.get("name", "").lower() in hard_reqs
    )

    return _log_normalize(relevant_endorsements, scale=20.0)


def score_f6(candidate: dict, jd_object: dict) -> float:
    """
    Compute F6 Market Validation score.

    Parameters
    ----------
    candidate  : Raw candidate dict.
    jd_object  : Parsed JD object.

    Returns
    -------
    float in [0, 1].
    """
    signals = candidate.get("redrob_signals", {})

    saved       = _log_normalize(
        signals.get("saved_by_recruiters_30d", 0), scale=20.0
    )
    appearances = _log_normalize(
        signals.get("search_appearance_30d", 0), scale=100.0
    )
    endorsements = _relevant_endorsements(candidate, jd_object)

    score = (
        0.40 * saved
        + 0.35 * appearances
        + 0.25 * endorsements
    )

    return float(max(0.0, min(1.0, score)))