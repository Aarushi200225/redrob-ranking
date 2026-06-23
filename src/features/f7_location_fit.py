"""
features/f7_location_fit.py
────────────────────────────
F7 — Location Fit multiplier.

Applied multiplicatively in composite scoring — not as an
additive component. Dampens scores for location mismatches
without eliminating candidates entirely.

Values reflect JD language:
  "preferred" → tier_1 cities
  "welcome to apply" → tier_2 cities
  "case-by-case" → outside India
"""

from src.config import (
    LOCATION_MULTIPLIERS,
    LOCATION_TIER_1,
    LOCATION_TIER_2,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


def score_f7(candidate: dict, jd_object: dict) -> float:
    """
    Compute F7 Location Fit multiplier.

    Parameters
    ----------
    candidate  : Raw candidate dict.
    jd_object  : Parsed JD object (unused, for interface consistency).

    Returns
    -------
    float multiplier in [0.30, 1.00].
    """
    profile          = candidate.get("profile", {})
    signals          = candidate.get("redrob_signals", {})
    location         = profile.get("location", "").lower().strip()
    country          = profile.get("country", "").lower().strip()
    willing_relocate = signals.get("willing_to_relocate", False)

    # ── Tier 1 — Pune, Noida ─────────────────────────────────────────────────
    if any(city in location for city in LOCATION_TIER_1):
        return LOCATION_MULTIPLIERS["tier_1"]

    # ── Tier 2 — Hyderabad, Mumbai, Delhi NCR ────────────────────────────────
    if any(city in location for city in LOCATION_TIER_2):
        return LOCATION_MULTIPLIERS["tier_2"]

    # ── Other India ───────────────────────────────────────────────────────────
    if country == "india":
        if willing_relocate:
            return LOCATION_MULTIPLIERS["india_relocate"]
        return LOCATION_MULTIPLIERS["india_no_relocate"]

    # ── Outside India ─────────────────────────────────────────────────────────
    if willing_relocate:
        return LOCATION_MULTIPLIERS["outside_relocate"]
    return LOCATION_MULTIPLIERS["outside_no_relocate"]