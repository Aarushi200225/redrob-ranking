"""
features/f2_experience_quality.py
──────────────────────────────────
F2 — Experience Quality scorer.

Signals:
  - Years of experience band fit
  - AI/ML-specific years in relevant roles
  - Production deployment signal in career descriptions
  - Pre-2022 ML work (not LangChain-era only)
  - Education tier bonus
"""

from src.utils.logger import get_logger

log = get_logger(__name__)

PRODUCTION_KEYWORDS = {
    "deployed", "production", "prod", "shipped", "launched",
    "serving", "real traffic", "live", "at scale", "users",
    "customers", "million", "thousand requests", "latency",
    "throughput", "real-time", "online system",
}

AI_ML_ROLE_KEYWORDS = {
    "machine learning", "ml", "ai", "deep learning", "nlp",
    "computer vision", "data science", "research", "llm",
    "neural", "model", "embedding", "ranking", "retrieval",
    "recommendation", "search", "generative",
}

PRE_2022_CUTOFF = "2022-01-01"


def _score_experience_band(yoe: float) -> float:
    """Score years of experience against JD band (5-9 years)."""
    if 5 <= yoe <= 9:
        return 1.00
    elif 4 <= yoe <= 10:
        return 0.85
    elif 3 <= yoe < 4:
        return 0.55
    elif 10 < yoe <= 12:
        return 0.65
    elif yoe > 12:
        return 0.45
    else:
        return 0.25


def _compute_ai_ml_months(career: list[dict]) -> float:
    """Count months spent in AI/ML-relevant roles."""
    ai_months = 0
    for role in career:
        title = role.get("title", "").lower()
        desc  = role.get("description", "").lower()
        if any(kw in title or kw in desc for kw in AI_ML_ROLE_KEYWORDS):
            ai_months += role.get("duration_months", 0)
    return ai_months


def _has_production_signal(career: list[dict]) -> bool:
    """Return True if any role description mentions production deployment."""
    for role in career:
        desc = role.get("description", "").lower()
        if any(kw in desc for kw in PRODUCTION_KEYWORDS):
            return True
    return False


def _has_pre_2022_ml(career: list[dict]) -> bool:
    """Return True if candidate has ML/AI work before 2022."""
    for role in career:
        end_date = role.get("end_date") or ""
        start_date = role.get("start_date", "")
        title = role.get("title", "").lower()
        desc  = role.get("description", "").lower()

        is_ml = any(kw in title or kw in desc for kw in AI_ML_ROLE_KEYWORDS)
        is_pre_2022 = start_date < PRE_2022_CUTOFF

        if is_ml and is_pre_2022:
            return True
    return False


def _score_education(education: list[dict]) -> float:
    """Score education tier."""
    for edu in education:
        tier = edu.get("tier", "unknown")
        if tier == "tier_1":
            return 1.0
        elif tier == "tier_2":
            return 0.75
        elif tier == "tier_3":
            return 0.55
    return 0.40


def score_f2(candidate: dict, jd_object: dict) -> float:
    """
    Compute F2 Experience Quality score.

    Parameters
    ----------
    candidate  : Raw candidate dict.
    jd_object  : Parsed JD object from Stage 1.

    Returns
    -------
    float in [0, 1].
    """
    profile   = candidate.get("profile", {})
    career    = candidate.get("career_history", [])
    education = candidate.get("education", [])

    yoe = float(profile.get("years_of_experience", 0))

    # ── Band score ────────────────────────────────────────────────────────────
    band_score = _score_experience_band(yoe)

    # ── AI/ML years ratio ─────────────────────────────────────────────────────
    ai_months    = _compute_ai_ml_months(career)
    ai_ratio     = min(1.0, ai_months / max(yoe * 12, 1))
    ai_score     = ai_ratio

    # ── Production signal ─────────────────────────────────────────────────────
    production_score = 1.0 if _has_production_signal(career) else 0.3

    # ── Pre-2022 ML ───────────────────────────────────────────────────────────
    pre_2022_bonus = 0.15 if _has_pre_2022_ml(career) else 0.0

    # ── Education tier ────────────────────────────────────────────────────────
    edu_score = _score_education(education)

    # ── Composite F2 ─────────────────────────────────────────────────────────
    base = (
        0.35 * band_score
        + 0.25 * ai_score
        + 0.25 * production_score
        + 0.15 * edu_score
    )

    score = min(1.0, base + pre_2022_bonus)
    return float(score)