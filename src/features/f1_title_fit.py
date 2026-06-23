"""
features/f1_title_fit.py
────────────────────────
F1 — Role/Title Fit scorer.

Signals:
  - Title semantic similarity to target role
  - Career progression (titles trending toward seniority)
  - Consulting-only hard gate
  - Product company ratio across career
"""

from src.config import CONSULTING_FIRMS
from src.utils.logger import get_logger

log = get_logger(__name__)

# Seniority level mapping for progression scoring
SENIORITY_LEVELS = {
    "intern": 0, "trainee": 0, "junior": 1, "associate": 1,
    "engineer": 2, "developer": 2, "analyst": 2,
    "senior": 3, "lead": 3, "staff": 3,
    "principal": 4, "architect": 4, "manager": 4,
    "director": 5, "vp": 5, "head": 5, "chief": 6,
}

# AI/ML title keywords — positive signal for this JD
AI_ML_TITLE_KEYWORDS = {
    "machine learning", "ml engineer", "ai engineer",
    "data scientist", "nlp", "research engineer",
    "applied scientist", "cv engineer", "deep learning",
    "llm", "generative ai", "search engineer",
    "ranking engineer", "retrieval", "recommendations",
}


def _is_consulting_company(company: str, industry: str) -> bool:
    """Return True if company appears to be a consulting/IT services firm."""
    company_lower = company.lower().strip()
    industry_lower = industry.lower().strip()

    # Direct name match
    for firm in CONSULTING_FIRMS:
        if firm in company_lower:
            return True

    # Industry signal
    consulting_industries = {
        "it services", "consulting", "outsourcing",
        "information technology and services",
        "it consulting", "staffing"
    }
    if any(ind in industry_lower for ind in consulting_industries):
        return True

    return False


def _get_seniority_score(title: str) -> int:
    """Map a job title to a seniority level integer."""
    title_lower = title.lower()
    best = 0
    for keyword, level in SENIORITY_LEVELS.items():
        if keyword in title_lower:
            best = max(best, level)
    return best


def score_f1(candidate: dict, jd_object: dict) -> float:
    """
    Compute F1 Role/Title Fit score.

    Parameters
    ----------
    candidate  : Raw candidate dict.
    jd_object  : Parsed JD object from Stage 1.

    Returns
    -------
    float in [0, 1]. Returns -1.0 for hard-eliminated candidates
    (consulting-only career).
    """
    profile = candidate.get("profile", {})
    career  = candidate.get("career_history", [])

    if not career:
        return 0.2

    # ── Consulting-only hard gate ─────────────────────────────────────────────
    product_roles    = []
    consulting_roles = []

    for role in career:
        company  = role.get("company", "")
        industry = role.get("industry", "")
        if _is_consulting_company(company, industry):
            consulting_roles.append(role)
        else:
            product_roles.append(role)

    # Hard eliminate ONLY if zero product company roles
    if len(product_roles) == 0 and len(consulting_roles) > 0:
        log.debug(
            f"Consulting-only gate triggered: "
            f"{candidate.get('candidate_id')}"
        )
        return -1.0  # Sentinel for hard elimination

    # ── Product company ratio ─────────────────────────────────────────────────
    total_months   = sum(r.get("duration_months", 0) for r in career)
    product_months = sum(r.get("duration_months", 0) for r in product_roles)
    product_ratio  = product_months / total_months if total_months > 0 else 0.0

    # ── Current title relevance ───────────────────────────────────────────────
    current_title = profile.get("current_title", "").lower()
    title_is_ai   = any(kw in current_title for kw in AI_ML_TITLE_KEYWORDS)
    title_score   = 0.9 if title_is_ai else 0.5

    # ── Career progression ────────────────────────────────────────────────────
    if len(career) >= 2:
        sorted_career  = sorted(
            career,
            key=lambda r: r.get("start_date", ""),
        )
        early_seniority = _get_seniority_score(
            sorted_career[0].get("title", "")
        )
        recent_seniority = _get_seniority_score(
            sorted_career[-1].get("title", "")
        )
        progression_score = min(
            1.0,
            0.5 + 0.1 * (recent_seniority - early_seniority)
        )
    else:
        progression_score = 0.5

    # ── Composite F1 ─────────────────────────────────────────────────────────
    score = (
        0.40 * title_score
        + 0.35 * product_ratio
        + 0.25 * progression_score
    )

    return float(np.clip(score, 0.0, 1.0)) if False else float(
        max(0.0, min(1.0, score))
    )