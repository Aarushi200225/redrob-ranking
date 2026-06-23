"""
features/f3_skills_match.py
────────────────────────────
F3 — Skills Match scorer.

Signals:
  - Hard requirement coverage (expanded taxonomy)
  - Skill depth (advanced/expert proficiency on relevant skills)
  - Skill assessment scores (if populated)
  - Certification bonus
  - Anti-stuffer penalty
"""

from src.config import (
    ANTI_STUFFER_SKILL_COUNT,
    ANTI_STUFFER_AVG_DURATION_MONTHS,
)
from src.utils.logger import get_logger
from src.utils.data_loader import load_json_artifact
from src.config import SKILL_TAXONOMY_PATH

log = get_logger(__name__)

PROFICIENCY_WEIGHTS = {
    "expert": 1.0,
    "advanced": 0.75,
    "intermediate": 0.50,
    "beginner": 0.25,
}

# Load taxonomy once at module level
try:
    _TAXONOMY = load_json_artifact(SKILL_TAXONOMY_PATH)
    _ALL_TAXONOMY_TERMS = {
        term.lower()
        for terms in _TAXONOMY.values()
        for term in terms
    }
except Exception:
    _TAXONOMY = {}
    _ALL_TAXONOMY_TERMS = set()


def _get_candidate_skill_names(candidate: dict) -> set[str]:
    """Return lowercased set of candidate skill names."""
    return {
        s.get("name", "").lower()
        for s in candidate.get("skills", [])
    }


def _compute_hard_req_coverage(
    candidate_skills: set[str],
    jd_object: dict,
) -> float:
    """
    Fraction of hard requirements covered by candidate skills.
    Uses expanded taxonomy for matching.
    """
    hard_reqs = jd_object.get("hard_requirements", [])
    if not hard_reqs:
        return 0.5

    covered = 0
    for req in hard_reqs:
        req_lower = req.lower()
        # Direct match
        if req_lower in candidate_skills:
            covered += 1
            continue
        # Taxonomy expansion match
        for category, terms in _TAXONOMY.items():
            if req_lower in category or any(
                req_lower in t.lower() for t in terms
            ):
                if any(t.lower() in candidate_skills for t in terms):
                    covered += 1
                    break

    return covered / len(hard_reqs)


def _compute_skill_depth(candidate: dict, jd_object: dict) -> float:
    """Score depth of proficiency on JD-relevant skills."""
    hard_reqs = {r.lower() for r in jd_object.get("hard_requirements", [])}
    skills    = candidate.get("skills", [])

    relevant_scores = []
    for skill in skills:
        name  = skill.get("name", "").lower()
        prof  = skill.get("proficiency", "beginner")
        weight = PROFICIENCY_WEIGHTS.get(prof, 0.25)

        # Check if skill is relevant to JD
        is_relevant = (
            name in hard_reqs
            or any(name in t.lower() for t in _ALL_TAXONOMY_TERMS)
        )
        if is_relevant:
            relevant_scores.append(weight)

    if not relevant_scores:
        return 0.2
    return min(1.0, sum(relevant_scores) / max(len(relevant_scores), 1))


def _compute_assessment_bonus(
    candidate: dict,
    jd_object: dict,
) -> float:
    """Score platform assessment results on JD-relevant skills."""
    assessments = candidate.get(
        "redrob_signals", {}
    ).get("skill_assessment_scores", {})

    if not assessments:
        return 0.5  # Neutral — not penalised for missing assessments

    hard_reqs = {r.lower() for r in jd_object.get("hard_requirements", [])}
    relevant_scores = []

    for skill_name, score in assessments.items():
        if skill_name.lower() in hard_reqs:
            relevant_scores.append(score / 100.0)

    if not relevant_scores:
        return 0.5
    return sum(relevant_scores) / len(relevant_scores)


def _anti_stuffer_penalty(candidate: dict) -> float:
    """
    Detect keyword stuffers — many skills with very low usage duration.
    Returns a penalty multiplier [0.5, 1.0].
    """
    skills = candidate.get("skills", [])
    if len(skills) <= ANTI_STUFFER_SKILL_COUNT:
        return 1.0  # No penalty

    durations = [s.get("duration_months", 0) for s in skills]
    avg_duration = sum(durations) / len(durations) if durations else 0

    if avg_duration < ANTI_STUFFER_AVG_DURATION_MONTHS:
        return 0.6  # Significant penalty for stuffers

    return 1.0


def score_f3(candidate: dict, jd_object: dict) -> float:
    """
    Compute F3 Skills Match score.

    Parameters
    ----------
    candidate  : Raw candidate dict.
    jd_object  : Parsed JD object from Stage 1.

    Returns
    -------
    float in [0, 1].
    """
    candidate_skills = _get_candidate_skill_names(candidate)

    coverage    = _compute_hard_req_coverage(candidate_skills, jd_object)
    depth       = _compute_skill_depth(candidate, jd_object)
    assessment  = _compute_assessment_bonus(candidate, jd_object)
    penalty     = _anti_stuffer_penalty(candidate)

    # Certification bonus
    certs = candidate.get("certifications", [])
    cert_bonus = min(0.1, len(certs) * 0.03)

    base = (
        0.45 * coverage
        + 0.35 * depth
        + 0.20 * assessment
    )

    score = min(1.0, (base + cert_bonus) * penalty)
    return float(score)