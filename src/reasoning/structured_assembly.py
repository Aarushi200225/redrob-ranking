"""
reasoning/structured_assembly.py
──────────────────────────────────
Dynamic structured reasoning assembly for ranks 41-100.

NOT template-driven — each candidate's reasoning leads with
their actual strongest scoring signal. No two candidates
produce identical strings unless their profiles are identical.

Also serves as the Stage 5 fallback when Qwen LLM fails —
used per-candidate so one failure never cascades to others.
"""

from src.utils.logger import get_logger

log = get_logger(__name__)

SIGNAL_LABELS = {
    "cross_encoder": "strong JD relevance match",
    "vibe":          "strong culture-fit signals",
    "experience":    "strong experience quality",
    "rrf":           "strong semantic match",
}


def _top_skills_str(candidate: dict, n: int = 3) -> str:
    """Return top N skill names as comma-separated string."""
    proficiency_order = {
        "expert": 4, "advanced": 3,
        "intermediate": 2, "beginner": 1,
    }
    skills = sorted(
        candidate.get("skills", []),
        key=lambda s: proficiency_order.get(
            s.get("proficiency", "beginner"), 0
        ),
        reverse=True,
    )[:n]
    return ", ".join(s.get("name", "") for s in skills if s.get("name"))


def _availability_snippet(candidate: dict) -> str:
    """Build a short availability note from signals."""
    signals = candidate.get("redrob_signals", {})
    parts   = []

    notice = int(signals.get("notice_period_days", 60))
    if notice <= 30:
        parts.append("available immediately")
    elif notice <= 60:
        parts.append(f"{notice}d notice")

    if signals.get("open_to_work_flag", False):
        parts.append("actively looking")

    return "; ".join(parts)


def build_structured_reasoning(
    candidate: dict,
    jd_object: dict,
) -> str:
    """
    Build a dynamic, signal-driven reasoning string.

    Leads with the candidate's strongest scoring signal.
    Appends availability note if strong.

    Parameters
    ----------
    candidate  : Candidate dict with _score_breakdown attached.
    jd_object  : Parsed JD object.

    Returns
    -------
    str — recruiter-facing reasoning string.
    """
    profile         = candidate.get("profile", {})
    score_breakdown = candidate.get("_score_breakdown", {})

    title   = profile.get("current_title", "Candidate")
    company = profile.get("current_company", "")
    yoe     = float(profile.get("years_of_experience", 0))

    # Determine strongest signal
    top_signal = max(score_breakdown, key=score_breakdown.get) \
                 if score_breakdown else "experience"

    parts = []

    # Lead sentence — varies by top signal
    if top_signal in ("experience", "cross_encoder", "rrf"):
        top_skills = _top_skills_str(candidate, 3)
        parts.append(
            f"{title} at {company} with {yoe:.1f}y experience"
            + (f"; top skills: {top_skills}" if top_skills else "")
        )

    elif top_signal == "vibe":
        signals = candidate.get("redrob_signals", {})
        github  = signals.get("github_activity_score", -1)

        career = candidate.get("career_history", [])
        startup_sizes = {"1-10", "11-50", "51-200"}
        startup_roles = sum(
            1 for r in career
            if r.get("company_size", "") in startup_sizes
        )
        startup_note = f"; {startup_roles} startup roles" if startup_roles >= 2 else ""
        github_note  = f"; GitHub score {github:.0f}" if github > 0 else ""

        parts.append(
            f"Strong culture-fit signals — "
            f"{title} at {company}{startup_note}{github_note}"
        )

    else:
        top_skills = _top_skills_str(candidate, 3)
        parts.append(
            f"{title} at {company}, {yoe:.1f}y experience"
            + (f"; skills: {top_skills}" if top_skills else "")
        )

    # Availability note
    avail = _availability_snippet(candidate)
    if avail:
        parts.append(avail)

    return "; ".join(parts) + "."