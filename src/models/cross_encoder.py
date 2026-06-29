"""
models/cross_encoder.py
───────────────────────
Wrapper for ms-marco-MiniLM-L-6-v2 cross-encoder reranker.

Handles:
  - Model loading
  - Input construction per candidate (full career history, not just recent role)
  - Batch pair scoring
  - Raw logit output (normalisation handled in scoring.py)

Note on input construction:
  Five career roles are included — not just the most recent.
  The JD cares about pre-2022 ML experience and product company
  history, both of which may appear in earlier roles.

Note on output:
  Raw unbounded logits are returned. MinMax normalisation over
  the candidate batch is applied in scoring.py before composite
  score computation — never here.
"""

# from sentence_transformers import CrossEncoder
import numpy as np

from src.config import CROSS_ENCODER_MODEL_ID
from src.utils.logger import get_logger

log = get_logger(__name__)


def load_cross_encoder():
    """
    Load ms-marco-MiniLM-L-6-v2 cross-encoder.

    Returns
    -------
    CrossEncoder model ready for pair scoring.
    """
    from sentence_transformers import CrossEncoder
    log.info(f"Loading cross-encoder: {CROSS_ENCODER_MODEL_ID}")
    model = CrossEncoder(CROSS_ENCODER_MODEL_ID)
    log.info("Cross-encoder loaded")
    return model


def build_cross_encoder_input(candidate: dict) -> str:
    """
    Build structured candidate text for cross-encoder input.

    Includes up to 5 career roles (compressed) to ensure
    pre-2022 ML experience and product company history
    are visible to the cross-encoder.

    Parameters
    ----------
    candidate : Raw candidate dict.

    Returns
    -------
    str — structured candidate representation.
    """
    profile = candidate.get("profile", {})
    career  = candidate.get("career_history", [])
    skills  = candidate.get("skills", [])

    parts = []

    # Headline and summary (truncated)
    headline = profile.get("headline", "").strip()
    summary  = profile.get("summary", "")[:250].strip()
    if headline:
        parts.append(f"HEADLINE: {headline}")
    if summary:
        parts.append(f"SUMMARY: {summary}")

    # Up to 5 career roles — compressed format
    if career:
        career_parts = ["CAREER:"]
        for role in career[:5]:
            title    = role.get("title", "")
            company  = role.get("company", "")
            duration = role.get("duration_months", 0)
            industry = role.get("industry", "")
            desc     = role.get("description", "")[:150].strip()
            career_parts.append(
                f"- {title} at {company} "
                f"({duration}mo, {industry}): {desc}"
            )
        parts.append("\n".join(career_parts))

    # Top 8 skills with proficiency
    proficiency_order = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    sorted_skills = sorted(
        skills,
        key=lambda s: proficiency_order.get(s.get("proficiency", "beginner"), 0),
        reverse=True,
    )[:8]
    if sorted_skills:
        skill_str = ", ".join(
            f"{s.get('name', '')}({s.get('proficiency', '')})"
            for s in sorted_skills
        )
        parts.append(f"SKILLS: {skill_str}")

    return "\n".join(parts)


def score_pairs(
    model,
    jd_text: str,
    candidates: list[dict],
) -> np.ndarray:
    """
    Score (JD, candidate) pairs using the cross-encoder.

    Parameters
    ----------
    model      : Loaded CrossEncoder model.
    jd_text    : Job description text (key sections).
    candidates : List of candidate dicts to score.

    Returns
    -------
    np.ndarray of raw logit scores, shape [len(candidates)].
    Unbounded — normalisation applied downstream in scoring.py.
    """
    if not candidates:
        raise ValueError("candidates list is empty")

    log.info(f"Cross-encoder scoring {len(candidates):,} pairs ...")

    pairs = [
        [jd_text, build_cross_encoder_input(c)]
        for c in candidates
    ]

    scores = model.predict(pairs, show_progress_bar=len(pairs) > 200)

    log.info(
        f"Cross-encoder complete — "
        f"logit range: [{scores.min():.2f}, {scores.max():.2f}]"
    )
    return np.array(scores, dtype=np.float32)