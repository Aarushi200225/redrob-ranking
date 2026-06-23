"""
features/f4_vibe_score.py
──────────────────────────
F4 — Behavioral/Vibe Score.

Primary differentiator — most competing systems will not model this.

Signals:
  - Vibe semantic matching: cosine sim between candidate text
    and pre-computed vibe phrase cluster embeddings
  - Writing quality proxy: lexical diversity + description length
  - Startup DNA: % of career at small companies
  - GitHub activity score

Vibe cluster embeddings are pre-computed in Stage 1 and passed
in via jd_object['vibe_cluster_vecs'].
"""

import numpy as np
from src.utils.logger import get_logger

log = get_logger(__name__)

COMPANY_SIZE_STARTUP = {"1-10", "11-50", "51-200"}


def _vibe_semantic_score(
    candidate_text: str,
    vibe_cluster_vecs: dict,
    candidate_vec: np.ndarray | None,
) -> float:
    """
    Compute semantic similarity between candidate text
    and vibe phrase cluster embeddings.

    Parameters
    ----------
    candidate_text     : Candidate summary + career descriptions.
    vibe_cluster_vecs  : Dict of {signal_name: np.ndarray embedding}.
    candidate_vec      : Pre-computed candidate embedding (optional).

    Returns
    -------
    float — average cosine similarity across vibe dimensions.
    """
    if not vibe_cluster_vecs or candidate_vec is None:
        return 0.5  # Neutral if embeddings not available

    scores = []
    for signal_name, vibe_vec in vibe_cluster_vecs.items():
        # Cosine similarity (vectors are already normalised)
        sim = float(np.dot(candidate_vec, vibe_vec))
        sim = max(0.0, min(1.0, (sim + 1) / 2))  # Map [-1,1] → [0,1]
        scores.append(sim)

    return float(np.mean(scores)) if scores else 0.5


def _writing_quality_score(candidate: dict) -> float:
    """
    Proxy for writing quality via lexical diversity and length.

    Candidates who write well (async_writer signal) tend to have
    longer, more varied career descriptions.
    """
    texts = []

    summary = candidate.get("profile", {}).get("summary", "")
    if summary:
        texts.append(summary)

    for role in candidate.get("career_history", [])[:3]:
        desc = role.get("description", "")
        if desc:
            texts.append(desc)

    if not texts:
        return 0.3

    combined  = " ".join(texts)
    words     = combined.lower().split()
    if not words:
        return 0.3

    # Lexical diversity — unique words / total words
    diversity = len(set(words)) / len(words)

    # Length score — longer descriptions signal more thoughtful writing
    avg_len   = len(combined) / len(texts)
    length_score = min(1.0, avg_len / 500)

    return float(0.5 * diversity + 0.5 * length_score)


def _startup_dna_score(candidate: dict) -> float:
    """
    Score percentage of career at small/startup companies.
    """
    career = candidate.get("career_history", [])
    if not career:
        return 0.3

    total_months   = sum(r.get("duration_months", 0) for r in career)
    startup_months = sum(
        r.get("duration_months", 0)
        for r in career
        if r.get("company_size", "") in COMPANY_SIZE_STARTUP
    )

    if total_months == 0:
        return 0.3

    ratio = startup_months / total_months
    return float(min(1.0, ratio * 1.2))  # Slight boost for startup-heavy


def _github_score(candidate: dict) -> float:
    """
    Normalise GitHub activity score.
    -1 (no GitHub linked) → 0.3 neutral, not 0.
    """
    raw = candidate.get(
        "redrob_signals", {}
    ).get("github_activity_score", -1)

    if raw == -1:
        return 0.3  # No GitHub — neutral, not penalised
    return float(max(0.0, min(1.0, raw / 100.0)))


def score_f4(
    candidate: dict,
    jd_object: dict,
    candidate_vec: np.ndarray | None = None,
) -> float:
    """
    Compute F4 Behavioral/Vibe score.

    Parameters
    ----------
    candidate      : Raw candidate dict.
    jd_object      : Parsed JD object (contains vibe_cluster_vecs).
    candidate_vec  : Pre-computed embedding vector for this candidate.

    Returns
    -------
    float in [0, 1].
    """
    vibe_vecs = jd_object.get("vibe_cluster_vecs", {})

    # Build candidate text for vibe matching
    profile     = candidate.get("profile", {})
    summary     = profile.get("summary", "")
    career_text = " ".join(
        r.get("description", "")
        for r in candidate.get("career_history", [])[:3]
    )
    candidate_text = f"{summary} {career_text}"

    vibe_semantic = _vibe_semantic_score(
        candidate_text, vibe_vecs, candidate_vec
    )
    writing       = _writing_quality_score(candidate)
    startup       = _startup_dna_score(candidate)
    github        = _github_score(candidate)

    score = (
        0.40 * vibe_semantic
        + 0.25 * writing
        + 0.20 * startup
        + 0.15 * github
    )

    return float(max(0.0, min(1.0, score)))