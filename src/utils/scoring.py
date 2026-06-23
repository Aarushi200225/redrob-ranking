"""
utils/scoring.py
────────────────
Composite score computation for the final ranking stage.

Responsibilities:
  - normalize_batch: MinMax normalisation over a candidate batch
  - compute_composite_score: weighted combination of all signals
  - apply_tiebreaking: deterministic rank assignment on score ties

Bug fix applied:
  ms-marco cross-encoder outputs unbounded logits (typically -8 to +8).
  Adding raw logits to normalised [0-1] feature scores causes the
  cross-encoder to dominate final scoring regardless of assigned weights.
  All additive components are MinMax normalised over the 500-candidate
  batch before computing the weighted sum.
"""

import numpy as np
from src.config import SCORE_WEIGHTS
from src.utils.logger import get_logger

log = get_logger(__name__)


def normalize_batch(scores: np.ndarray) -> np.ndarray:
    """
    MinMax normalise a score array to [0, 1] over the batch.

    Applied to every additive composite component before scoring.
    Handles the edge case where all scores are identical.

    Parameters
    ----------
    scores : np.ndarray of raw scores (any range).

    Returns
    -------
    np.ndarray normalised to [0, 1].
    """
    s_min = scores.min()
    s_max = scores.max()

    if s_max - s_min < 1e-8:
        # All scores identical — return neutral 0.5
        return np.ones_like(scores) * 0.5

    return (scores - s_min) / (s_max - s_min + 1e-8)


def compute_composite_scores(
    rrf_scores: np.ndarray,
    ce_scores: np.ndarray,
    f2_experience: np.ndarray,
    f3_skills: np.ndarray,
    f4_vibe: np.ndarray,
    f5_availability: np.ndarray,
    f6_market: np.ndarray,
    f7_location: np.ndarray,
    recency_multipliers: np.ndarray,
) -> np.ndarray:
    """
    Compute final composite scores for the reranking pool.

    All additive components are MinMax normalised over the batch
    before the weighted sum. Location and recency are applied as
    multiplicative dampeners after the additive sum.

    Parameters
    ----------
    rrf_scores       : RRF retrieval scores (small floats, unbounded low end).
    ce_scores        : Cross-encoder logits (unbounded, typically -8 to +8).
    f2_experience    : Experience quality scores [0-1].
    f3_skills        : Skills match scores [0-1].
    f4_vibe          : Vibe/behavioural scores [0-1].
    f5_availability  : Availability scores [0-1].
    f6_market        : Market validation scores [0-1].
    f7_location      : Location fit multipliers [0-1].
    recency_multipliers : Recency decay multipliers [0-1].

    Returns
    -------
    np.ndarray of final composite scores.
    """
    # Normalise all additive components over this batch
    rrf_norm  = normalize_batch(rrf_scores)
    ce_norm   = normalize_batch(ce_scores)
    f2_norm   = normalize_batch(f2_experience)
    f3_norm   = normalize_batch(f3_skills)
    f4_norm   = normalize_batch(f4_vibe)
    f5_norm   = normalize_batch(f5_availability)
    f6_norm   = normalize_batch(f6_market)

    w = SCORE_WEIGHTS

    additive = (
        w["semantic"]      * rrf_norm
        + w["cross_encoder"] * ce_norm
        + w["experience"]    * f2_norm
        + w["skills"]        * f3_norm
        + w["vibe"]          * f4_norm
        + w["availability"]  * f5_norm
        + w["market"]        * f6_norm
    )

    # Location and recency are multiplicative — applied after additive sum
    final = additive * f7_location * recency_multipliers

    log.info(
        f"Composite scores — min: {final.min():.4f}, "
        f"max: {final.max():.4f}, "
        f"mean: {final.mean():.4f}"
    )

    return final


def apply_tiebreaking(
    candidates: list[dict],
    scores: np.ndarray,
) -> list[tuple[dict, float]]:
    """
    Sort candidates by score descending with deterministic tie-breaking.

    Tie-breaking order (per hackathon spec):
      1. score descending
      2. recruiter_response_rate descending
      3. candidate_id ascending

    Parameters
    ----------
    candidates : List of candidate dicts.
    scores     : Composite score per candidate (same order as candidates).

    Returns
    -------
    List of (candidate, score) tuples sorted by rank order.
    """
    paired = list(zip(candidates, scores.tolist()))

    def sort_key(item):
        candidate, score = item
        response_rate = candidate.get(
            "redrob_signals", {}
        ).get("recruiter_response_rate", 0.0)
        cand_id = candidate.get("candidate_id", "")
        # Negate score and response_rate for descending sort
        return (-score, -response_rate, cand_id)

    paired.sort(key=sort_key)
    return paired