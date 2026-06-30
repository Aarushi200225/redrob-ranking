"""
utils/scoring.py
────────────────
Composite score computation for the final ranking stage.

Bug fix applied:
  ms-marco cross-encoder outputs unbounded logits (~-8 to +8).
  All additive components MinMax-normalised over the 500-candidate
  batch before weighted sum. Without this, CE dominates final scores
  regardless of assigned weight.

F7 (location) and F9 (salary) applied as multiplicative dampeners
after the additive sum — already bounded [0-1], not normalised.
Recency multiplier applied as final dampener for inactive candidates.
"""

import numpy as np
from src.config import SCORE_WEIGHTS
from src.utils.logger import get_logger

log = get_logger(__name__)


def normalize_batch(scores: np.ndarray) -> np.ndarray:
    """
    MinMax normalise a score array to [0, 1] over the batch.

    Handles edge case where all scores are identical.
    Applied to every additive composite component before scoring.

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
        return np.ones_like(scores) * 0.5

    return (scores - s_min) / (s_max - s_min + 1e-8)


# Alias for external validation scripts
min_max_normalize = normalize_batch


def compute_composite_scores(
    rrf_scores:          np.ndarray,
    ce_scores:           np.ndarray,
    f2_experience:       np.ndarray,
    f3_skills:           np.ndarray,
    f4_vibe:             np.ndarray,
    f5_availability:     np.ndarray,
    f6_market:           np.ndarray,
    f7_location:         np.ndarray,
    recency_multipliers: np.ndarray,
    f9_salary:           np.ndarray,
) -> np.ndarray:
    """
    Compute final composite scores for the reranking pool.

    All additive components MinMax-normalised over the batch.
    F7 (location), F9 (salary), and recency applied multiplicatively.

    Parameters
    ----------
    rrf_scores          : RRF retrieval scores.
    ce_scores           : Cross-encoder logits (unbounded).
    f2_experience       : Experience quality scores [0-1].
    f3_skills           : Skills match scores [0-1].
    f4_vibe             : Vibe/behavioural scores [0-1].
    f5_availability     : Availability scores [0-1].
    f6_market           : Market validation scores [0-1].
    f7_location         : Location fit multipliers [0.30-1.00].
    recency_multipliers : Recency decay multipliers [0.30-1.00].
    f9_salary           : Salary fit multipliers [0.65-1.00].

    Returns
    -------
    np.ndarray of final composite scores.
    """
    # Normalise all additive components to [0, 1] over this batch
    rrf_norm = normalize_batch(rrf_scores)
    ce_norm  = normalize_batch(ce_scores)    # Critical — unbounded logits
    f2_norm  = normalize_batch(f2_experience)
    f3_norm  = normalize_batch(f3_skills)
    f4_norm  = normalize_batch(f4_vibe)
    f5_norm  = normalize_batch(f5_availability)
    f6_norm  = normalize_batch(f6_market)

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

    # Multiplicative dampeners — applied after additive sum
    final = additive * f7_location * f9_salary * recency_multipliers

    log.info(
        f"Composite scores — "
        f"min: {final.min():.4f}, "
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
    scores     : Composite score per candidate.

    Returns
    -------
    List of (candidate, score) tuples sorted by rank order.
    """
    paired = list(zip(candidates, scores.tolist()))

    def sort_key(item):
        candidate, score   = item
        response_rate      = candidate.get(
            "redrob_signals", {}
        ).get("recruiter_response_rate", 0.0)
        cand_id = candidate.get("candidate_id", "")
        return (-score, -response_rate, cand_id)

    paired.sort(key=sort_key)
    return paired