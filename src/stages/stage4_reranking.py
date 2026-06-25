"""
stages/stage4_reranking.py
──────────────────────────
Stage 4 — Feature Extraction + Multi-Signal Reranking.

Responsibilities:
  - F1-F7 sequential feature extraction on 2K candidates
  - Hard elimination of consulting-only candidates (F1 = -1.0)
  - Pass A: cross-encoder reranking 2K → 500
  - Pass B: normalised composite scoring 500 → 100

Runtime: ~65s
Output:  list[dict] — top 100 candidates with score_breakdown
"""

import numpy as np

from src.config import (
    CROSS_ENCODER_TOP_K,
    RERANK_POOL_SIZE,
    FINAL_TOP_K,
)
from src.utils.logger import get_logger, log_pool_transition
from src.utils.scoring import compute_composite_scores, apply_tiebreaking
# from src.models.model_context import ModelContext
# from src.models.cross_encoder import (
#     load_cross_encoder,
#     score_pairs,
# )
from src.features.f1_title_fit import score_f1
from src.features.f2_experience_quality import score_f2
from src.features.f3_skills_match import score_f3
from src.features.f4_vibe_score import score_f4
from src.features.f5_availability import score_f5
from src.features.f6_market_validation import score_f6
from src.features.f7_location_fit import score_f7

log = get_logger(__name__)


def _extract_features(
    candidates: list[dict],
    jd_object: dict,
    rrf_score_map: dict,
    candidate_vecs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Extract F1-F7 features for all candidates sequentially.

    Returns
    -------
    tuple:
      feature_matrix    : np.ndarray [n_survivors, 6] — F2-F7 scores
      rrf_scores        : np.ndarray [n_survivors] — RRF scores
      survivors         : list[dict] — candidates passing F1 hard gate
    """
    survivors    = []
    f2_scores    = []
    f3_scores    = []
    f4_scores    = []
    f5_scores    = []
    recency_mults = []
    f6_scores    = []
    f7_scores    = []
    rrf_scores   = []

    for i, candidate in enumerate(candidates):
        cid = candidate.get("candidate_id", "")

        # F1 — hard gate check (consulting-only → -1.0)
        f1 = score_f1(candidate, jd_object)
        if f1 < 0:
            continue  # Hard eliminated

        # F2-F7
        f2        = score_f2(candidate, jd_object)
        f3        = score_f3(candidate, jd_object)

        # F4 — pass candidate embedding vec if available
        cand_vec  = candidate_vecs[i] if candidate_vecs is not None else None
        f4        = score_f4(candidate, jd_object, cand_vec)

        f5, r_mult = score_f5(candidate, jd_object)
        f6        = score_f6(candidate, jd_object)
        f7        = score_f7(candidate, jd_object)
        rrf       = rrf_score_map.get(cid, 0.0)

        survivors.append(candidate)
        f2_scores.append(f2)
        f3_scores.append(f3)
        f4_scores.append(f4)
        f5_scores.append(f5)
        recency_mults.append(r_mult)
        f6_scores.append(f6)
        f7_scores.append(f7)
        rrf_scores.append(rrf)

    n = len(survivors)
    log.info(
        f"Feature extraction complete — "
        f"{n:,} survivors from {len(candidates):,} candidates"
    )

    feature_matrix = np.column_stack([
        f2_scores, f3_scores, f4_scores,
        f5_scores, f6_scores,
    ]).astype(np.float32)

    return (
        feature_matrix,
        np.array(rrf_scores, dtype=np.float32),
        np.array(f7_scores, dtype=np.float32),
        np.array(recency_mults, dtype=np.float32),
        survivors,
    )


def run(
    retrieval_pool: list[dict],
    jd_object: dict,
    rrf_score_map: dict | None = None,
) -> list[dict]:
    """
    Execute Stage 4 — Feature Extraction + Reranking.

    Parameters
    ----------
    retrieval_pool : 2K candidates from Stage 3.
    jd_object      : Parsed JD object from Stage 1.
    rrf_score_map  : {candidate_id: rrf_score} from Stage 3.

    Returns
    -------
    list[dict] — top 100 candidates, each with 'score' and
                 'score_breakdown' fields attached.
    """
    from src.models.model_context import ModelContext
    from src.models.cross_encoder import load_cross_encoder, score_pairs
    if rrf_score_map is None:
        rrf_score_map = {}

    # ── F1-F7 feature extraction ──────────────────────────────────────────────
    log.info("Extracting features F1-F7 ...")
    (
        feature_matrix,
        rrf_scores,
        f7_scores,
        recency_mults,
        survivors,
    ) = _extract_features(retrieval_pool, jd_object, rrf_score_map)

    if len(survivors) < FINAL_TOP_K:
        log.warning(
            f"Only {len(survivors)} survivors — "
            f"fewer than {FINAL_TOP_K} requested"
        )

    log_pool_transition(
        log, "F1-F7 extraction",
        len(retrieval_pool), len(survivors),
        note="after consulting-only gate"
    )

    # ── Pass A: Cross-encoder reranking ───────────────────────────────────────
    log.info(f"Cross-encoder reranking {len(survivors):,} candidates ...")
    jd_text    = jd_object.get("raw_text", "")
    ce_scores  = np.zeros(len(survivors), dtype=np.float32)

    try:
        with ModelContext(load_cross_encoder) as cross_encoder:
            ce_scores = score_pairs(cross_encoder, jd_text, survivors)
    except Exception as exc:
        log.warning(f"Cross-encoder failed ({exc}) — using zero scores")

    # Select top RERANK_POOL_SIZE by cross-encoder score
    ce_top_indices = np.argsort(ce_scores)[::-1][:RERANK_POOL_SIZE]

    rerank_pool     = [survivors[i]    for i in ce_top_indices]
    ce_pool_scores  = ce_scores[ce_top_indices]
    rrf_pool_scores = rrf_scores[ce_top_indices]
    f2_pool         = feature_matrix[ce_top_indices, 0]
    f3_pool         = feature_matrix[ce_top_indices, 1]
    f4_pool         = feature_matrix[ce_top_indices, 2]
    f5_pool         = feature_matrix[ce_top_indices, 3]
    f6_pool         = feature_matrix[ce_top_indices, 4]
    f7_pool         = f7_scores[ce_top_indices]
    recency_pool    = recency_mults[ce_top_indices]

    log_pool_transition(
        log, "Pass A cross-encoder",
        len(survivors), len(rerank_pool)
    )

    # ── Pass B: Normalised composite scoring ──────────────────────────────────
    log.info("Computing normalised composite scores ...")
    final_scores = compute_composite_scores(
        rrf_scores        = rrf_pool_scores,
        ce_scores         = ce_pool_scores,
        f2_experience     = f2_pool,
        f3_skills         = f3_pool,
        f4_vibe           = f4_pool,
        f5_availability   = f5_pool,
        f6_market         = f6_pool,
        f7_location       = f7_pool,
        recency_multipliers = recency_pool,
    )

    # ── Tie-breaking + top 100 selection ─────────────────────────────────────
    ranked = apply_tiebreaking(rerank_pool, final_scores)
    top_100_pairs = ranked[:FINAL_TOP_K]

    # Attach score and score_breakdown to each candidate
    top_100 = []
    for rank_idx, (candidate, score) in enumerate(top_100_pairs):
        candidate = dict(candidate)  # shallow copy — don't mutate original
        candidate["_score"] = float(score)
        candidate["_rank"]  = rank_idx + 1
        candidate["_score_breakdown"] = {
            "cross_encoder": float(ce_pool_scores[
                rerank_pool.index(
                    top_100_pairs[rank_idx][0]
                )
            ]) if top_100_pairs[rank_idx][0] in rerank_pool else 0.0,
        }
        top_100.append(candidate)

    log_pool_transition(
        log, "Pass B composite scoring",
        len(rerank_pool), len(top_100)
    )

    return top_100