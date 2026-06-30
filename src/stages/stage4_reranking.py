"""
stages/stage4_reranking.py
──────────────────────────
Stage 4 — Feature Extraction + Multi-Signal Reranking.

Pipeline:
  Pass 0: F1-F9 sequential feature extraction (2000 candidates)
          F1 consulting gate → hard elimination
          Structural pre-filter: F2 < 0.20 AND F3 < 0.20
            → assign ce_score=0.0, skip cross-encoder
            → candidate stays in pool (other signals may rescue)
  Pass A: Cross-encoder reranking on plausible candidates → top 500
  Pass B: MinMax-normalised composite scoring → top 100

Bug fix: cross-encoder outputs unbounded logits (-8 to +8).
All additive components MinMax-normalised over batch before
weighted sum. Without this, CE dominates regardless of weights.

Runtime: ~55s | Memory peak: ~2GB
"""

import numpy as np

from src.config import (
    RERANK_POOL_SIZE,
    FINAL_TOP_K,
    SCORE_WEIGHTS,
    CE_PREFILTER_F2_THRESHOLD,
    CE_PREFILTER_F3_THRESHOLD,
)
from src.utils.logger import get_logger, log_pool_transition, memory_gate
from src.utils.scoring import compute_composite_scores, apply_tiebreaking

from src.features.f1_title_fit       import score_f1
from src.features.f2_experience_quality import score_f2
from src.features.f3_skills_match    import score_f3
from src.features.f4_vibe_score      import score_f4
from src.features.f5_availability    import score_f5
from src.features.f6_market_validation import score_f6
from src.features.f7_location_fit    import score_f7
from src.features.f9_salary_fit      import score_f9_salary_fit

log = get_logger(__name__)


def _extract_features(
    candidates: list[dict],
    jd_object: dict,
    rrf_score_map: dict,
) -> tuple:
    """
    Extract F1-F9 features sequentially for all candidates.

    Applies:
      - F1 consulting-only hard gate (returns -1.0 → eliminated)
      - Structural pre-filter: F2 < 0.20 AND F3 < 0.20
          → ce_score pre-assigned 0.0, bypass cross-encoder
          → candidate stays in pool for other signals

    Returns
    -------
    tuple of aligned lists/arrays for survivors:
      (survivors, rrf_arr, f2_arr, f3_arr, f4_arr,
       f5_arr, recency_arr, f6_arr, f7_arr, f9_arr,
       ce_preassigned_arr, needs_ce_arr)
    """
    survivors       = []
    rrf_scores      = []
    f2_scores       = []
    f3_scores       = []
    f4_scores       = []
    f5_scores       = []
    recency_mults   = []
    f6_scores       = []
    f7_scores       = []
    f9_scores       = []
    ce_preassigned  = []   # Pre-assigned CE score (0.0 for filtered)
    needs_ce        = []   # True = needs cross-encoder inference

    consulting_gate_count = 0
    prefilter_count       = 0

    for candidate in candidates:
        cid = candidate.get("candidate_id", "")

        # ── F1: consulting-only hard gate ─────────────────────────────────────
        f1 = score_f1(candidate, jd_object)
        if f1 < 0:
            consulting_gate_count += 1
            continue

        # ── F2-F9 feature extraction ──────────────────────────────────────────
        f2          = score_f2(candidate, jd_object)
        f3          = score_f3(candidate, jd_object)
        f4          = score_f4(candidate, jd_object)
        f5, r_mult  = score_f5(candidate, jd_object)
        f6          = score_f6(candidate, jd_object)
        f7          = score_f7(candidate, jd_object)
        f9          = score_f9_salary_fit(candidate, jd_object)
        rrf         = rrf_score_map.get(cid, 0.0)

        # ── Structural pre-filter ─────────────────────────────────────────────
        # Both F2 AND F3 below threshold = clearly not an AI engineer
        # Assign ce_score=0.0, skip inference, keep in pool
        if f2 < CE_PREFILTER_F2_THRESHOLD and f3 < CE_PREFILTER_F3_THRESHOLD:
            prefilter_count += 1
            ce_preassigned.append(0.0)
            needs_ce.append(False)
        else:
            ce_preassigned.append(None)   # Will be filled by cross-encoder
            needs_ce.append(True)

        survivors.append(candidate)
        rrf_scores.append(rrf)
        f2_scores.append(f2)
        f3_scores.append(f3)
        f4_scores.append(f4)
        f5_scores.append(f5)
        recency_mults.append(r_mult)
        f6_scores.append(f6)
        f7_scores.append(f7)
        f9_scores.append(f9)

    log.info(
        f"Feature extraction: "
        f"{consulting_gate_count} consulting-only eliminated, "
        f"{prefilter_count} pre-filtered (CE=0.0), "
        f"{sum(needs_ce)} candidates need cross-encoder"
    )

    return (
        survivors,
        np.array(rrf_scores,    dtype=np.float32),
        np.array(f2_scores,     dtype=np.float32),
        np.array(f3_scores,     dtype=np.float32),
        np.array(f4_scores,     dtype=np.float32),
        np.array(f5_scores,     dtype=np.float32),
        np.array(recency_mults, dtype=np.float32),
        np.array(f6_scores,     dtype=np.float32),
        np.array(f7_scores,     dtype=np.float32),
        np.array(f9_scores,     dtype=np.float32),
        ce_preassigned,
        needs_ce,
    )


def run(
    retrieval_pool: list[dict],
    jd_object: dict,
    rrf_score_map: dict | None = None,
) -> list[dict]:
    """
    Execute Stage 4 — Feature Extraction + Multi-Signal Reranking.

    Parameters
    ----------
    retrieval_pool : Top 2000 candidates from Stage 3.
    jd_object      : Parsed JD object from Stage 1.
    rrf_score_map  : {candidate_id: rrf_score} from Stage 3.

    Returns
    -------
    list[dict] — top 100 candidates with _score and
                 _score_breakdown attached.
    """
    from src.models.model_context import ModelContext
    from src.models.cross_encoder import load_cross_encoder, score_pairs

    if rrf_score_map is None:
        rrf_score_map = {}

    log.info(f"Stage 4: {len(retrieval_pool):,} candidates")

    # ── Feature extraction ────────────────────────────────────────────────────
    (
        survivors,
        rrf_arr, f2_arr, f3_arr, f4_arr,
        f5_arr, recency_arr, f6_arr, f7_arr, f9_arr,
        ce_preassigned, needs_ce,
    ) = _extract_features(retrieval_pool, jd_object, rrf_score_map)

    if len(survivors) < FINAL_TOP_K:
        log.warning(
            f"Only {len(survivors)} survivors after feature extraction "
            f"— fewer than {FINAL_TOP_K} requested"
        )

    log_pool_transition(
        log, "F1-F9 extraction",
        len(retrieval_pool), len(survivors),
        note="after consulting gate"
    )

    # ── Pass A: Cross-encoder reranking ───────────────────────────────────────
    ce_scores = np.zeros(len(survivors), dtype=np.float32)

    # Fill pre-assigned scores
    for i, (preassigned, run_ce) in enumerate(
        zip(ce_preassigned, needs_ce)
    ):
        if not run_ce:
            ce_scores[i] = 0.0

    # Run cross-encoder on candidates that need it
    ce_candidates = [
        (i, c) for i, (c, run_ce)
        in enumerate(zip(survivors, needs_ce)) if run_ce
    ]

    if ce_candidates:
        ce_indices   = [i for i, _ in ce_candidates]
        ce_cands     = [c for _, c in ce_candidates]
        jd_text      = jd_object.get("raw_text", "")

        try:
            with ModelContext(load_cross_encoder) as cross_encoder:
                raw_scores = score_pairs(
                    cross_encoder, jd_text, ce_cands
                )
            for idx, score in zip(ce_indices, raw_scores):
                ce_scores[idx] = score
        except Exception as exc:
            log.warning(
                f"Cross-encoder failed ({exc}) — "
                f"using zero scores, RRF+features only"
            )

    memory_gate("Stage 4 cross-encoder", log)

    # ── Select top RERANK_POOL_SIZE by cross-encoder ──────────────────────────
    top_ce_indices = np.argsort(ce_scores)[::-1][:RERANK_POOL_SIZE]

    pool_survivors  = [survivors[i]     for i in top_ce_indices]
    pool_ce         = ce_scores[top_ce_indices]
    pool_rrf        = rrf_arr[top_ce_indices]
    pool_f2         = f2_arr[top_ce_indices]
    pool_f3         = f3_arr[top_ce_indices]
    pool_f4         = f4_arr[top_ce_indices]
    pool_f5         = f5_arr[top_ce_indices]
    pool_recency    = recency_arr[top_ce_indices]
    pool_f6         = f6_arr[top_ce_indices]
    pool_f7         = f7_arr[top_ce_indices]
    pool_f9         = f9_arr[top_ce_indices]

    log_pool_transition(
        log, "Pass A cross-encoder",
        len(survivors), len(pool_survivors)
    )

    # ── Pass B: MinMax-normalised composite scoring ───────────────────────────
    log.info("Computing MinMax-normalised composite scores ...")

    final_scores = compute_composite_scores(
        rrf_scores        = pool_rrf,
        ce_scores         = pool_ce,
        f2_experience     = pool_f2,
        f3_skills         = pool_f3,
        f4_vibe           = pool_f4,
        f5_availability   = pool_f5,
        f6_market         = pool_f6,
        f7_location       = pool_f7,
        recency_multipliers = pool_recency,
        f9_salary         = pool_f9,
    )

    # ── Tie-breaking + top 100 ────────────────────────────────────────────────
    ranked     = apply_tiebreaking(pool_survivors, final_scores)
    top_100_pairs = ranked[:FINAL_TOP_K]

    top_100 = []
    for rank_idx, (candidate, score) in enumerate(top_100_pairs, start=1):
        c = dict(candidate)
        c["_score"] = float(score)
        c["_rank"]  = rank_idx
        c["_score_breakdown"] = {
            "cross_encoder": float(
                pool_ce[pool_survivors.index(candidate)]
                if candidate in pool_survivors else 0.0
            ),
            "rrf":      float(pool_rrf[pool_survivors.index(candidate)]
                              if candidate in pool_survivors else 0.0),
            "vibe":     float(pool_f4[pool_survivors.index(candidate)]
                              if candidate in pool_survivors else 0.0),
            "experience": float(pool_f2[pool_survivors.index(candidate)]
                                if candidate in pool_survivors else 0.0),
        }
        top_100.append(c)

    log_pool_transition(
        log, "Pass B composite scoring",
        len(pool_survivors), len(top_100)
    )

    return top_100