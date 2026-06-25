"""
retrieval/rrf_fusion.py
────────────────────────
6-stream Reciprocal Rank Fusion.

Fuses rankings from:
  Dense:  Q1 (HyDE primary), Q2 (technical), Q3 (experience), Q4 (vibe)
  Sparse: BM25_A (raw JD terms), BM25_B (expanded taxonomy)

Formula (corrected — each stream is independent):
  RRF(d) = Σ_{q ∈ streams} 1 / (k + rank_q(d))

If a candidate is absent from a stream, that term = 0.
This is mathematically correct — a missing candidate receives
no credit from that stream, not a penalty.

k=60 is the standard RRF constant, empirically validated
across IR benchmarks.
"""

from collections import defaultdict
from src.config import RRF_K, RETRIEVAL_FINAL_POOL
from src.utils.logger import get_logger, log_pool_transition

log = get_logger(__name__)


def compute_rrf_scores(
    stream_results: dict,
    k: int = RRF_K,
) -> dict:
    """
    Compute RRF scores across all ranking streams.

    Parameters
    ----------
    stream_results : Dict of {stream_name: [ordered_candidate_indices]}.
                     Indices reference positions in the candidate list.
    k              : RRF constant (default 60).

    Returns
    -------
    dict of {candidate_index: rrf_score}
    """
    rrf_scores = defaultdict(float)

    for stream_name, ranked_indices in stream_results.items():
        for rank, candidate_idx in enumerate(ranked_indices, start=1):
            rrf_scores[candidate_idx] += 1.0 / (k + rank)

    log.info(
        f"RRF fusion: {len(rrf_scores):,} unique candidates scored "
        f"across {len(stream_results)} streams"
    )

    return dict(rrf_scores)


def select_top_by_rrf(
    rrf_scores: dict,
    candidates: list[dict],
    top_k: int = RETRIEVAL_FINAL_POOL,
) -> tuple[list[dict], list[float]]:
    """
    Select top candidates by RRF score.

    Parameters
    ----------
    rrf_scores : Dict of {candidate_index: rrf_score}.
    candidates : Full candidate list (BM25 pool).
    top_k      : Number of candidates to return.

    Returns
    -------
    tuple:
      top_candidates : list[dict] ordered by RRF score descending
      top_rrf_scores : list[float] corresponding RRF scores
    """
    sorted_items = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    top_candidates = [candidates[idx] for idx, _ in sorted_items]
    top_scores     = [score for _, score in sorted_items]

    log_pool_transition(
        log, "RRF fusion",
        len(rrf_scores), len(top_candidates),
        note=f"top RRF score: {top_scores[0]:.4f}" if top_scores else ""
    )

    return top_candidates, top_scores