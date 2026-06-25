"""
stages/stage3_embedding.py
──────────────────────────
Stage 3 — Semantic Embedding + Hybrid Retrieval.

Responsibilities:
  - Build text blobs for 15K BM25 pool candidates
  - Batch encode with bge-small-en-v1.5
  - Build FAISS IndexFlatIP
  - Multi-query search (Q1-Q4)
  - Map BM25 stream results to FAISS indices
  - 6-stream RRF fusion → top 2K candidates

Runtime: ~26s
Output:  (retrieval_pool, rrf_score_map)
"""

import numpy as np

from src.config import FAISS_TOP_K_PER_QUERY, RETRIEVAL_FINAL_POOL
from src.utils.logger import get_logger, log_pool_transition
# from src.models.model_context import ModelContext
# from src.models.embedder import load_embedder, batch_encode, build_candidate_text_blob
from src.retrieval.faiss_index import build_faiss_index, multi_query_search
from src.retrieval.rrf_fusion import compute_rrf_scores, select_top_by_rrf

log = get_logger(__name__)


def _build_bm25_stream_indices(
    bm25_pool: list[dict],
    top_a: list[dict],
    top_b: list[dict],
) -> dict:
    """
    Map BM25 chamber results to pool indices for RRF fusion.

    BM25 results are candidate dicts — convert to index positions
    in bm25_pool for consistent index-based RRF scoring.
    """
    id_to_idx = {c["candidate_id"]: i for i, c in enumerate(bm25_pool)}

    bm25_a_indices = [
        id_to_idx[c["candidate_id"]]
        for c in top_a
        if c["candidate_id"] in id_to_idx
    ]
    bm25_b_indices = [
        id_to_idx[c["candidate_id"]]
        for c in top_b
        if c["candidate_id"] in id_to_idx
    ]

    return {
        "BM25_A": bm25_a_indices,
        "BM25_B": bm25_b_indices,
    }


def run(
    bm25_pool: list[dict],
    query_vectors: dict,
    bm25_top_a: list[dict] | None = None,
    bm25_top_b: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """
    Execute Stage 3 — Embedding + Hybrid Retrieval.

    Parameters
    ----------
    bm25_pool           : 15K candidates from Stage 2.
    query_vectors       : {Q1, Q2, Q3, Q4} np.ndarray vectors from Stage 1.
    bm25_top_a          : Top candidates from BM25 Chamber A.
    bm25_top_b          : Top candidates from BM25 Chamber B.

    Returns
    -------
    tuple:
      retrieval_pool : list[dict] — top 2K candidates
      rrf_score_map  : dict {candidate_id: rrf_score}
    """
    from src.models.model_context import ModelContext
    from src.models.embedder import load_embedder, batch_encode, build_candidate_text_blob
    if not bm25_pool:
        raise ValueError("BM25 pool is empty — cannot proceed")

    if not query_vectors:
        raise ValueError(
            "No query vectors provided — Stage 1 may have failed"
        )

    # ── Build text blobs ──────────────────────────────────────────────────────
    log.info(f"Building text blobs for {len(bm25_pool):,} candidates ...")
    text_blobs = [build_candidate_text_blob(c) for c in bm25_pool]

    # ── Batch encode ──────────────────────────────────────────────────────────
    embeddings = None
    with ModelContext(load_embedder) as embedder:
        embeddings = batch_encode(embedder, text_blobs)

    log.info(f"Embeddings shape: {embeddings.shape}")

    # ── FAISS index + multi-query search ──────────────────────────────────────
    faiss_index  = build_faiss_index(embeddings)
    faiss_results = multi_query_search(
        faiss_index,
        query_vectors,
        top_k=FAISS_TOP_K_PER_QUERY,
    )

    # Build BM25 stream indices for 6-stream RRF
    all_streams = {**faiss_results}
    if bm25_top_a and bm25_top_b:
        bm25_stream_indices = _build_bm25_stream_indices(
            bm25_pool, bm25_top_a, bm25_top_b
        )
        all_streams.update(bm25_stream_indices)
        log.info(f"6-stream RRF: 4 dense + 2 sparse streams")
    else:
        log.warning("BM25 streams not available — using 4-stream dense RRF only")


    rrf_scores = compute_rrf_scores(all_streams)
    retrieval_pool, top_rrf_scores = select_top_by_rrf(
        rrf_scores, bm25_pool, top_k=RETRIEVAL_FINAL_POOL
    )

    # Build candidate_id → rrf_score map for composite scoring
    rrf_score_map = {
        c["candidate_id"]: score
        for c, score in zip(retrieval_pool, top_rrf_scores)
    }

    log_pool_transition(
        log, "Stage 3",
        len(bm25_pool), len(retrieval_pool),
        note="after 6-stream RRF fusion"
    )

    return retrieval_pool, rrf_score_map