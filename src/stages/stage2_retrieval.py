"""
stages/stage2_retrieval.py
──────────────────────────
Stage 2 — Honeypot Gate + Dual-Chamber BM25 Retrieval.

Pipeline:
  1. Load all candidates
  2. F8 honeypot gate (vectorised numpy, all candidates)
  3. Dual-chamber BM25 — Chamber A (raw JD) + Chamber B (expanded)
  4. Union chambers → retrieval pool
  5. Free BM25 corpus + index before returning (RAM for Stage 3)

Returns 4-tuple: (bm25_pool, valid_ids, top_a, top_b)
top_a and top_b passed to Stage 3 for 6-stream RRF fusion.

Runtime: ~90s | Memory peak: ~3GB (corpus freed before return)
"""

import gc
from pathlib import Path

from src.config import (
    CANDIDATES_PATH,
    BM25_CHAMBER_A_TOP_K,
    BM25_CHAMBER_B_TOP_K,
    BM25_UNION_CAP,
    DEV_CANDIDATE_LIMIT,
)
from src.utils.data_loader import load_all_candidates
from src.utils.logger import get_logger, log_pool_transition, memory_gate
from src.features.f8_honeypot_gate import (
    compute_honeypot_scores,
    apply_honeypot_gate,
)
from src.retrieval.bm25_retrieval import (
    build_bm25_index,
    retrieve_chamber_a,
    retrieve_chamber_b,
    union_chambers,
)

log = get_logger(__name__)


def run(
    candidates_path: Path,
    jd_object: dict,
) -> tuple:
    """
    Execute Stage 2 — Honeypot Gate + BM25 Retrieval.

    Parameters
    ----------
    candidates_path : Path to candidates JSONL file.
    jd_object       : Parsed JD object from Stage 1.

    Returns
    -------
    tuple: (bm25_pool, valid_candidate_ids, top_a, top_b)
      bm25_pool           : list[dict] — union of both chambers
      valid_candidate_ids : set[str]  — all IDs from source file
      top_a               : list[dict] — Chamber A results for RRF
      top_b               : list[dict] — Chamber B results for RRF
    """
    # ── Load candidates ───────────────────────────────────────────────────────
    all_candidates = load_all_candidates(candidates_path)

    # Dev profile: limit candidate pool for fast local testing
    if DEV_CANDIDATE_LIMIT is not None:
        log.info(
            f"DEV mode: limiting to {DEV_CANDIDATE_LIMIT} candidates"
        )
        all_candidates = all_candidates[:DEV_CANDIDATE_LIMIT]

    valid_candidate_ids = {c["candidate_id"] for c in all_candidates}
    log.info(f"Loaded {len(all_candidates):,} candidates")

    # ── F8: Honeypot gate ─────────────────────────────────────────────────────
    honeypot_scores  = compute_honeypot_scores(all_candidates)
    clean_candidates = apply_honeypot_gate(all_candidates, honeypot_scores)

    # Free original list — clean_candidates is what we need
    del all_candidates
    gc.collect()

    # ── BM25 index ────────────────────────────────────────────────────────────
    log.info("Building BM25 index ...")
    corpus, index = build_bm25_index(clean_candidates)

    # ── Chamber A: raw JD terms ───────────────────────────────────────────────
    jd_text = jd_object.get("raw_text", "")
    top_a   = retrieve_chamber_a(
        index, corpus, clean_candidates,
        jd_text, BM25_CHAMBER_A_TOP_K,
    )
    log.info(f"Chamber A: {len(top_a):,} candidates")

    # ── Chamber B: expanded taxonomy terms ───────────────────────────────────
    expanded = jd_object.get("expanded_requirements", [])
    top_b    = retrieve_chamber_b(
        index, corpus, clean_candidates,
        expanded, BM25_CHAMBER_B_TOP_K,
    )
    log.info(f"Chamber B: {len(top_b):,} candidates")

    # ── Union chambers ────────────────────────────────────────────────────────
    bm25_pool = union_chambers(top_a, top_b, cap=BM25_UNION_CAP)

    # ── Free BM25 memory before Stage 3 ──────────────────────────────────────
    # BM25 corpus (~500MB) must be released before embedding stage
    del corpus
    del index
    del clean_candidates
    gc.collect()
    memory_gate("Stage 2 BM25", log)

    log_pool_transition(
        log, "Stage 2",
        len(valid_candidate_ids), len(bm25_pool),
        note="after honeypot gate + dual-chamber BM25"
    )

    return bm25_pool, valid_candidate_ids, top_a, top_b