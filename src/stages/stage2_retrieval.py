"""
stages/stage2_retrieval.py
──────────────────────────
Stage 2 — Honeypot Gate + Dual-Chamber BM25 Retrieval.

Responsibilities:
  - F8 honeypot gate on all 100K candidates (vectorised numpy)
  - Dual-chamber BM25 retrieval
  - Union of chambers → 15K candidate pool

Runtime: ~19s
Output:  (bm25_pool, valid_candidate_ids)
"""

from pathlib import Path

from src.config import (
    CANDIDATES_PATH,
    BM25_CHAMBER_A_TOP_K,
    BM25_CHAMBER_B_TOP_K,
    BM25_UNION_CAP,
)
from src.utils.data_loader import load_all_candidates
from src.utils.logger import get_logger, log_pool_transition
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
) -> tuple[list[dict], set[str], list[dict], list[dict]]:
    """
    Execute Stage 2 — Honeypot Gate + BM25 Retrieval.

    Parameters
    ----------
    candidates_path : Path to candidates JSONL file.
    jd_object       : Parsed JD object from Stage 1.

    Returns
    -------
    tuple:
      bm25_pool          : list[dict] — up to 15K candidates
      valid_candidate_ids: set[str]  — all IDs from source file
      top_a              : list[dict] — candidates from Chamber A
      top_b              : list[dict] — candidates from Chamber B
    """
    # ── Load all candidates ───────────────────────────────────────────────────
    candidates = load_all_candidates(candidates_path)
    valid_candidate_ids = {c["candidate_id"] for c in candidates}

    log.info(f"Loaded {len(candidates):,} candidates")

    # ── F8: Honeypot gate on full pool ────────────────────────────────────────
    honeypot_scores = compute_honeypot_scores(candidates)
    clean_candidates = apply_honeypot_gate(candidates, honeypot_scores)

    # ── Build BM25 index ──────────────────────────────────────────────────────
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

    # ── Union ─────────────────────────────────────────────────────────────────
    bm25_pool = union_chambers(top_a, top_b, cap=BM25_UNION_CAP)

    # Free BM25 corpus memory before returning — Stage 3 needs the RAM
    del corpus
    del index
    del clean_candidates
    del candidates
    import gc
    gc.collect()

    log_pool_transition(
        log, "Stage 2",
        len(valid_candidate_ids), len(bm25_pool),
        note="after honeypot gate + dual-chamber BM25"
    )


    return bm25_pool, valid_candidate_ids, top_a, top_b