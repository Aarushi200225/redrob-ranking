"""
pipeline.py
───────────
Top-level orchestrator for the Redrob Intelligent Candidate Ranking System.

Wires all five stages in sequence with the full exception-handling
hierarchy. Tracks total wall-clock time and writes the validated CSV.

Entry points:
    python -m src.pipeline
    make run
"""

import argparse
import sys
import time
from pathlib import Path

from src.config import (
    CANDIDATES_PATH,
    JD_PATH,
    OUTPUT_CSV_PATH,
    TIMING_TARGETS,
)
from src.utils.logger import get_logger, stage_timer, log_fallback

log = get_logger("pipeline")


# ── Lazy stage imports ────────────────────────────────────────────────────────
# Imported inside functions to avoid loading model dependencies at import time.

def _run_stage1(jd_path: Path) -> dict:
    from src.stages.stage1_jd_intelligence import run as stage1
    return stage1(jd_path)


def _run_stage2(candidates_path: Path, jd_object: dict) -> tuple:
    from src.stages.stage2_retrieval import run as stage2
    return stage2(candidates_path, jd_object)


def _run_stage3(bm25_pool: list, query_vectors: dict) -> list:
    from src.stages.stage3_embedding import run as stage3
    return stage3(bm25_pool, query_vectors)


def _run_stage4(retrieval_pool: list, jd_object: dict) -> list:
    from src.stages.stage4_reranking import run as stage4
    return stage4(retrieval_pool, jd_object)


def _run_stage5(top_100: list, jd_object: dict, output_path: Path) -> Path:
    from src.stages.stage5_output import run as stage5
    return stage5(top_100, jd_object, output_path)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    jd_path: Path = JD_PATH,
    candidates_path: Path = CANDIDATES_PATH,
    output_path: Path = OUTPUT_CSV_PATH,
) -> Path:
    """
    Execute the full five-stage ranking pipeline.

    Parameters
    ----------
    jd_path         : Path to the job description text file.
    candidates_path : Path to gzipped JSONL candidate pool.
    output_path     : Destination for ranked_output.csv.

    Returns
    -------
    Path to the written output CSV.

    Exception handling hierarchy
    ────────────────────────────
    Stage 1 failure  → fallback minimal JD object; pipeline continues
    Stage 2 failure  → fatal re-raise (no candidates = no ranking)
    Stage 3 failure  → fatal re-raise (no retrieval pool = no ranking)
    Stage 4 Pass A   → fallback: skip cross-encoder, feature scores only
    Stage 4 Pass B   → fallback: equal weights across features
    Stage 5 LLM      → fallback: structured assembly for all 100
    Validator        → always raises; bad output = disqualification
    """
    pipeline_start = time.perf_counter()

    log.info("=" * 60)
    log.info("  Redrob Intelligent Candidate Ranking System")
    log.info("  India Runs Hackathon — Track 1")
    log.info("=" * 60)

    # ── Input validation ──────────────────────────────────────────────────────
    if not jd_path.exists():
        log.error(f"JD file not found: {jd_path}")
        sys.exit(1)
    if not candidates_path.exists():
        log.error(f"Candidates file not found: {candidates_path}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    jd_object = None
    with stage_timer("Stage 1 — JD Intelligence", log):
        try:
            jd_object = _run_stage1(jd_path)
            log.info(
                f"JD parsed: "
                f"{len(jd_object.get('hard_requirements', []))} hard requirements, "
                f"{len(jd_object.get('soft_positives', []))} soft positives"
            )
        except Exception as exc:
            log_fallback(log, "Stage 1", str(exc))
            log.warning("Reduced ranking quality — falling back to minimal JD object")
            from src.stages.stage1_jd_intelligence import build_minimal_fallback
            jd_object = build_minimal_fallback(jd_path)

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    bm25_pool           = None
    valid_candidate_ids = set()
    with stage_timer("Stage 2 — Honeypot Gate + BM25 Retrieval", log):
        try:
            bm25_pool, valid_candidate_ids = _run_stage2(candidates_path, jd_object)
            log.info(f"BM25 pool: {len(bm25_pool):,} candidates")
        except Exception as exc:
            log.error(f"Stage 2 fatal: {exc}")
            raise

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    retrieval_pool = None
    with stage_timer("Stage 3 — Semantic Embedding + Hybrid Retrieval", log):
        try:
            query_vectors  = jd_object.get("query_vectors", {})
            retrieval_pool = _run_stage3(bm25_pool, query_vectors)
            log.info(f"Retrieval pool: {len(retrieval_pool):,} candidates")
        except Exception as exc:
            log.error(f"Stage 3 fatal: {exc}")
            raise

    # ── Stage 4 ───────────────────────────────────────────────────────────────
    top_100 = None
    with stage_timer("Stage 4 — Feature Extraction + Reranking", log):
        try:
            top_100 = _run_stage4(retrieval_pool, jd_object)
            log.info(f"Top {len(top_100)} candidates selected")
        except Exception as exc:
            log_fallback(log, "Stage 4", str(exc))
            log.warning("Falling back to RRF score order")
            top_100 = retrieval_pool[:100]

    # ── Stage 5 ───────────────────────────────────────────────────────────────
    output_file = None
    with stage_timer("Stage 5 — Reasoning + Output", log):
        try:
            output_file = _run_stage5(top_100, jd_object, output_path)
        except Exception as exc:
            log_fallback(log, "Stage 5 LLM", str(exc))
            log.warning("LLM failed — structured assembly for all 100")
            from src.stages.stage5_output import run_structured_only
            output_file = run_structured_only(top_100, jd_object, output_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    total   = time.perf_counter() - pipeline_start
    budget  = TIMING_TARGETS["budget"]
    status  = "✓ within budget" if total <= budget else "✗ OVER BUDGET"

    log.info("=" * 60)
    log.info(f"  Pipeline complete: {total:.1f}s / {budget}s  {status}")
    log.info(f"  Output: {output_file}")
    log.info("=" * 60)

    return output_file


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redrob Intelligent Candidate Ranking System",
    )
    parser.add_argument("--jd",         type=Path, default=JD_PATH)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    parser.add_argument("--output",     type=Path, default=OUTPUT_CSV_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_pipeline(
        jd_path=args.jd,
        candidates_path=args.candidates,
        output_path=args.output,
    )
    sys.exit(0 if result else 1)