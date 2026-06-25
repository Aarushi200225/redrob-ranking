"""
stages/stage5_output.py
────────────────────────
Stage 5 — Reasoning Generation + Output.

Responsibilities:
  - Ranks 1-40: Qwen LLM reasoning (sequential, Pydantic guardrail)
  - Ranks 41-100: dynamic structured assembly
  - Format validation (hard assertions)
  - CSV write

Runtime: ~35s
Output:  Path to ranked_output.csv
"""

from pathlib import Path
import pandas as pd

from src.config import (
    LLM_REASONING_TOP_N,
    OUTPUT_CSV_PATH,
    LLM_N_CTX_REASONING
)
from src.utils.logger import get_logger
from src.utils.validator import validate_output
# from src.models.model_context import ModelContext
# from src.models.llm import load_llm, generate_reasoning
from src.reasoning.structured_assembly import build_structured_reasoning
from functools import partial

log = get_logger(__name__)


def run(
    top_100: list[dict],
    jd_object: dict,
    output_path: Path = OUTPUT_CSV_PATH,
) -> Path:
    """
    Execute Stage 5 — Reasoning + Output.

    Parameters
    ----------
    top_100     : Top 100 ranked candidates from Stage 4.
    jd_object   : Parsed JD object from Stage 1.
    output_path : Destination for ranked_output.csv.

    Returns
    -------
    Path to written CSV file.
    """
    from src.models.model_context import ModelContext
    from src.models.llm import load_llm, generate_reasoning
    reasonings = []

    # ── Ranks 1-40: Qwen LLM reasoning ───────────────────────────────────────
    llm_candidates  = top_100[:LLM_REASONING_TOP_N]
    rest_candidates = top_100[LLM_REASONING_TOP_N:]

    log.info(
        f"Generating LLM reasoning for ranks 1-{LLM_REASONING_TOP_N} ..."
    )

    with ModelContext(load_llm,  LLM_N_CTX_REASONING) as llm:
        for candidate in llm_candidates:
            score_breakdown = candidate.get("_score_breakdown", {})
            reasoning = generate_reasoning(
                llm, candidate, jd_object, score_breakdown
            )
            reasonings.append(reasoning)

    # ── Ranks 41-100: structured assembly ────────────────────────────────────
    log.info(
        f"Generating structured reasoning for "
        f"ranks {LLM_REASONING_TOP_N + 1}-100 ..."
    )

    for candidate in rest_candidates:
        reasoning = build_structured_reasoning(candidate, jd_object)
        reasonings.append(reasoning)

    # ── Build output DataFrame ────────────────────────────────────────────────
    valid_ids = set()  # Populated from pipeline — passed via jd_object
    rows = []
    for rank_idx, (candidate, reasoning) in enumerate(
        zip(top_100, reasonings), start=1
    ):
        rows.append({
            "candidate_id": candidate["candidate_id"],
            "rank":         rank_idx,
            "score":        round(float(candidate.get("_score", 0.0)), 6),
            "reasoning":    reasoning,
        })

    df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])

    # ── Format validation ─────────────────────────────────────────────────────
    # Load valid IDs for validation
    from src.utils.data_loader import load_all_candidates
    from src.config import CANDIDATES_PATH

    try:
        all_candidates = load_all_candidates(CANDIDATES_PATH)
        valid_ids = {c["candidate_id"] for c in all_candidates}
    except Exception:
        valid_ids = {c["candidate_id"] for c in top_100}
        log.warning("Could not load full candidate pool for ID validation")

    validate_output(df, valid_ids)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info(f"Output written: {output_path}")

    return output_path


def run_structured_only(
    top_100: list[dict],
    jd_object: dict,
    output_path: Path = OUTPUT_CSV_PATH,
) -> Path:
    """
    Fallback — structured assembly for all 100 candidates.
    Called when LLM reasoning fails entirely.
    """
    log.info("Running structured assembly for all 100 candidates ...")

    rows = []
    for rank_idx, candidate in enumerate(top_100, start=1):
        reasoning = build_structured_reasoning(candidate, jd_object)
        rows.append({
            "candidate_id": candidate["candidate_id"],
            "rank":         rank_idx,
            "score":        round(float(candidate.get("_score", 0.0)), 6),
            "reasoning":    reasoning,
        })

    df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])

    from src.utils.data_loader import load_all_candidates
    from src.config import CANDIDATES_PATH
    try:
        all_candidates = load_all_candidates(CANDIDATES_PATH)
        valid_ids = {c["candidate_id"] for c in all_candidates}
    except Exception:
        valid_ids = {c["candidate_id"] for c in top_100}

    validate_output(df, valid_ids)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info(f"Fallback output written: {output_path}")

    return output_path