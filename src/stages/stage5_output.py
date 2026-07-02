"""
stages/stage5_output.py
────────────────────────
Stage 5 — Reasoning Generation + Output.

Pipeline:
  1. Ranks 1-40: Qwen LLM reasoning (n_ctx=512, n_batch=32)
     Per-candidate try/except — one failure never cascades.
     Fallback: build_structured_reasoning() per candidate.
  2. Ranks 41-100: dynamic structured assembly
  3. Format validation (hard assertions)
  4. CSV write

Design note on Qwen reasoning:
  Qwen 2.5-0.5B generates grounded, candidate-specific reasoning
  text but struggles with strict JSON formatting constraints.
  The per-candidate fallback to structured assembly ensures 100%
  output coverage while preserving LLM-generated reasoning for
  candidates where Qwen succeeds. Production upgrade path:
  Phi-3-mini-4k-instruct (3.8B) produces reliable JSON output.

Runtime: ~35s (submission) | ~5s (dev, structured only)
"""

from pathlib import Path
import pandas as pd

from src.config import (
    LLM_REASONING_TOP_N,
    OUTPUT_CSV_PATH,
    LLM_N_CTX_REASONING,
    LLM_N_BATCH_REASONING,
)
from src.utils.logger import get_logger, memory_gate
from src.utils.validator import validate_output
from src.reasoning.structured_assembly import build_structured_reasoning

log = get_logger(__name__)


def run(
    top_100: list[dict],
    jd_object: dict,
    output_path: Path = OUTPUT_CSV_PATH,
    valid_ids: set = None,
) -> Path:
    """
    Execute Stage 5 — Reasoning + Output.

    Parameters
    ----------
    top_100     : Top 100 ranked candidates from Stage 4.
    jd_object   : Parsed JD object from Stage 1.
    output_path : Destination for ranked_output.csv.
    valid_ids   : Set of valid candidate_ids for format validation.

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

    try:
        with ModelContext(
            load_llm, LLM_N_CTX_REASONING, LLM_N_BATCH_REASONING
        ) as llm:
            for candidate in llm_candidates:
                score_breakdown = candidate.get("_score_breakdown", {})
                # Per-candidate fallback — one Qwen failure never cascades
                try:
                    reasoning = generate_reasoning(
                        llm, candidate, jd_object, score_breakdown
                    )
                except Exception as e:
                    log.debug(
                        f"LLM reasoning failed for "
                        f"{candidate.get('candidate_id')}: {e}"
                    )
                    reasoning = build_structured_reasoning(
                        candidate, jd_object
                    )
                reasonings.append(reasoning)

    except Exception as e:
        log.warning(
            f"Qwen unavailable ({e}) — "
            f"structured assembly for ranks 1-{LLM_REASONING_TOP_N}"
        )
        reasonings = [
            build_structured_reasoning(c, jd_object)
            for c in llm_candidates
        ]

    memory_gate("Stage 5 Qwen", log)

    # ── Ranks 41-100: dynamic structured assembly ─────────────────────────────
    log.info(
        f"Generating structured reasoning for "
        f"ranks {LLM_REASONING_TOP_N + 1}-100 ..."
    )
    for candidate in rest_candidates:
        reasoning = build_structured_reasoning(candidate, jd_object)
        reasonings.append(reasoning)

    # ── Build output DataFrame ────────────────────────────────────────────────
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

    df = pd.DataFrame(
        rows,
        columns=["candidate_id", "rank", "score", "reasoning"]
    )

    # ── Format validation ─────────────────────────────────────────────────────
    if valid_ids is None:
        from src.utils.data_loader import load_all_candidates
        from src.config import CANDIDATES_PATH
        all_candidates = load_all_candidates(CANDIDATES_PATH)
        valid_ids = {c["candidate_id"] for c in all_candidates}

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
    valid_ids: set = None,
) -> Path:
    """
    Fallback — structured assembly for all 100 candidates.
    Called when Qwen fails entirely at the stage level.
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

    df = pd.DataFrame(
        rows,
        columns=["candidate_id", "rank", "score", "reasoning"]
    )

    if valid_ids is None:
        from src.utils.data_loader import load_all_candidates
        from src.config import CANDIDATES_PATH
        all_candidates = load_all_candidates(CANDIDATES_PATH)
        valid_ids = {c["candidate_id"] for c in all_candidates}

    validate_output(df, valid_ids)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info(f"Fallback output written: {output_path}")

    return output_path
