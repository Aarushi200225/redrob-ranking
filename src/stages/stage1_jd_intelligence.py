"""
stages/stage1_jd_intelligence.py
─────────────────────────────────
Stage 1 — JD Intelligence.

Responsibilities:
  - Parse JD into structured object via Qwen
  - Generate HyDE hypothetical candidate profile
  - Expand hard requirements via skill taxonomy
  - Map culture signals to industry phrase clusters
  - Generate four query vectors (Q1-Q4) via bge-small

Runtime: ~30s
Output:  jd_object dict with query_vectors and vibe_cluster_vecs attached
"""

from pathlib import Path
import numpy as np

from src.config import (
    JD_QUERY_WEIGHT,
    HYDE_QUERY_WEIGHT,
    SKILL_TAXONOMY_PATH,
    VIBE_KEYWORDS_PATH,
)
from src.utils.data_loader import load_text_file, load_json_artifact
from src.utils.logger import get_logger
# from src.models.model_context import ModelContext
# from src.models.llm import load_llm, parse_jd, generate_hyde_profile
# from src.models.embedder import load_embedder, encode_single

log = get_logger(__name__)


def _expand_requirements(
    hard_requirements: list[str],
    taxonomy: dict,
) -> list[str]:
    """
    Expand hard requirements using skill taxonomy.
    Returns flat list of all original + expanded terms.
    """
    expanded = list(hard_requirements)
    for req in hard_requirements:
        req_lower = req.lower()
        for category, terms in taxonomy.items():
            if req_lower in category.lower() or any(
                req_lower in t.lower() for t in terms
            ):
                expanded.extend(terms)
    # Deduplicate preserving order
    seen = set()
    result = []
    for term in expanded:
        if term.lower() not in seen:
            seen.add(term.lower())
            result.append(term)
    return result


def _build_vibe_cluster_text(vibe_keywords: dict) -> dict:
    """
    Build concatenated text per vibe signal for embedding.
    Returns {signal_name: joined_phrase_string}.
    """
    return {
        signal: " ".join(phrases)
        for signal, phrases in vibe_keywords.items()
    }


def build_minimal_fallback(jd_path: Path) -> dict:
    """
    Minimal JD object built from raw text when Qwen parsing fails.
    Imported by pipeline.py for Stage 1 exception fallback.
    """
    from src.models.llm import _minimal_jd_fallback
    jd_text = load_text_file(jd_path)
    obj = _minimal_jd_fallback()
    obj["raw_text"]       = jd_text
    obj["query_vectors"]  = {}
    obj["vibe_cluster_vecs"] = {}
    obj["expanded_requirements"] = obj["hard_requirements"]
    return obj


def run(jd_path: Path) -> dict:
    """
    Execute Stage 1 — JD Intelligence.

    Parameters
    ----------
    jd_path : Path to job_description.txt

    Returns
    -------
    dict — fully populated JD object including:
      - All parsed fields (hard_requirements, disqualifiers, etc.)
      - query_vectors: {Q1, Q2, Q3, Q4} as np.ndarray
      - vibe_cluster_vecs: {signal_name: np.ndarray}
      - expanded_requirements: taxonomy-expanded hard requirements
    """
    # ── Load inputs ───────────────────────────────────────────────────────────
    from src.models.model_context import ModelContext
    from src.models.llm import load_llm, parse_jd, generate_hyde_profile
    from src.models.embedder import load_embedder, encode_single
    jd_text  = load_text_file(jd_path)
    taxonomy = load_json_artifact(SKILL_TAXONOMY_PATH)
    vibe_kw  = load_json_artifact(VIBE_KEYWORDS_PATH)

    log.info(f"JD loaded: {len(jd_text)} chars")

    # ── Qwen: JD parsing + HyDE ───────────────────────────────────────────────
    jd_object   = {}
    hyde_profile = ""

    with ModelContext(load_llm) as llm:
        jd_object    = parse_jd(llm, jd_text)
        hyde_profile = generate_hyde_profile(llm, jd_text)

    jd_object["raw_text"] = jd_text

    # ── Taxonomy expansion ────────────────────────────────────────────────────
    expanded = _expand_requirements(
        jd_object.get("hard_requirements", []),
        taxonomy,
    )
    jd_object["expanded_requirements"] = expanded
    log.info(
        f"Requirements expanded: "
        f"{len(jd_object.get('hard_requirements', []))} → {len(expanded)}"
    )

    # ── Vibe cluster text ─────────────────────────────────────────────────────
    vibe_texts = _build_vibe_cluster_text(vibe_kw)

    # ── bge-small: query vectors + vibe embeddings ────────────────────────────
    query_vectors    = {}
    vibe_cluster_vecs = {}

    with ModelContext(load_embedder) as embedder:
        # Primary query vectors
        jd_vec   = encode_single(embedder, jd_text[:2000])
        hyde_vec = encode_single(embedder, hyde_profile)

        # Q1: HyDE-weighted primary
        q1 = JD_QUERY_WEIGHT * jd_vec + HYDE_QUERY_WEIGHT * hyde_vec
        q1 = q1 / (np.linalg.norm(q1) + 1e-8)  # Renormalise after blend

        # Q2-Q4: section-specific vectors
        tech_section = " ".join(expanded[:30])
        exp_section  = (
            f"experience {jd_object.get('experience_band', {})} "
            f"years production deployment AI ML"
        )
        vibe_section = " ".join([
            p for phrases in vibe_kw.values()
            for p in phrases
        ][:40])

        query_vectors = {
            "Q1": q1,
            "Q2": encode_single(embedder, tech_section),
            "Q3": encode_single(embedder, exp_section),
            "Q4": encode_single(embedder, vibe_section),
        }

        # Vibe cluster embeddings for F4 scoring
        for signal_name, text in vibe_texts.items():
            vibe_cluster_vecs[signal_name] = encode_single(
                embedder, text
            )

    jd_object["query_vectors"]    = query_vectors
    jd_object["vibe_cluster_vecs"] = vibe_cluster_vecs
    jd_object["hyde_profile"]     = hyde_profile

    log.info(
        f"Stage 1 complete — "
        f"{len(query_vectors)} query vectors, "
        f"{len(vibe_cluster_vecs)} vibe clusters"
    )

    return jd_object