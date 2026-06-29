"""
stages/stage1_jd_intelligence.py
─────────────────────────────────
Stage 1 — JD Intelligence.

Pipeline:
  1. Qwen parses JD → structured object (CoT + grammar constraints)
  2. Qwen generates HyDE profile
  3. Skill taxonomy expands hard requirements
  4. Vibe keywords map culture signals to phrase clusters
  5. bge-small generates 4 query vectors + vibe cluster embeddings

All model imports are lazy (inside functions) to prevent
module-level loading of heavy dependencies.

Runtime: ~25s | Memory peak: ~2GB (models released between steps)
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
from src.utils.logger import get_logger, memory_gate

log = get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _expand_requirements(hard_requirements: list[str], taxonomy: dict) -> list[str]:
    """Expand hard requirements using skill taxonomy."""
    expanded = list(hard_requirements)
    for req in hard_requirements:
        req_lower = req.lower()
        for category, terms in taxonomy.items():
            if req_lower in category.lower() or any(
                req_lower in t.lower() for t in terms
            ):
                expanded.extend(terms)

    # Deduplicate preserving order
    seen, result = set(), []
    for term in expanded:
        if term.lower() not in seen:
            seen.add(term.lower())
            result.append(term)
    return result


def _build_vibe_cluster_texts(vibe_keywords: dict) -> dict:
    """Concatenate vibe phrases per signal for embedding."""
    return {
        signal: " ".join(phrases)
        for signal, phrases in vibe_keywords.items()
    }


def build_minimal_fallback(jd_path: Path) -> dict:
    """
    Minimal JD object for Stage 1 complete failure.
    Imported by pipeline.py exception handler.
    """
    from src.models.llm import _minimal_jd_fallback
    jd_text = load_text_file(jd_path)
    obj = _minimal_jd_fallback()
    obj["raw_text"]              = jd_text
    obj["query_vectors"]         = {}
    obj["vibe_cluster_vecs"]     = {}
    obj["expanded_requirements"] = obj["hard_requirements"]
    obj["hyde_profile"]          = ""
    return obj


# ── Main stage ────────────────────────────────────────────────────────────────

def run(jd_path: Path) -> dict:
    """
    Execute Stage 1 — JD Intelligence.

    Returns
    -------
    dict — JD object with all fields populated including:
      query_vectors     : {Q1, Q2, Q3, Q4} np.ndarray
      vibe_cluster_vecs : {signal_name: np.ndarray}
      expanded_requirements : list[str]
      raw_text          : str
      hyde_profile      : str
    """
    # Lazy imports — prevent module-level model loading
    from src.models.model_context import ModelContext
    from src.models.llm import (
        load_llm, parse_jd, generate_hyde_profile, _minimal_jd_fallback
    )
    from src.models.embedder import load_embedder, encode_single
    from src.config import LLM_N_CTX, LLM_N_BATCH

    # ── Load inputs ───────────────────────────────────────────────────────────
    jd_text  = load_text_file(jd_path)
    taxonomy = load_json_artifact(SKILL_TAXONOMY_PATH)
    vibe_kw  = load_json_artifact(VIBE_KEYWORDS_PATH)
    log.info(f"JD loaded: {len(jd_text)} chars")

    # ── Qwen: JD parsing + HyDE ───────────────────────────────────────────────
    jd_object    = {}
    hyde_profile = ""

    with ModelContext(load_llm, LLM_N_CTX, LLM_N_BATCH) as llm:
        jd_object    = parse_jd(llm, jd_text)
        hyde_profile = generate_hyde_profile(llm, jd_text)

    jd_object["raw_text"]    = jd_text
    jd_object["hyde_profile"] = hyde_profile
    memory_gate("Stage 1 Qwen", log)

    # ── Taxonomy expansion ────────────────────────────────────────────────────
    expanded = _expand_requirements(
        jd_object.get("hard_requirements", []), taxonomy
    )
    jd_object["expanded_requirements"] = expanded
    log.info(
        f"Requirements expanded: "
        f"{len(jd_object.get('hard_requirements', []))} → {len(expanded)}"
    )

    # ── Vibe cluster texts ────────────────────────────────────────────────────
    vibe_texts = _build_vibe_cluster_texts(vibe_kw)

    # ── bge-small: query vectors + vibe embeddings ────────────────────────────
    query_vectors     = {}
    vibe_cluster_vecs = {}

    with ModelContext(load_embedder) as embedder:
        # Primary query: HyDE-weighted blend
        jd_vec   = encode_single(embedder, jd_text[:2000])
        hyde_vec = encode_single(embedder, hyde_profile) if hyde_profile else jd_vec

        q1 = JD_QUERY_WEIGHT * jd_vec + HYDE_QUERY_WEIGHT * hyde_vec
        norm = np.linalg.norm(q1)
        q1   = q1 / (norm + 1e-8)   # Renormalise after blend

        # Section-specific query vectors
        tech_text  = " ".join(expanded[:30])
        exp_text   = (
            f"production ML experience {jd_object.get('experience_band', {})} "
            f"years deployed systems AI engineer product company"
        )
        vibe_text  = " ".join([
            p for phrases in vibe_kw.values() for p in phrases
        ][:40])

        query_vectors = {
            "Q1": q1,
            "Q2": encode_single(embedder, tech_text),
            "Q3": encode_single(embedder, exp_text),
            "Q4": encode_single(embedder, vibe_text),
        }

        # Vibe cluster embeddings for F4 scoring
        for signal_name, text in vibe_texts.items():
            vibe_cluster_vecs[signal_name] = encode_single(embedder, text)

    memory_gate("Stage 1 embedder", log)

    jd_object["query_vectors"]     = query_vectors
    jd_object["vibe_cluster_vecs"] = vibe_cluster_vecs

    log.info(
        f"Stage 1 complete — "
        f"{len(query_vectors)} query vectors, "
        f"{len(vibe_cluster_vecs)} vibe clusters"
    )

    return jd_object