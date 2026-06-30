"""
models/embedder.py
──────────────────
Wrapper for bge-small-en-v1.5 embedding model.

Handles:
  - Model loading with sentence-transformers
  - Batched encoding with normalisation (required for cosine via IP)
  - Single text encoding for query vectors
  - Input validation and logging
"""

import numpy as np
from pathlib import Path
# from sentence_transformers import SentenceTransformer

from src.config import EMBEDDER_MODEL_ID, EMBED_BATCH_SIZE, EMBED_NORMALIZE
from src.utils.logger import get_logger

log = get_logger(__name__)


def load_embedder():
    """
    Load bge-small-en-v1.5 from local sentence-transformers cache.

    Returns
    -------
    SentenceTransformer model ready for encoding.
    """
    from sentence_transformers import SentenceTransformer
    log.info(f"Loading embedder: {EMBEDDER_MODEL_ID}")
    model = SentenceTransformer(EMBEDDER_MODEL_ID)
    log.info(
        f"Embedder loaded — "
        f"embedding dim: {model.get_embedding_dimension()}"
    )
    return model


def batch_encode(
    model,
    texts: list[str],
    batch_size: int = EMBED_BATCH_SIZE,
    normalize: bool = EMBED_NORMALIZE,
) -> np.ndarray:
    """
    Encode a list of texts into normalised embedding vectors.

    Parameters
    ----------
    model      : Loaded SentenceTransformer model.
    texts      : List of text strings to encode.
    batch_size : Number of texts per encoding batch.
    normalize  : Whether to L2-normalise output vectors.
                 Must be True for cosine similarity via inner product.

    Returns
    -------
    np.ndarray of shape [len(texts), embedding_dim], dtype float32.
    """
    if not texts:
        raise ValueError("texts list is empty — nothing to encode")

    log.info(f"Encoding {len(texts):,} texts (batch_size={batch_size}) ...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 1000,
    )

    log.info(
        f"Encoding complete — "
        f"shape: {embeddings.shape}, dtype: {embeddings.dtype}"
    )
    return embeddings


def encode_single(
    model,
    text: str,
    normalize: bool = EMBED_NORMALIZE,
) -> np.ndarray:
    """
    Encode a single text string into a normalised embedding vector.

    Used for query vector generation in Stage 1.

    Parameters
    ----------
    model     : Loaded SentenceTransformer model.
    text      : Text string to encode.
    normalize : Whether to L2-normalise output vector.

    Returns
    -------
    np.ndarray of shape [embedding_dim], dtype float32.
    """
    vec = model.encode(
        [text],
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return vec[0]


def build_candidate_text_blob(candidate: dict) -> str:
    """
    Construct a single text blob per candidate for embedding.

    Concatenates highest-signal text fields:
      - headline
      - summary
      - top 3 career role descriptions
      - top 10 skill names (sorted by proficiency + duration)

    Parameters
    ----------
    candidate : Raw candidate dict from JSONL.

    Returns
    -------
    str — concatenated text blob for embedding.
    """
    profile = candidate.get("profile", {})
    parts   = []

    # Headline and summary
    headline = profile.get("headline", "").strip()
    summary  = profile.get("summary", "").strip()
    if headline:
        parts.append(headline)
    if summary:
        parts.append(summary)

    # Top 3 career role descriptions
    career = candidate.get("career_history", [])
    for role in career[:3]:
        desc = role.get("description", "").strip()
        if desc:
            parts.append(desc)

    # Top 10 skills sorted by proficiency depth then duration
    proficiency_order = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    skills = candidate.get("skills", [])
    sorted_skills = sorted(
        skills,
        key=lambda s: (
            proficiency_order.get(s.get("proficiency", "beginner"), 0),
            s.get("duration_months", 0),
        ),
        reverse=True,
    )
    skill_names = [s["name"] for s in sorted_skills[:10] if s.get("name")]
    if skill_names:
        parts.append(", ".join(skill_names))

    return " | ".join(parts)