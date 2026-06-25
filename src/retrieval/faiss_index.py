"""
retrieval/faiss_index.py
─────────────────────────
FAISS IndexFlatIP — exact cosine similarity search.

At 15K vectors of 384 dimensions, exact search is fast enough
(~1s) and preferable to approximate methods (HNSW, IVF) which
add index-building overhead for negligible speed benefit at this scale.

Vectors must be L2-normalised before indexing — inner product
on normalised vectors equals cosine similarity.
"""

import numpy as np
import faiss

from src.utils.logger import get_logger

log = get_logger(__name__)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build FAISS IndexFlatIP from normalised embedding matrix.

    Parameters
    ----------
    embeddings : np.ndarray of shape [n, dim], L2-normalised, float32.

    Returns
    -------
    faiss.IndexFlatIP — exact inner product index.
    """
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    n, dim = embeddings.shape
    log.info(f"Building FAISS IndexFlatIP — {n:,} vectors, dim={dim}")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    log.info(f"FAISS index built — {index.ntotal:,} vectors indexed")
    return index


def multi_query_search(
    index: faiss.IndexFlatIP,
    query_vectors: dict,
    top_k: int,
) -> dict:
    """
    Run multiple query vectors against the FAISS index.

    Each query produces an independent ranked list —
    required for correct 6-stream RRF fusion.

    Parameters
    ----------
    index         : Built FAISS index.
    query_vectors : Dict of {stream_name: np.ndarray query vector}.
    top_k         : Candidates to retrieve per query.

    Returns
    -------
    dict of {stream_name: [ordered candidate indices]}
    """
    results = {}

    for stream_name, query_vec in query_vectors.items():
        # Ensure correct shape and dtype
        q = np.array(query_vec, dtype=np.float32).reshape(1, -1)

        # Renormalise — safety measure for blended vectors
        norm = np.linalg.norm(q)
        if norm > 1e-8:
            q = q / norm

        distances, indices = index.search(q, top_k)

        # Filter out -1 indices (FAISS returns -1 for unfilled slots)
        valid_indices = [
            int(idx) for idx in indices[0] if idx >= 0
        ]
        results[stream_name] = valid_indices

        log.info(
            f"FAISS {stream_name}: "
            f"{len(valid_indices)} results, "
            f"top score: {distances[0][0]:.4f}"
        )

    return results