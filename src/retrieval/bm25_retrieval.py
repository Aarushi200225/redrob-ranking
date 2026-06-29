"""
retrieval/bm25_retrieval.py
────────────────────────────
Dual-chamber BM25 retrieval.

Chamber A: raw JD terms — captures general semantic intent
Chamber B: expanded taxonomy terms — captures structural skill alignment

Chambers kept separate to prevent vocabulary inflation from
combined query execution (BM25 length normalisation artifact).

Parallelism note:
  Windows uses spawn-based multiprocessing which requires pickling
  entire chunks across process boundaries. At 100K candidates this
  causes MemoryError regardless of chunk size. Sequential tokenisation
  is used instead — takes ~8-10s and is fully reliable on all platforms.
"""

import re
import numpy as np
from rank_bm25 import BM25Okapi

from src.utils.logger import get_logger

log = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, split into tokens."""
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    STOP_WORDS = {
        'the', 'and', 'a', 'of', 'to', 'in', 'is', 'with', 'for',
        'on', 'at', 'by', 'an', 'from', 'as', 'about', 'that', 'this',
        'are', 'be', 'or', 'it', 'was', 'your', 'our', 'their', 'will',
        'can', 'not', 'which', 'who', 'into', 'have', 'has', 'had',
        'been', 'but', 'more', 'also', 'we', 'you', 'my', 'they',
        'he', 'she', 'its', 'me', 'him', 'her', 'us', 'do', 'did',
        'work', 'worked', 'working', 'experience', 'year', 'years'
    }
    return [t for t in text.split() if t not in STOP_WORDS and len(t) > 2]


def _extract_candidate_text(candidate: dict) -> str:
    """
    Extract searchable text from candidate dict.
    Concatenates headline, summary, career descriptions, skill names.
    """
    profile = candidate.get("profile", {})
    parts   = []

    headline = profile.get("headline", "")
    summary  = profile.get("summary", "")
    if headline:
        parts.append(headline)
    if summary:
        parts.append(summary)

    for role in candidate.get("career_history", [])[:5]:
        desc = role.get("description", "")
        if desc:
            parts.append(desc)

    skill_names = " ".join(
        s.get("name", "") for s in candidate.get("skills", [])
    )
    if skill_names:
        parts.append(skill_names)

    return " ".join(parts)


def build_bm25_index(
    candidates: list[dict],
) -> tuple[list[list[str]], BM25Okapi]:
    """
    Build BM25Okapi index over the candidate corpus.

    Memory-optimised: extracts text and tokenises in a single pass,
    discarding intermediate strings immediately to reduce peak RAM.

    Parameters
    ----------
    candidates : Clean candidate list (post-honeypot gate).

    Returns
    -------
    tuple:
      corpus : list[list[str]] — tokenised documents
      index  : BM25Okapi index
    """
    n = len(candidates)
    log.info(f"Extracting and tokenising {n:,} candidates ...")

    corpus = []
    for i, candidate in enumerate(candidates):
        # Extract text and tokenise immediately — don't store raw text
        text   = _extract_candidate_text(candidate)
        tokens = _tokenize(text)
        corpus.append(tokens)
        
        # Explicit progress + periodic GC to keep memory flat
        if (i + 1) % 10000 == 0:
            log.info(f"  Tokenised {i + 1:,} / {n:,} ...")
            import gc; gc.collect() # Force garbage collection to free memory

    log.info(f"Building BM25 index over {len(corpus):,} documents ...")
    index = BM25Okapi(corpus)
    log.info("BM25 index built")

    return corpus, index


def retrieve_chamber_a(
    index: BM25Okapi,
    corpus: list[list[str]],
    candidates: list[dict],
    jd_text: str,
    top_k: int,
) -> list[dict]:
    """Chamber A — retrieve using raw JD terms only."""
    query   = _tokenize(jd_text)
    scores  = index.get_scores(query)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [candidates[i] for i in top_idx]


def retrieve_chamber_b(
    index: BM25Okapi,
    corpus: list[list[str]],
    candidates: list[dict],
    expanded_terms: list[str],
    top_k: int,
) -> list[dict]:
    """Chamber B — retrieve using expanded taxonomy terms only."""
    query   = _tokenize(" ".join(expanded_terms))
    scores  = index.get_scores(query)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [candidates[i] for i in top_idx]


def union_chambers(
    top_a: list[dict],
    top_b: list[dict],
    cap: int,
) -> list[dict]:
    """Union two chamber results, deduplicated by candidate_id."""
    seen  = set()
    union = []

    for candidate in top_a + top_b:
        cid = candidate.get("candidate_id")
        if cid not in seen:
            seen.add(cid)
            union.append(candidate)
        if len(union) >= cap:
            break

    return union