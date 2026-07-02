"""
utils/data_loader.py
────────────────────
Streaming data loader for the candidate JSONL pool.

Handles both .jsonl and .jsonl.gz formats transparently.
Streams candidates in chunks to avoid loading 465MB into memory at once.
Provides a fast full-load path for the BM25 tokenisation stage.

All file I/O in the pipeline goes through this module —
stage modules never open files directly.
"""

import gzip
import orjson
from pathlib import Path
from typing import Generator

from src.utils.logger import get_logger

log = get_logger(__name__)


def _open_jsonl(path: Path):
    """Return the correct file handle for .jsonl or .jsonl.gz."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def stream_candidates(
    path: Path,
    chunk_size: int = 5000,
) -> Generator[list[dict], None, None]:
    """
    Stream candidates from JSONL file in chunks.

    Yields lists of candidate dicts of length <= chunk_size.
    Memory-efficient — only one chunk in memory at a time.

    Parameters
    ----------
    path       : Path to .jsonl or .jsonl.gz file.
    chunk_size : Number of candidates per yielded chunk.

    Yields
    ------
    list[dict] — chunk of parsed candidate records.
    """
    chunk = []
    total = 0

    with _open_jsonl(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                candidate = orjson.loads(line)
                chunk.append(candidate)
                total += 1
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
            except orjson.JSONDecodeError as e:
                log.warning(f"Skipping malformed line {total}: {e}")

    if chunk:
        yield chunk

    log.info(f"Streamed {total:,} candidates from {path.name}")


def load_all_candidates(path: Path) -> list[dict]:
    """
    Load all candidates into memory at once.

    Handles two formats transparently:
      1. JSONL (.jsonl) — one JSON object per line (production format)
      2. JSON array (.json) — pretty-printed array (sample/test format)
      3. Gzipped JSONL (.jsonl.gz)

    Parameters
    ----------
    path : Path to candidates file.

    Returns
    -------
    list[dict] — all parsed candidate records.
    """
    log.info(f"Loading all candidates from {path.name} ...")

    # Peek at first non-whitespace character to detect format
    with _open_jsonl(path) as f:
        first_char = ""
        while not first_char.strip():
            byte = f.read(1)
            if not byte:
                break
            first_char = byte.decode("utf-8") if isinstance(byte, bytes) else byte

    if first_char == "[":
        # JSON array format — load and parse entire file at once
        with _open_jsonl(path) as f:
            raw = f.read()
        candidates = orjson.loads(raw)
        log.info(f"Loaded {len(candidates):,} candidates (JSON array format)")
        return candidates

    # JSONL format — parse line by line
    candidates = []
    with _open_jsonl(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(orjson.loads(line))
            except orjson.JSONDecodeError as e:
                log.warning(f"Skipping malformed line: {e}")

    log.info(f"Loaded {len(candidates):,} candidates")
    return candidates


def load_text_file(path: Path) -> str:
    """
    Load a plain text file (e.g. job_description.txt).

    Parameters
    ----------
    path : Path to text file.

    Returns
    -------
    str — file contents stripped of leading/trailing whitespace.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_json_artifact(path: Path) -> dict | list:
    """
    Load a static JSON artifact (skill_taxonomy, vibe_keywords).

    Parameters
    ----------
    path : Path to .json file.

    Returns
    -------
    dict or list — parsed JSON content.
    """
    with open(path, "rb") as f:
        return orjson.loads(f.read())