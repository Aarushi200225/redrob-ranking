"""
utils/logger.py
───────────────
Structured logger for the ranking pipeline.

Every stage logs:
  - Start event with stage name
  - Candidate count in → count out
  - Wall-clock duration
  - Warnings or fallback activations
"""

import logging
import sys
import time
from contextlib import contextmanager
from typing import Optional


class PipelineFormatter(logging.Formatter):
    """Colour-coded formatter for terminal readability during dev runs."""

    LEVEL_COLORS = {
        "DEBUG":    "\033[36m",
        "INFO":     "\033[32m",
        "WARNING":  "\033[33m",
        "ERROR":    "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelname, "")
        level = f"{color}{record.levelname:<8}{self.RESET}"
        name  = f"{self.BOLD}{record.name}{self.RESET}"
        return f"{level} [{name}] {record.getMessage()}"


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Usage:
        from src.utils.logger import get_logger
        log = get_logger(__name__)
        log.info("Stage 3 started")
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(PipelineFormatter())
    logger.addHandler(handler)
    logger.propagate = False

    return logger


@contextmanager
def stage_timer(stage_name: str, logger: Optional[logging.Logger] = None):
    """
    Context manager — logs start, end, and wall-clock duration for a stage.

    Usage:
        with stage_timer("Stage 3 — Embedding", log):
            embeddings = embedder.encode(...)
    """
    _log = logger or get_logger("pipeline.timer")
    _log.info(f"{'─' * 60}")
    _log.info(f"▶  {stage_name} started")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        _log.info(f"✓  {stage_name} completed in {elapsed:.1f}s")
        _log.info(f"{'─' * 60}")


def log_pool_transition(
    logger: logging.Logger,
    stage: str,
    count_in: int,
    count_out: int,
    note: str = "",
) -> None:
    """Log candidate pool size transitions between stages."""
    note_str = f"  [{note}]" if note else ""
    logger.info(f"Pool: {count_in:>8,} → {count_out:>8,}{note_str}")


def log_fallback(logger: logging.Logger, stage: str, reason: str) -> None:
    """Log when a stage activates its degraded-mode fallback."""
    logger.warning(f"FALLBACK activated in {stage}: {reason}")