"""
utils/logger.py
───────────────
Structured logger with memory gates for pipeline monitoring.

memory_gate() logs RSS memory between stages — critical for
verifying we stay within the 16GB sandbox budget and for
diagnosing timing issues during development.
"""

import gc
import logging
import sys
import time
import ctypes
from contextlib import contextmanager
from typing import Optional


# ── Formatter ─────────────────────────────────────────────────────────────────

class PipelineFormatter(logging.Formatter):
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


# ── Factory ───────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
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


# ── Stage timer ───────────────────────────────────────────────────────────────

@contextmanager
def stage_timer(stage_name: str, logger: Optional[logging.Logger] = None):
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


# ── Memory gate ───────────────────────────────────────────────────────────────

def memory_gate(stage_name: str, logger: Optional[logging.Logger] = None) -> float:
    """
    Force full GC, return OS memory, and log current RSS.

    Called between every pipeline stage to:
      1. Release memory held by Python allocator
      2. Return freed pages to OS (Linux only via malloc_trim)
      3. Log current process RSS for budget monitoring

    Parameters
    ----------
    stage_name : Label for the log entry.
    logger     : Logger instance. Uses default if None.

    Returns
    -------
    float — current RSS in MB.
    """
    _log = logger or get_logger("pipeline.memory")

    # Three GC passes — handles simple objects, cyclic refs, and finalizers
    gc.collect()
    gc.collect()
    gc.collect()

    # Force OS memory return on Linux (no-op on Windows/Mac)
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass

    # Log RSS if psutil available
    try:
        import psutil
        import os
        process  = psutil.Process(os.getpid())
        rss_mb   = process.memory_info().rss / 1024 / 1024
        _log.info(f"[MEM] After {stage_name}: {rss_mb:.0f} MB RSS")
        return rss_mb
    except ImportError:
        _log.debug(f"[MEM] psutil not available — install for memory monitoring")
        return 0.0


# ── Pool transition ───────────────────────────────────────────────────────────

def log_pool_transition(
    logger: logging.Logger,
    stage: str,
    count_in: int,
    count_out: int,
    note: str = "",
) -> None:
    note_str = f"  [{note}]" if note else ""
    logger.info(f"Pool: {count_in:>8,} → {count_out:>8,}{note_str}")


# ── Fallback warning ──────────────────────────────────────────────────────────

def log_fallback(logger: logging.Logger, stage: str, reason: str) -> None:
    logger.warning(f"FALLBACK activated in {stage}: {reason}")