"""
models/model_context.py
───────────────────────
ModelContext — sequential model lifecycle manager.

Enforces strict load/release pattern across the pipeline:
  - One model in memory at a time
  - Explicit GC between model loads
  - malloc_trim forces OS memory return on Linux sandboxes

Usage:
    with ModelContext(loader_fn, *args, **kwargs) as model:
        result = model.encode(...)
    # model is released, memory returned before next load
"""

import ctypes
import gc
from typing import Any, Callable

from src.utils.logger import get_logger

log = get_logger(__name__)


def _force_memory_release() -> None:
    """
    Force Python and OS-level memory release.

    Two GC passes handle cyclic references in complex objects
    like transformer models. malloc_trim returns freed heap
    pages to the OS immediately — critical on constrained
    Docker sandboxes where the allocator holds memory by default.
    """
    gc.collect()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        # Non-Linux environment (Windows/Mac dev) — safe to skip
        pass


class ModelContext:
    """
    Context manager for sequential model lifecycle management.

    Parameters
    ----------
    loader_fn : Callable that loads and returns the model.
    *args     : Positional args passed to loader_fn.
    **kwargs  : Keyword args passed to loader_fn.

    Example
    -------
    with ModelContext(load_embedder) as embedder:
        vecs = embedder.encode(texts)
    # embedder released here — memory returned before next model loads
    """

    def __init__(self, loader_fn: Callable, *args: Any, **kwargs: Any) -> None:
        self._loader_fn = loader_fn
        self._args      = args
        self._kwargs    = kwargs
        self._model     = None

    def __enter__(self):
        log.info(f"Loading model via {self._loader_fn.__name__} ...")
        self._model = self._loader_fn(*self._args, **self._kwargs)
        log.info("Model loaded")
        return self._model

    def __exit__(self, *args: Any) -> None:
        log.info("Releasing model from memory ...")
        self._model = None
        del self._model
        _force_memory_release()
        log.info("Model released — memory returned")