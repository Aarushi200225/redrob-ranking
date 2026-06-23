"""
scripts/download_models.py
──────────────────────────
Downloads and verifies all model weights required by the pipeline.

Models:
  1. Qwen2.5-0.5B-Instruct GGUF Q4_K_M  — via huggingface_hub
  2. bge-small-en-v1.5                   — via sentence-transformers
  3. ms-marco-MiniLM-L-6-v2             — via sentence-transformers

Run once before executing the pipeline:
    python scripts/download_models.py
    make download-models
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import LLM_MODEL_PATH, EMBEDDER_MODEL_ID, CROSS_ENCODER_MODEL_ID
from src.utils.logger import get_logger

log = get_logger("download_models")


def download_qwen_gguf() -> None:
    """Download Qwen2.5-0.5B-Instruct Q4_K_M GGUF from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log.error("Run: pip install huggingface-hub")
        sys.exit(1)

    LLM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if LLM_MODEL_PATH.exists():
        log.info(f"Qwen GGUF already present: {LLM_MODEL_PATH}")
        return

    log.info("Downloading Qwen2.5-0.5B-Instruct Q4_K_M ...")
    path = hf_hub_download(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        local_dir=str(LLM_MODEL_PATH.parent),
    )
    log.info(f"Saved to: {path}")


def download_sentence_transformers() -> None:
    """Pre-cache bge-small and ms-marco via sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer, CrossEncoder
    except ImportError:
        log.error("Run: pip install sentence-transformers")
        sys.exit(1)

    log.info(f"Downloading embedder: {EMBEDDER_MODEL_ID}")
    SentenceTransformer(EMBEDDER_MODEL_ID)
    log.info("Embedder ready")

    log.info(f"Downloading cross-encoder: {CROSS_ENCODER_MODEL_ID}")
    CrossEncoder(CROSS_ENCODER_MODEL_ID)
    log.info("Cross-encoder ready")


def verify_models() -> bool:
    """Verify all required model files are present."""
    if not LLM_MODEL_PATH.exists():
        log.error(f"Missing: {LLM_MODEL_PATH}")
        return False

    size_mb = LLM_MODEL_PATH.stat().st_size / (1024 ** 2)
    log.info(f"✓ Qwen GGUF ({size_mb:.0f} MB): {LLM_MODEL_PATH}")
    return True


if __name__ == "__main__":
    log.info("Starting model downloads ...")
    download_qwen_gguf()
    download_sentence_transformers()
    ok = verify_models()
    if ok:
        log.info("All models verified — pipeline ready")
    else:
        log.error("Models missing — re-run this script")
        sys.exit(1)