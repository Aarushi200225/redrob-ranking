"""
config.py
─────────
Single source of truth for all tunable parameters, file paths, and
model identifiers across the pipeline. Nothing is hardcoded in stage
or feature modules — they import from here.

Changing a weight, threshold, or path means editing exactly one file.
"""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT_DIR / "data"
MODEL_DIR    = ROOT_DIR / "models"
ARTIFACT_DIR = ROOT_DIR / "artifacts"
OUTPUT_DIR   = ROOT_DIR / "outputs"
CACHE_DIR    = ROOT_DIR / ".cache"

# ── Input files ───────────────────────────────────────────────────────────────
CANDIDATES_PATH     = DATA_DIR / "candidates.jsonl"
JD_PATH             = DATA_DIR / "job_description.txt"
SKILL_TAXONOMY_PATH = ARTIFACT_DIR / "skill_taxonomy.json"
VIBE_KEYWORDS_PATH  = ARTIFACT_DIR / "vibe_keywords.json"

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_CSV_PATH = OUTPUT_DIR / "ranked_output.csv"

# ── Model identifiers ─────────────────────────────────────────────────────────
LLM_MODEL_PATH         = MODEL_DIR / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
EMBEDDER_MODEL_ID      = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── LLM inference settings ────────────────────────────────────────────────────
LLM_N_CTX      = 2048   # Context window
LLM_N_CTX_REASONING = 512   # Smaller context for reasoning generation only
LLM_MAX_TOKENS = 120   # Reasoning output cap — 1-2 sentences
LLM_TEMPERATURE = 0.3  # Low temp — factual, consistent
LLM_N_THREADS  = 4     # CPU thread count for llama.cpp

# ── Embedding settings ────────────────────────────────────────────────────────
EMBED_BATCH_SIZE  = 512   # Optimal for CPU batch inference
EMBED_NORMALIZE   = True  # Required for cosine via inner product

# ── Stage 2 — BM25 ───────────────────────────────────────────────────────────
import os
BM25_WORKERS         = 6      # ProcessPoolExecutor workers # Retained for future use — BM25 runs sequentially on Windows
BM25_CHAMBER_A_TOP_K = 5000   # Raw JD terms
BM25_CHAMBER_B_TOP_K = 5000   # Expanded taxonomy terms
BM25_UNION_CAP       = 15000  # Max unique after union

# ── Stage 3 — FAISS + RRF ────────────────────────────────────────────────────
FAISS_TOP_K_PER_QUERY = 2000  # Per query vector
RRF_K                 = 60    # RRF constant
RETRIEVAL_FINAL_POOL  = 2000  # After RRF fusion

# ── Stage 4 — Reranking ──────────────────────────────────────────────────────
CROSS_ENCODER_TOP_K = 2000
RERANK_POOL_SIZE    = 500
FINAL_TOP_K         = 100

# Composite score weights — must sum to 1.0
# Full justification in ARCHITECTURE.md §12
SCORE_WEIGHTS = {
    "cross_encoder": 0.25,
    "vibe":          0.18,
    "semantic":      0.20,
    "experience":    0.15,
    "skills":        0.12,
    "availability":  0.07,
    "market":        0.03,
}

assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-6, (
    f"Score weights must sum to 1.0, got {sum(SCORE_WEIGHTS.values()):.4f}"
)

# ── F2 — Experience band scoring ──────────────────────────────────────────────
EXPERIENCE_BAND_SCORES = [
    (5,  9,  1.00),   # sweet spot
    (4,  10, 0.85),   # JD: judgment can come at 4 years
    (3,  4,  0.55),
    (10, 12, 0.65),   # over-experienced risk
    (12, 99, 0.45),
    (0,  3,  0.25),
]

# ── F5 — Notice period scoring ────────────────────────────────────────────────
NOTICE_PERIOD_SCORES = {
    30:  1.00,
    60:  0.88,
    90:  0.75,
    180: 0.55,
}

# ── F7 — Location multipliers ─────────────────────────────────────────────────
LOCATION_MULTIPLIERS = {
    "tier_1":              1.00,
    "tier_2":              0.92,
    "india_relocate":      0.82,
    "india_no_relocate":   0.65,
    "outside_relocate":    0.55,
    "outside_no_relocate": 0.30,
}

LOCATION_TIER_1 = {"pune", "noida"}
LOCATION_TIER_2 = {"hyderabad", "mumbai", "delhi", "delhi ncr", "gurugram", "gurgaon"}

# ── F8 — Honeypot gate ────────────────────────────────────────────────────────
HONEYPOT_TIMELINE_DELTA_MONTHS = 36   # Only flag genuine overcount
HONEYPOT_EXPERT_SKILL_THRESHOLD  = 8    #Tightened from 10 to 8 expert skills
HONEYPOT_YOE_THRESHOLD           = 3
HONEYPOT_SCORE_GATE              = 0.55 #Raised from 0.4 to 0.55 — stricter gate

# ── Anti-stuffer penalty ──────────────────────────────────────────────────────
ANTI_STUFFER_SKILL_COUNT          = 15
ANTI_STUFFER_AVG_DURATION_MONTHS  = 10

# ── Recency decay ─────────────────────────────────────────────────────────────
RECENCY_HARD_CUTOFF_DAYS  = 180
RECENCY_HARD_MULTIPLIER   = 0.3

# ── Consulting firms — hard gate ──────────────────────────────────────────────
CONSULTING_FIRMS = {
    "tcs", "tata consultancy services",
    "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl",
    "tech mahindra", "mphasis", "hexaware",
    "l&t infotech", "ltimindtree", "persistent systems",
    "mindtree", "niit technologies",
}

# ── Stage 5 — Reasoning split ─────────────────────────────────────────────────
LLM_REASONING_TOP_N   = 40
STRUCTURED_START_RANK = 41

# ── HyDE query weighting ──────────────────────────────────────────────────────
HYDE_QUERY_WEIGHT = 0.6
JD_QUERY_WEIGHT   = 0.4

# ── Pipeline timing targets (seconds) ────────────────────────────────────────
TIMING_TARGETS = {
    "stage_1": 30,
    "stage_2": 19,
    "stage_3": 26,
    "stage_4": 65,
    "stage_5": 35,
    "total":   240,
    "budget":  300,
}