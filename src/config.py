"""
config.py
─────────
Single source of truth for all pipeline parameters.

Two-track environment support:
  REDROB_ENV=submission  (default) — hackathon Linux sandbox, full power
  REDROB_ENV=dev                   — local Windows testing, reduced params

Usage:
  # Local dev run
  set REDROB_ENV=dev && python run.py run-sample

  # Submission (default, no flag needed)
  python run.py run
"""

import os
from pathlib import Path

# ── Environment ───────────────────────────────────────────────────────────────
ENV = os.getenv("REDROB_ENV", "submission")

# ── Project paths ─────────────────────────────────────────────────────────────
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

# ── LLM settings ──────────────────────────────────────────────────────────────
LLM_N_THREADS         = 4
LLM_TEMPERATURE       = 0.1    # Low — factual extraction
LLM_MAX_TOKENS        = 1025    # JD parsing output cap
LLM_REASONING_TOKENS  = 120    # Reasoning string output cap

# Two n_ctx values — JD parsing needs more context than reasoning generation
LLM_N_CTX             = 2048   # Stage 1 JD parsing + HyDE
LLM_N_CTX_REASONING   = 512    # Stage 5 reasoning generation
LLM_N_BATCH           = 512    # Stage 1 — larger batch for parsing
LLM_N_BATCH_REASONING = 32     # Stage 5 — sequential generation, no benefit from large batch

# ── Embedding settings ────────────────────────────────────────────────────────
# Two-track: full batch size for submission, reduced for local dev
EMBED_BATCH_SIZE = 512 if ENV == "submission" else 32
EMBED_NORMALIZE  = True

# ── Stage 2 — BM25 ───────────────────────────────────────────────────────────
BM25_CHAMBER_A_TOP_K = 5000
BM25_CHAMBER_B_TOP_K = 5000
BM25_UNION_CAP       = 15000 if ENV == "submission" else 2000

# ── Stage 3 — FAISS + RRF ────────────────────────────────────────────────────
FAISS_TOP_K_PER_QUERY = 2000
RRF_K                 = 60
RETRIEVAL_FINAL_POOL  = 2000 if ENV == "submission" else 200

# ── Stage 4 — Reranking ──────────────────────────────────────────────────────
CROSS_ENCODER_TOP_K = 2000 if ENV == "submission" else 200
RERANK_POOL_SIZE    = 500  if ENV == "submission" else 100
FINAL_TOP_K         = 100

# Pre-filter thresholds — bypass cross-encoder for clearly irrelevant candidates
# Both must be below threshold — conservative, avoids false exclusions
CE_PREFILTER_F2_THRESHOLD = 0.20
CE_PREFILTER_F3_THRESHOLD = 0.20

# Composite score weights — must sum to 1.0
# Justified in ARCHITECTURE.md §12
SCORE_WEIGHTS = {
    "cross_encoder": 0.25,   # Strongest pairwise relevance signal
    "vibe":          0.18,   # JD: culture-fit > skills-fit
    "semantic":      0.20,   # RRF retrieval confidence
    "experience":    0.15,   # Production deployment, pre-2022 ML
    "skills":        0.12,   # Deliberately below vibe — skills teachable
    "availability":  0.07,   # Recruiter actionability
    "market":        0.03,   # Weak noisy signal
}

assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-6, (
    f"Score weights must sum to 1.0, got {sum(SCORE_WEIGHTS.values()):.6f}"
)

# ── Stage 5 — Reasoning ──────────────────────────────────────────────────────
LLM_REASONING_TOP_N   = 40 if ENV == "submission" else 5
STRUCTURED_START_RANK = LLM_REASONING_TOP_N + 1

# ── Dev profile — candidate limit ────────────────────────────────────────────
# None = process all 100K (submission)
# 500  = process first 500 candidates (local dev testing)
DEV_CANDIDATE_LIMIT = None if ENV == "submission" else 500

# ── F2 — Experience band scoring ──────────────────────────────────────────────
EXPERIENCE_BAND_SCORES = [
    (5,  9,  1.00),   # Sweet spot per JD
    (4,  10, 0.85),   # JD: "judgment can come at 4 years"
    (3,  4,  0.55),
    (10, 12, 0.65),   # Over-experienced risk
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
LOCATION_TIER_2 = {"hyderabad", "mumbai", "delhi", "delhi ncr",
                   "gurugram", "gurgaon"}

# ── F8 — Honeypot gate ────────────────────────────────────────────────────────
HONEYPOT_TIMELINE_DELTA_MONTHS   = 36
HONEYPOT_EXPERT_SKILL_THRESHOLD  = 8
HONEYPOT_YOE_THRESHOLD           = 3
HONEYPOT_SCORE_GATE              = 0.55

# ── F9 — Salary fit multiplier ────────────────────────────────────────────────
SALARY_TARGET_MIN = 15.0
SALARY_TARGET_MAX = 80.0
SALARY_EDGE_MIN   = 10.0
SALARY_EDGE_MAX   = 120.0

# ── Anti-stuffer penalty ──────────────────────────────────────────────────────
ANTI_STUFFER_SKILL_COUNT         = 15
ANTI_STUFFER_AVG_DURATION_MONTHS = 10

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

# ── HyDE query weighting ──────────────────────────────────────────────────────
HYDE_QUERY_WEIGHT = 0.6
JD_QUERY_WEIGHT   = 0.4

# ── Pipeline timing targets (seconds) ────────────────────────────────────────
TIMING_TARGETS = {
    "stage_1": 25,
    "stage_2": 90,
    "stage_3": 25,
    "stage_4": 55,
    "stage_5": 35,
    "total":   240,
    "budget":  300,
}