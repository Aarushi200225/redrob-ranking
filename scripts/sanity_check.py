"""
scripts/sanity_check.py
────────────────────────
Quick sanity checks for core pipeline components.
Run before every full pipeline execution.

Usage: python scripts/sanity_check.py
"""

import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.scoring import normalize_batch
from src.features.f8_honeypot_gate import (
    _has_expert_zero_duration,
    _expert_per_year_ratio,
)

print("=" * 50)
print("Pipeline Sanity Check")
print("=" * 50)

# Check 1: MinMax normalisation
print("\n[1] MinMax normalisation...")
raw = np.array([-5.2, 0.1, 3.5, 1.2], dtype=np.float32)
norm = normalize_batch(raw)
assert abs(norm.max() - 1.0) < 1e-5, "Ceiling broken"
assert abs(norm.min() - 0.0) < 1e-5, "Floor broken"
print(f"    Raw:        {raw.tolist()}")
print(f"    Normalised: {norm.tolist()}")
print("    ✓ PASS")

# Check 2: Honeypot detection
print("\n[2] Honeypot detection...")
honeypot = {
    "profile": {"years_of_experience": 2},
    "career_history": [{"duration_months": 24}],
    "skills": [
        {"name": "FAISS",   "proficiency": "expert", "duration_months": 0},
        {"name": "PyTorch", "proficiency": "expert", "duration_months": 0},
    ]
}
legitimate = {
    "profile": {"years_of_experience": 7},
    "career_history": [{"duration_months": 84}],
    "skills": [
        {"name": "Python", "proficiency": "expert", "duration_months": 72},
    ]
}
assert _has_expert_zero_duration(honeypot)  is True
assert _has_expert_zero_duration(legitimate) is False
print("    ✓ PASS")

# Check 3: Score weights sum to 1.0
print("\n[3] Score weights validation...")
from src.config import SCORE_WEIGHTS
total = sum(SCORE_WEIGHTS.values())
assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, not 1.0"
print(f"    Weights sum: {total:.6f}")
print("    ✓ PASS")

# Check 4: Config env
print("\n[4] Environment config...")
import os
from src.config import ENV, FINAL_TOP_K, EMBED_BATCH_SIZE
print(f"    REDROB_ENV:       {os.getenv('REDROB_ENV', 'submission')}")
print(f"    ENV profile:      {ENV}")
print(f"    FINAL_TOP_K:      {FINAL_TOP_K}")
print(f"    EMBED_BATCH_SIZE: {EMBED_BATCH_SIZE}")
print("    ✓ PASS")

print("\n" + "=" * 50)
print("All checks passed — pipeline is ready")
print("=" * 50)