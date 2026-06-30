"""
features/f9_salary_fit.py
──────────────────────────
F9 — Salary Fit multiplier.

Soft gate for market alignment with Senior AI Engineer role
at a Series A company in India. Target range: ₹15-80 LPA.

Applied multiplicatively in composite scoring.
Missing salary data = neutral (1.0), never penalised.

Not an elimination gate — a signal that a candidate expecting
₹3 LPA is likely junior, and one expecting ₹300 LPA is likely
out of range for a Series A startup.
"""

from src.config import (
    SALARY_TARGET_MIN,
    SALARY_TARGET_MAX,
    SALARY_EDGE_MIN,
    SALARY_EDGE_MAX,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


def score_f9_salary_fit(candidate: dict, jd_object: dict) -> float:
    """
    Compute salary fit multiplier.

    Parameters
    ----------
    candidate  : Raw candidate dict.
    jd_object  : Parsed JD object (unused, for interface consistency).

    Returns
    -------
    float multiplier in [0.65, 1.00].
    Returns 1.0 (neutral) if salary data missing or malformed.
    """
    signals = candidate.get("redrob_signals", {})
    if not signals:
        return 1.0

    salary = signals.get("expected_salary_range_inr_lpa", {})
    if not salary or not isinstance(salary, dict):
        return 1.0

    min_sal = salary.get("min")
    max_sal = salary.get("max")

    if min_sal is None and max_sal is None:
        return 1.0

    min_val = float(min_sal) if min_sal is not None else 30.0
    max_val = float(max_sal) if max_sal is not None else 30.0
    mid     = (min_val + max_val) / 2.0

    if SALARY_TARGET_MIN <= mid <= SALARY_TARGET_MAX:
        return 1.00   # Prime range for Senior AI Engineer
    elif SALARY_EDGE_MIN <= mid <= SALARY_EDGE_MAX:
        return 0.85   # Slightly outside but plausible
    else:
        return 0.65   # Clear mismatch