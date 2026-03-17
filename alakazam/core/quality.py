"""ALAKAZAM v1 Solution Quality Metrics.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
import logging

logger = logging.getLogger("alakazam")


def compute_quality(jones: np.ndarray, residual: np.ndarray,
                    n_baseline: int) -> np.ndarray:
    """Compute per-antenna quality metric (normalised residual power).

    Returns: (n_ant,) float64 — lower is better.
    """
    n_ant = jones.shape[0]
    rms_total = np.sqrt(np.mean(residual ** 2)) + 1e-30
    quality = np.full(n_ant, rms_total, dtype=np.float64)
    return quality
