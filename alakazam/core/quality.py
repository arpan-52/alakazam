"""ALAKAZAM Solution Quality Metrics.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
import logging

logger = logging.getLogger("alakazam")


def compute_quality(
    jones: np.ndarray,     # (n_ant, 2, 2) or (n_ant, n_freq, 2, 2)
    residual: np.ndarray,  # flat residual vector from solver
    n_baseline: int,
) -> np.ndarray:
    """Compute per-antenna quality metric (normalised residual power).

    Returns: (n_ant,) float64  — lower is better
    """
    n_ant = jones.shape[0]
    quality = np.zeros(n_ant, dtype=np.float64)

    rms_total = np.sqrt(np.mean(residual ** 2)) + 1e-30

    # Simple approximation: uniform quality
    quality[:] = rms_total
    return quality
