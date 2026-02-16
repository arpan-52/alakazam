"""
ALAKAZAM Solution Quality Metrics.

Computes SNR, RMSE, reduced chi-square, R-squared per solution cell.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Quality metrics for one solution cell."""
    snr: float = 0.0
    rmse: float = 0.0
    chi2_red: float = 0.0
    r_squared: float = 0.0
    cost_init: float = 0.0
    cost_final: float = 0.0
    nfev: int = 0
    success: bool = False
    n_data: int = 0
    n_params: int = 0


def compute_quality(info: dict) -> QualityMetrics:
    """Compute quality metrics from solver info dict."""
    cost_init = info.get("cost_init", 0.0)
    cost_final = info.get("cost_final", 0.0)
    n_data = info.get("n_data", 0)
    n_params = info.get("n_params", 0)
    dof = max(1, n_data - n_params)

    chi2_red = cost_final / dof if dof > 0 else 0.0
    rmse = np.sqrt(cost_final / max(1, n_data))
    r_squared = 1.0 - cost_final / max(cost_init, 1e-30) if cost_init > 0 else 0.0
    snr = np.sqrt(max(0, cost_init - cost_final)) / max(rmse, 1e-30) if rmse > 0 else 0.0

    return QualityMetrics(
        snr=float(snr),
        rmse=float(rmse),
        chi2_red=float(chi2_red),
        r_squared=float(r_squared),
        cost_init=float(cost_init),
        cost_final=float(cost_final),
        nfev=int(info.get("nfev", 0)),
        success=bool(info.get("success", False)),
        n_data=n_data,
        n_params=n_params,
    )
