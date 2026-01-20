"""
Solution Quality Metrics Module.

Computes quality metrics for calibration solutions:
- SNR: Signal-to-Noise Ratio
- RMSE: Root Mean Squared Error of residuals
- Reduced χ²: Should be ≈ 1 if noise model is correct
- R²: Variance explained (coefficient of determination)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any
import logging

logger = logging.getLogger('jackal')


@dataclass
class SolutionQuality:
    """
    Quality metrics for a calibration solution.

    Attributes
    ----------
    snr : float
        Effective Signal-to-Noise Ratio
    rmse : float
        Root Mean Squared Error of residuals
    reduced_chi2 : float
        Reduced χ² (chi-squared per degree of freedom)
    r_squared : float
        R² coefficient of determination (variance explained)
    cost_reduction_ratio : float
        Ratio of final to initial cost (convergence measure)
    unflagged_fraction : float
        Fraction of unflagged visibilities
    n_iterations : int
        Number of optimization iterations
    n_data_points : int
        Number of unflagged data points
    n_parameters : int
        Number of free parameters
    convergence_success : bool
        Whether optimization converged
    """
    snr: float
    rmse: float
    reduced_chi2: float
    r_squared: float
    cost_reduction_ratio: float
    unflagged_fraction: float
    n_iterations: int
    n_data_points: int
    n_parameters: int
    convergence_success: bool

    def is_low_snr(self, threshold: float = 3.0) -> bool:
        """Check if SNR is below warning threshold."""
        return self.snr < threshold

    def quality_summary(self) -> str:
        """Generate summary string of quality metrics."""
        lines = []
        lines.append("Quality Metrics:")
        lines.append(f"  SNR:             {self.snr:.2f}")
        lines.append(f"  RMSE:            {self.rmse:.4f}")
        lines.append(f"  Reduced χ²:      {self.reduced_chi2:.4f}")
        lines.append(f"  R²:              {self.r_squared:.4f}")
        lines.append(f"  Cost reduction:  {self.cost_reduction_ratio:.4f}")
        lines.append(f"  Unflagged data:  {self.unflagged_fraction*100:.1f}%")
        lines.append(f"  Iterations:      {self.n_iterations}")
        lines.append(f"  Converged:       {self.convergence_success}")

        # Add warning if low SNR
        if self.is_low_snr():
            lines.append(f"  WARNING: Low SNR ({self.snr:.2f} < 3.0)")

        return "\n".join(lines)


def compute_snr(
    residual_rms: float,
    n_data_points: int
) -> float:
    """
    Compute effective Signal-to-Noise Ratio.

    SNR = sqrt(N) / RMS

    Where:
    - N = number of unflagged data points
    - RMS = root mean squared residual

    Parameters
    ----------
    residual_rms : float
        RMS of residuals
    n_data_points : int
        Number of unflagged data points

    Returns
    -------
    float
        Effective SNR
    """
    if residual_rms < 1e-10 or n_data_points < 1:
        return 0.0

    snr = np.sqrt(n_data_points) / residual_rms
    return snr


def compute_rmse(cost: float, n_data_points: int) -> float:
    """
    Compute Root Mean Squared Error.

    RMSE = sqrt(sum(residuals²) / N)

    For least_squares with method='lm', cost = 0.5 * sum(residuals²)
    So: RMSE = sqrt(2 * cost / N)

    Parameters
    ----------
    cost : float
        Final cost from optimizer (from least_squares)
    n_data_points : int
        Number of unflagged data points

    Returns
    -------
    float
        RMSE
    """
    if n_data_points < 1:
        return np.inf

    # least_squares returns cost = 0.5 * sum(residuals²)
    # But result.cost already accounts for this, so:
    rmse = np.sqrt(2.0 * cost / n_data_points)
    return rmse


def compute_reduced_chi2(
    cost: float,
    n_data_points: int,
    n_parameters: int
) -> float:
    """
    Compute reduced chi-squared statistic.

    χ²_ν = χ² / ν

    Where:
    - χ² = sum(residuals²) = 2 * cost (for least_squares)
    - ν = degrees of freedom = N_data - N_params

    For a good fit with correct noise model: χ²_ν ≈ 1

    Parameters
    ----------
    cost : float
        Final cost from optimizer
    n_data_points : int
        Number of unflagged data points
    n_parameters : int
        Number of free parameters

    Returns
    -------
    float
        Reduced χ²
    """
    dof = n_data_points - n_parameters

    if dof <= 0:
        return np.inf

    # cost = 0.5 * sum(residuals²), so chi2 = 2 * cost
    chi2 = 2.0 * cost
    reduced_chi2 = chi2 / dof

    return reduced_chi2


def compute_r_squared(cost_init: float, cost_final: float) -> float:
    """
    Compute R² coefficient of determination.

    R² = 1 - (SS_res / SS_tot)
       = 1 - (cost_final / cost_init)

    Where:
    - SS_res = sum of squared residuals = cost_final
    - SS_tot = total sum of squares ≈ cost_init (before calibration)

    R² = 1 means perfect fit
    R² = 0 means no improvement
    R² < 0 means fit is worse than mean

    Parameters
    ----------
    cost_init : float
        Initial cost (before optimization)
    cost_final : float
        Final cost (after optimization)

    Returns
    -------
    float
        R² statistic
    """
    if cost_init < 1e-10:
        return 0.0

    r_squared = 1.0 - (cost_final / cost_init)
    return r_squared


def compute_quality_metrics(
    cost_init: float,
    cost_final: float,
    n_data_points: int,
    n_parameters: int,
    convergence_info: Dict[str, Any],
    n_total_vis: int
) -> SolutionQuality:
    """
    Compute all quality metrics for a solution.

    Parameters
    ----------
    cost_init : float
        Initial cost before optimization
    cost_final : float
        Final cost after optimization
    n_data_points : int
        Number of unflagged data points used in optimization
    n_parameters : int
        Number of free parameters
    convergence_info : dict
        Convergence information from optimizer
        Keys: 'nfev', 'success', etc.
    n_total_vis : int
        Total number of visibilities (including flagged)

    Returns
    -------
    SolutionQuality
        Quality metrics object
    """
    # RMSE
    rmse = compute_rmse(cost_final, n_data_points)

    # SNR
    snr = compute_snr(rmse, n_data_points)

    # Reduced χ²
    reduced_chi2 = compute_reduced_chi2(cost_final, n_data_points, n_parameters)

    # R²
    r_squared = compute_r_squared(cost_init, cost_final)

    # Cost reduction ratio
    cost_reduction_ratio = cost_final / max(cost_init, 1e-10)

    # Unflagged fraction
    unflagged_fraction = n_data_points / max(n_total_vis, 1)

    # Convergence info
    n_iterations = convergence_info.get('nfev', 0)
    success = convergence_info.get('success', False)

    return SolutionQuality(
        snr=snr,
        rmse=rmse,
        reduced_chi2=reduced_chi2,
        r_squared=r_squared,
        cost_reduction_ratio=cost_reduction_ratio,
        unflagged_fraction=unflagged_fraction,
        n_iterations=n_iterations,
        n_data_points=n_data_points,
        n_parameters=n_parameters,
        convergence_success=success
    )


def print_quality_metrics(quality: SolutionQuality, verbose: bool = True):
    """
    Print quality metrics to logger.

    Parameters
    ----------
    quality : SolutionQuality
        Quality metrics object
    verbose : bool
        If True, print detailed metrics
    """
    if not verbose:
        return

    logger.info(quality.quality_summary())

    # Print warnings
    if quality.is_low_snr():
        logger.warning(f"Low SNR solution: {quality.snr:.2f} < 3.0 - results may be unreliable")

    if not quality.convergence_success:
        logger.warning("Optimization did not converge successfully")

    if quality.unflagged_fraction < 0.2:
        logger.warning(f"Low unflagged fraction: {quality.unflagged_fraction*100:.1f}% - insufficient data")

    if quality.reduced_chi2 > 2.0:
        logger.warning(f"High reduced χ²: {quality.reduced_chi2:.2f} > 2.0 - poor fit or underestimated noise")


__all__ = [
    'SolutionQuality',
    'compute_quality_metrics',
    'compute_snr',
    'compute_rmse',
    'compute_reduced_chi2',
    'compute_r_squared',
    'print_quality_metrics',
]
