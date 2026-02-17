"""
ALAKAZAM Flux Scale Bootstrapping.

Transfers absolute flux scale from a calibrator field to a target field
by comparing gain amplitudes.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger("alakazam")


def compute_fluxscale(
    gains_calibrator: np.ndarray,
    gains_target: np.ndarray,
    working_ants: np.ndarray,
    ref_ant: int,
) -> Tuple[np.ndarray, Dict]:
    """Compute flux scale factors from calibrator to target gains.

    Parameters
    ----------
    gains_calibrator : ndarray (n_ant, 2, 2) complex128
        Gain solutions on calibrator (known flux).
    gains_target : ndarray (n_ant, 2, 2) complex128
        Gain solutions on target (unknown flux).
    working_ants : ndarray
        Working antenna indices.
    ref_ant : int
        Reference antenna.

    Returns
    -------
    scale_factors : ndarray (2,) float64
        Flux scale factor per polarization [p, q].
    info : dict
        Statistics (median ratio per antenna, scatter).
    """
    ratios_p = []
    ratios_q = []

    for ant in working_ants:
        if ant == ref_ant:
            continue

        # Amplitude ratio: |g_cal| / |g_target|
        amp_cal_p = np.abs(gains_calibrator[ant, 0, 0])
        amp_cal_q = np.abs(gains_calibrator[ant, 1, 1])
        amp_tgt_p = np.abs(gains_target[ant, 0, 0])
        amp_tgt_q = np.abs(gains_target[ant, 1, 1])

        if amp_tgt_p > 1e-20 and np.isfinite(amp_cal_p):
            ratios_p.append(amp_cal_p / amp_tgt_p)
        if amp_tgt_q > 1e-20 and np.isfinite(amp_cal_q):
            ratios_q.append(amp_cal_q / amp_tgt_q)

    if len(ratios_p) == 0 or len(ratios_q) == 0:
        logger.warning("Fluxscale: insufficient valid antennas")
        return np.array([1.0, 1.0]), {"warning": "insufficient data"}

    ratios_p = np.array(ratios_p)
    ratios_q = np.array(ratios_q)

    # Flux density scales as amplitude^2
    scale_p = float(np.median(ratios_p))**2
    scale_q = float(np.median(ratios_q))**2

    info = {
        "scale_p": scale_p,
        "scale_q": scale_q,
        "n_antennas": len(ratios_p),
        "scatter_p": float(np.std(ratios_p) / np.mean(ratios_p)) if len(ratios_p) > 1 else 0.0,
        "scatter_q": float(np.std(ratios_q) / np.mean(ratios_q)) if len(ratios_q) > 1 else 0.0,
    }

    logger.info(
        f"Fluxscale: scale_p={scale_p:.4f}, scale_q={scale_q:.4f}, "
        f"scatter_p={info['scatter_p']:.3f}, scatter_q={info['scatter_q']:.3f}"
    )

    return np.array([scale_p, scale_q]), info


def apply_fluxscale(
    model_vis: np.ndarray,
    scale_factors: np.ndarray,
) -> np.ndarray:
    """Apply flux scale to model visibilities.

    model_vis:     (n_bl, [n_freq,] 2, 2) complex128
    scale_factors: (2,) float64

    Returns: scaled model visibilities
    """
    scaled = model_vis.copy()
    sp = np.sqrt(scale_factors[0])
    sq = np.sqrt(scale_factors[1])

    if scaled.ndim == 3:
        scaled[:, 0, 0] *= sp * sp
        scaled[:, 0, 1] *= sp * sq
        scaled[:, 1, 0] *= sq * sp
        scaled[:, 1, 1] *= sq * sq
    elif scaled.ndim == 4:
        scaled[:, :, 0, 0] *= sp * sp
        scaled[:, :, 0, 1] *= sp * sq
        scaled[:, :, 1, 0] *= sq * sp
        scaled[:, :, 1, 1] *= sq * sq

    return scaled
