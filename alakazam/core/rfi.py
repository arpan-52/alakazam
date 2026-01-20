"""
RFI Flagging Module.

Implements MAD (Median Absolute Deviation) based flagging per baseline.
Uses robust statistics to detect outliers in visibility data.
"""

import numpy as np
from numba import njit, prange
from typing import Dict, Tuple
import logging

logger = logging.getLogger('jackal')


@njit(cache=True)
def _compute_mad(data: np.ndarray) -> float:
    """
    Compute Median Absolute Deviation (MAD).

    MAD = median(|x - median(x)|)

    Returns MAD * 1.4826 for Gaussian equivalence (approximates standard deviation).
    """
    if len(data) == 0:
        return 0.0

    median = np.median(data)
    deviations = np.abs(data - median)
    mad = np.median(deviations)

    # Scale factor for Gaussian equivalence
    return mad * 1.4826


@njit(parallel=True, cache=True)
def _mad_flag_per_baseline(vis: np.ndarray, antenna1: np.ndarray, antenna2: np.ndarray,
                           threshold: float, existing_flags: np.ndarray) -> np.ndarray:
    """
    Flag outliers using MAD per baseline.

    vis: (n_bl, n_freq, 2, 2) or (n_bl, 2, 2)
    antenna1, antenna2: (n_bl,)
    threshold: sigma threshold for flagging (e.g., 5.0)
    existing_flags: same shape as vis

    Returns: flags (same shape as vis)
    """
    n_bl = len(antenna1)
    flags = existing_flags.copy()

    # Handle both 4D and 3D cases
    if vis.ndim == 4:
        n_freq = vis.shape[1]

        # Process each baseline
        for bl in prange(n_bl):
            for pol_i in range(2):
                for pol_j in range(2):
                    # Get unflagged amplitudes for this baseline and polarization
                    amps = []
                    for f in range(n_freq):
                        if not existing_flags[bl, f, pol_i, pol_j]:
                            amps.append(np.abs(vis[bl, f, pol_i, pol_j]))

                    if len(amps) < 3:
                        # Not enough data, flag everything
                        for f in range(n_freq):
                            flags[bl, f, pol_i, pol_j] = True
                        continue

                    amps_arr = np.array(amps)
                    mad = _compute_mad(amps_arr)
                    median_amp = np.median(amps_arr)

                    if mad < 1e-10:
                        # All data is identical or mad is zero, skip
                        continue

                    # Flag outliers
                    for f in range(n_freq):
                        if not existing_flags[bl, f, pol_i, pol_j]:
                            amp = np.abs(vis[bl, f, pol_i, pol_j])
                            if np.abs(amp - median_amp) > threshold * mad:
                                flags[bl, f, pol_i, pol_j] = True

    elif vis.ndim == 3:
        # Frequency-averaged case (n_bl, 2, 2)
        # Collect all amplitudes per polarization across baselines
        for pol_i in range(2):
            for pol_j in range(2):
                amps = []
                amp_indices = []

                for bl in range(n_bl):
                    if not existing_flags[bl, pol_i, pol_j]:
                        amps.append(np.abs(vis[bl, pol_i, pol_j]))
                        amp_indices.append(bl)

                if len(amps) < 3:
                    continue

                amps_arr = np.array(amps)
                mad = _compute_mad(amps_arr)
                median_amp = np.median(amps_arr)

                if mad < 1e-10:
                    continue

                # Flag outliers
                for idx, bl in enumerate(amp_indices):
                    amp = amps_arr[idx]
                    if np.abs(amp - median_amp) > threshold * mad:
                        flags[bl, pol_i, pol_j] = True

    return flags


def flag_rfi_mad(vis_obs: np.ndarray, antenna1: np.ndarray, antenna2: np.ndarray,
                 threshold: float = 5.0, existing_flags: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
    """
    Flag RFI using MAD per baseline.

    Parameters
    ----------
    vis_obs : np.ndarray
        Visibilities, shape (n_bl, n_freq, 2, 2) or (n_bl, 2, 2)
    antenna1, antenna2 : np.ndarray
        Baseline antenna indices
    threshold : float
        Sigma threshold for flagging (default: 5.0)
    existing_flags : np.ndarray, optional
        Existing flags to respect

    Returns
    -------
    flags : np.ndarray
        Boolean flags, same shape as vis_obs
    stats : dict
        Flagging statistics
    """
    n_bl = len(antenna1)

    if existing_flags is None:
        existing_flags = np.zeros(vis_obs.shape, dtype=bool)

    # Ensure contiguous
    vis_obs = np.ascontiguousarray(vis_obs, dtype=np.complex128)
    existing_flags = np.ascontiguousarray(existing_flags, dtype=bool)

    n_total = vis_obs.size
    n_flagged_before = np.sum(existing_flags)

    logger.info(f"  RFI flagging: MAD threshold={threshold:.1f}σ, {n_bl} baselines")

    # Apply MAD flagging
    flags = _mad_flag_per_baseline(vis_obs, antenna1, antenna2, threshold, existing_flags)

    n_flagged_after = np.sum(flags)
    n_new_flags = n_flagged_after - n_flagged_before

    stats = {
        'n_total': n_total,
        'n_flagged_before': int(n_flagged_before),
        'n_flagged_after': int(n_flagged_after),
        'n_new_flags': int(n_new_flags),
        'fraction_flagged': float(n_flagged_after) / n_total if n_total > 0 else 0.0,
        'fraction_new': float(n_new_flags) / n_total if n_total > 0 else 0.0
    }

    logger.info(f"  RFI flagged: {n_new_flags} new ({stats['fraction_new']*100:.2f}%), "
               f"total: {n_flagged_after} ({stats['fraction_flagged']*100:.2f}%)")

    return flags, stats


def apply_flags(vis: np.ndarray, flags: np.ndarray, fill_value: complex = 0.0+0.0j) -> np.ndarray:
    """
    Apply flags to visibilities by setting flagged data to fill_value.

    Parameters
    ----------
    vis : np.ndarray
        Visibilities
    flags : np.ndarray
        Boolean flags
    fill_value : complex
        Value to set for flagged data

    Returns
    -------
    vis_flagged : np.ndarray
        Visibilities with flags applied
    """
    vis_out = vis.copy()
    vis_out[flags] = fill_value
    return vis_out


__all__ = ['flag_rfi_mad', 'apply_flags']
