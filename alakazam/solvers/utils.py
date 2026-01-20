"""
Shared utilities for Jones solvers.

Functions:
- fit_phase_slope(): Fit delay from phase vs frequency
- find_ref_baselines(): Find baselines containing reference antenna
- average_with_flags(): Flag-aware averaging
"""

import numpy as np
from typing import Tuple, List
from numba import njit


@njit(cache=True)
def fit_phase_slope_weighted(phase: np.ndarray, freq: np.ndarray) -> float:
    """
    Fit delay from phase slope vs frequency.

    Linear fit: φ = -2π × τ × ν
    Return: τ in nanoseconds

    Parameters
    ----------
    phase : ndarray (n_freq,)
        Unwrapped phase in radians
    freq : ndarray (n_freq,)
        Frequencies in Hz

    Returns
    -------
    delay_ns : float
        Delay in nanoseconds
    """
    if len(phase) < 3:
        return 0.0

    # Weighted linear fit (uniform weights)
    freq_mean = np.mean(freq)
    phase_mean = np.mean(phase)

    numerator = np.sum((freq - freq_mean) * (phase - phase_mean))
    denominator = np.sum((freq - freq_mean) ** 2)

    if denominator < 1e-20:
        return 0.0

    slope = numerator / denominator  # radians/Hz
    delay_s = -slope / (2.0 * np.pi)
    delay_ns = delay_s * 1e9

    return delay_ns


def find_ref_baselines(
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    ref_ant: int
) -> List[Tuple[int, int, bool]]:
    """
    Find baselines containing reference antenna.

    Parameters
    ----------
    antenna1, antenna2 : ndarray (n_bl,)
        Antenna indices for each baseline
    ref_ant : int
        Reference antenna index

    Returns
    -------
    ref_baselines : list of (bl_idx, other_ant, flip)
        - bl_idx: baseline index
        - other_ant: the non-reference antenna
        - flip: True if ref is antenna2 (need conjugate)
    """
    ref_baselines = []

    for bl_idx in range(len(antenna1)):
        a1, a2 = antenna1[bl_idx], antenna2[bl_idx]

        if a1 == ref_ant:
            # Ref is first antenna
            ref_baselines.append((bl_idx, a2, False))
        elif a2 == ref_ant:
            # Ref is second antenna (need conjugate)
            ref_baselines.append((bl_idx, a1, True))

    return ref_baselines


def average_with_flags(
    data: np.ndarray,
    flags: np.ndarray,
    axis: int
) -> np.ndarray:
    """
    Average data along axis, respecting flags.

    Flagged data is excluded from average.

    Parameters
    ----------
    data : ndarray
        Data to average (complex or real)
    flags : ndarray (bool)
        Flags (True = flagged, False = good)
    axis : int
        Axis to average over

    Returns
    -------
    averaged : ndarray
        Averaged data
    """
    # Mask flagged data
    masked = np.where(flags, np.nan, data)

    # Average, ignoring NaNs
    with np.errstate(invalid='ignore'):
        averaged = np.nanmean(masked, axis=axis)

    return averaged


def create_working_antenna_mapping(
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    n_ant_total: int
) -> Tuple[np.ndarray, dict, dict]:
    """
    Create mapping between full and working antenna indices.

    Parameters
    ----------
    antenna1, antenna2 : ndarray
        Antenna indices in baselines
    n_ant_total : int
        Total number of antennas in array

    Returns
    -------
    working_ants : ndarray
        Full antenna indices of working antennas
    full_to_working : dict
        Map full index -> working index
    working_to_full : dict
        Map working index -> full index
    """
    working_ants = np.unique(np.concatenate([antenna1, antenna2]))
    working_ants = np.sort(working_ants)

    full_to_working = {ant: i for i, ant in enumerate(working_ants)}
    working_to_full = {i: ant for i, ant in enumerate(working_ants)}

    return working_ants, full_to_working, working_to_full


__all__ = [
    'fit_phase_slope_weighted',
    'find_ref_baselines',
    'average_with_flags',
    'create_working_antenna_mapping'
]
