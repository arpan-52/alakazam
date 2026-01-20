"""
Flag-aware averaging for calibration.

Averages data across time/frequency dimensions while respecting flags.
Flagged data is excluded from averages.
"""

import numpy as np
from typing import Tuple


def average_time_freq(
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    flags: np.ndarray,
    avg_time: bool,
    avg_freq: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Average visibilities across time and/or frequency, respecting flags.

    Parameters
    ----------
    vis_obs : ndarray (n_row, n_chan, 2, 2)
        Observed visibilities
    vis_model : ndarray (n_row, n_chan, 2, 2)
        Model visibilities
    flags : ndarray (n_row, n_chan, 2, 2) bool
        Flags (True = flagged)
    avg_time : bool
        Average time dimension?
    avg_freq : bool
        Average frequency dimension?

    Returns
    -------
    vis_obs_avg : ndarray
        Averaged observed visibilities
    vis_model_avg : ndarray
        Averaged model visibilities
    flags_avg : ndarray
        Averaged flags (True if ANY flagged)
    """
    if not avg_time and not avg_freq:
        # No averaging
        return vis_obs, vis_model, flags

    # Mask flagged data with NaN
    vis_obs_masked = np.where(flags, np.nan, vis_obs)
    vis_model_masked = np.where(flags, np.nan, vis_model)

    # Determine axes to average
    axes = []
    if avg_time:
        axes.append(0)
    if avg_freq:
        axes.append(1)

    # Average (ignoring NaNs)
    with np.errstate(invalid='ignore'):
        vis_obs_avg = np.nanmean(vis_obs_masked, axis=tuple(axes))
        vis_model_avg = np.nanmean(vis_model_masked, axis=tuple(axes))

    # Flags: True if ALL channels flagged (since we use nanmean to exclude bad data)
    flags_avg = np.all(flags, axis=tuple(axes))

    return vis_obs_avg, vis_model_avg, flags_avg


def average_per_baseline(
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    time: np.ndarray,
    flags: np.ndarray,
    avg_time: bool,
    avg_freq: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Average visibilities per unique baseline.

    Combines multiple time samples of same baseline into single averaged point.

    Parameters
    ----------
    vis_obs : ndarray (n_row, n_chan, 2, 2)
        Observed visibilities
    vis_model : ndarray (n_row, n_chan, 2, 2)
        Model visibilities
    antenna1, antenna2 : ndarray (n_row,)
        Antenna indices
    time : ndarray (n_row,)
        Time stamps
    flags : ndarray (n_row, n_chan, 2, 2)
        Flags
    avg_time : bool
        Average time within baseline?
    avg_freq : bool
        Average frequency?

    Returns
    -------
    vis_obs_avg : ndarray (n_bl, ...)
        Averaged observed visibilities
    vis_model_avg : ndarray (n_bl, ...)
        Averaged model visibilities
    ant1_bl : ndarray (n_bl,)
        Antenna 1 per baseline
    ant2_bl : ndarray (n_bl,)
        Antenna 2 per baseline
    time_bl : ndarray (n_bl,)
        Mean time per baseline
    flags_avg : ndarray (n_bl, ...)
        Averaged flags
    """
    # Get unique baselines
    baselines = set()
    for i in range(len(antenna1)):
        baselines.add((antenna1[i], antenna2[i]))

    baselines = sorted(list(baselines))
    n_bl = len(baselines)

    # Determine output shape
    n_chan = vis_obs.shape[1]
    if avg_freq:
        out_shape = (n_bl, 2, 2)
    else:
        out_shape = (n_bl, n_chan, 2, 2)

    # Allocate output arrays
    vis_obs_avg = np.zeros(out_shape, dtype=np.complex128)
    vis_model_avg = np.zeros(out_shape, dtype=np.complex128)
    flags_avg = np.zeros(out_shape, dtype=bool)
    ant1_bl = np.zeros(n_bl, dtype=np.int32)
    ant2_bl = np.zeros(n_bl, dtype=np.int32)
    time_bl = np.zeros(n_bl, dtype=np.float64)

    # Average per baseline
    for bl_idx, (a1, a2) in enumerate(baselines):
        # Find all rows for this baseline
        mask = (antenna1 == a1) & (antenna2 == a2)
        rows = np.where(mask)[0]

        if len(rows) == 0:
            continue

        # Extract data for this baseline
        vis_o = vis_obs[rows]
        vis_m = vis_model[rows]
        flag = flags[rows]
        t = time[rows]

        # Average time and/or freq
        vis_o_avg, vis_m_avg, flag_avg = average_time_freq(
            vis_o, vis_m, flag, avg_time=avg_time, avg_freq=avg_freq
        )

        vis_obs_avg[bl_idx] = vis_o_avg
        vis_model_avg[bl_idx] = vis_m_avg
        flags_avg[bl_idx] = flag_avg
        ant1_bl[bl_idx] = a1
        ant2_bl[bl_idx] = a2
        time_bl[bl_idx] = np.mean(t)

    return vis_obs_avg, vis_model_avg, ant1_bl, ant2_bl, time_bl, flags_avg


def average_visibilities(vis_obs: np.ndarray, flags: np.ndarray, jones_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Legacy function for old solver architecture.

    This is a compatibility stub for the old CalibrationSolver.
    New code should use average_time_freq or average_per_baseline.
    """
    # Simple averaging for backward compatibility
    # Determine averaging based on Jones type
    if jones_type.upper() == 'K':
        # K: average time, keep freq
        avg_time, avg_freq = True, False
    else:
        # G, B, D, Xf: average both
        avg_time, avg_freq = True, True

    # Mask flagged data
    vis_masked = np.where(flags, np.nan, vis_obs)

    # Determine axes
    axes = []
    if avg_time and vis_obs.ndim >= 3:
        axes.append(0)
    if avg_freq and vis_obs.ndim >= 3:
        axes.append(1)

    if axes:
        with np.errstate(invalid='ignore'):
            vis_avg = np.nanmean(vis_masked, axis=tuple(axes))
        flags_avg = np.all(flags, axis=tuple(axes))
    else:
        vis_avg = vis_obs
        flags_avg = flags

    return vis_avg, flags_avg


def average_model_visibilities(vis_model: np.ndarray, jones_type: str) -> np.ndarray:
    """
    Legacy function for old solver architecture.

    This is a compatibility stub for the old CalibrationSolver.
    New code should use average_time_freq or average_per_baseline.
    """
    # Determine averaging based on Jones type
    if jones_type.upper() == 'K':
        # K: average time, keep freq
        avg_time, avg_freq = True, False
    else:
        # G, B, D, Xf: average both
        avg_time, avg_freq = True, True

    # Determine axes
    axes = []
    if avg_time and vis_model.ndim >= 3:
        axes.append(0)
    if avg_freq and vis_model.ndim >= 3:
        axes.append(1)

    if axes:
        vis_avg = np.mean(vis_model, axis=tuple(axes))
    else:
        vis_avg = vis_model

    return vis_avg


__all__ = [
    'average_time_freq',
    'average_per_baseline',
    'average_visibilities',
    'average_model_visibilities'
]
