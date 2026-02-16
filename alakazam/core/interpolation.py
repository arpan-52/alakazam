"""
ALAKAZAM Jones Interpolation.

Interpolates Jones solutions from solution grid to target grid.
Uses amplitude/phase interpolation for diagonal Jones (not real/imag).
Uses nearest-neighbor for off-diagonal (leakage) terms.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger("alakazam")


def interpolate_jones_time(
    jones: np.ndarray,
    sol_times: np.ndarray,
    target_times: np.ndarray,
    flags: Optional[np.ndarray] = None,
    method: str = "amp_phase",
) -> np.ndarray:
    """Interpolate Jones matrices in time.

    jones:        (n_sol_time, n_ant, 2, 2) complex128
    sol_times:    (n_sol_time,) float64
    target_times: (n_target,) float64
    flags:        (n_sol_time, n_ant) bool, optional

    Returns: (n_target, n_ant, 2, 2) complex128
    """
    n_sol = jones.shape[0]
    n_ant = jones.shape[1]
    n_target = len(target_times)

    if n_sol == 1:
        return np.broadcast_to(jones[0], (n_target, n_ant, 2, 2)).copy()

    result = np.empty((n_target, n_ant, 2, 2), dtype=np.complex128)

    for a in range(n_ant):
        # Check which solution times are valid for this antenna
        if flags is not None:
            valid = ~flags[:, a]
        else:
            valid = np.isfinite(jones[:, a, 0, 0].real)

        valid_idx = np.where(valid)[0]

        if len(valid_idx) == 0:
            result[:, a] = np.nan + 0j
            continue

        if len(valid_idx) == 1:
            result[:, a] = jones[valid_idx[0], a]
            continue

        valid_times = sol_times[valid_idx]

        for i in range(2):
            for j in range(2):
                vals = jones[valid_idx, a, i, j]

                if method == "amp_phase" and i == j:
                    # Diagonal: interpolate amp and phase separately
                    amp = np.abs(vals)
                    phase = np.unwrap(np.angle(vals))
                    amp_interp = np.interp(target_times, valid_times, amp)
                    phase_interp = np.interp(target_times, valid_times, phase)
                    result[:, a, i, j] = amp_interp * np.exp(1j * phase_interp)
                elif method == "nearest" or i != j:
                    # Off-diagonal or nearest: use nearest neighbor
                    nearest_idx = np.searchsorted(valid_times, target_times, side="right") - 1
                    nearest_idx = np.clip(nearest_idx, 0, len(valid_idx) - 1)
                    # Check if next point is closer
                    for t in range(n_target):
                        ni = nearest_idx[t]
                        if ni < len(valid_idx) - 1:
                            d_left = abs(target_times[t] - valid_times[ni])
                            d_right = abs(target_times[t] - valid_times[ni + 1])
                            if d_right < d_left:
                                nearest_idx[t] = ni + 1
                    result[:, a, i, j] = vals[nearest_idx]
                else:
                    # Linear real/imag (fallback)
                    re_interp = np.interp(target_times, valid_times, vals.real)
                    im_interp = np.interp(target_times, valid_times, vals.imag)
                    result[:, a, i, j] = re_interp + 1j * im_interp

    return result


def interpolate_jones_freq(
    jones: np.ndarray,
    sol_freq_centers: np.ndarray,
    target_freq: np.ndarray,
    method: str = "amp_phase",
) -> np.ndarray:
    """Interpolate Jones matrices in frequency.

    jones:            (n_sol_freq, n_ant, 2, 2) complex128
    sol_freq_centers: (n_sol_freq,) float64  Hz
    target_freq:      (n_target_freq,) float64  Hz

    Returns: (n_target_freq, n_ant, 2, 2) complex128
    """
    n_sol = jones.shape[0]
    n_ant = jones.shape[1]
    n_target = len(target_freq)

    if n_sol == 1:
        return np.broadcast_to(jones[0], (n_target, n_ant, 2, 2)).copy()

    result = np.empty((n_target, n_ant, 2, 2), dtype=np.complex128)

    for a in range(n_ant):
        valid = np.isfinite(jones[:, a, 0, 0].real)
        valid_idx = np.where(valid)[0]

        if len(valid_idx) == 0:
            result[:, a] = np.nan + 0j
            continue

        if len(valid_idx) == 1:
            result[:, a] = jones[valid_idx[0], a]
            continue

        valid_freqs = sol_freq_centers[valid_idx]

        for i in range(2):
            for j in range(2):
                vals = jones[valid_idx, a, i, j]
                if method == "amp_phase" and i == j:
                    amp = np.abs(vals)
                    phase = np.unwrap(np.angle(vals))
                    amp_interp = np.interp(target_freq, valid_freqs, amp)
                    phase_interp = np.interp(target_freq, valid_freqs, phase)
                    result[:, a, i, j] = amp_interp * np.exp(1j * phase_interp)
                else:
                    # Nearest for off-diagonal
                    nearest_idx = np.argmin(
                        np.abs(valid_freqs[:, None] - target_freq[None, :]), axis=0
                    )
                    result[:, a, i, j] = vals[nearest_idx]

    return result


def interpolate_delay_to_freq(
    delay: np.ndarray,
    target_freq: np.ndarray,
) -> np.ndarray:
    """Convert delay parameters directly to Jones at target frequencies.

    No interpolation needed — exact computation from delay model.

    delay:       (n_ant, 2) float64  — nanoseconds
    target_freq: (n_freq,) float64   — Hz
    Returns:     (n_ant, n_freq, 2, 2) complex128
    """
    from ..jones import delay_to_jones
    return delay_to_jones(delay, target_freq)
