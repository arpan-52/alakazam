"""
Jones Interpolation Module.

Interpolates Jones solutions to different time/frequency grids.
Used when applying pre-solved Jones with different solution intervals.
"""

import numpy as np
from typing import Tuple, Literal
from scipy.interpolate import interp1d
import logging

logger = logging.getLogger('ALAKAZAM')

InterpMethod = Literal['nearest', 'linear', 'cubic']


def interpolate_jones_time(
    jones: np.ndarray,
    time_src: np.ndarray,
    time_tgt: float,
    method: InterpMethod = 'linear'
) -> np.ndarray:
    """
    Interpolate Jones matrices in time dimension.

    Parameters
    ----------
    jones : ndarray
        Source Jones matrices (n_time_src, n_freq, n_ant, 2, 2)
    time_src : ndarray
        Source time points (n_time_src,)
    time_tgt : float
        Target time point (scalar)
    method : str
        Interpolation method: 'nearest', 'linear', 'cubic'

    Returns
    -------
    jones_interp : ndarray
        Interpolated Jones (n_freq, n_ant, 2, 2)
    """
    n_time_src, n_freq, n_ant = jones.shape[:3]

    if n_time_src == 1:
        # Only one time point, return it
        return jones[0]

    if method == 'nearest':
        # Find nearest time
        idx = np.argmin(np.abs(time_src - time_tgt))
        return jones[idx]

    elif method in ('linear', 'cubic'):
        # Interpolate real and imaginary parts separately
        jones_interp = np.zeros((n_freq, n_ant, 2, 2), dtype=np.complex128)

        for f in range(n_freq):
            for a in range(n_ant):
                for i in range(2):
                    for j in range(2):
                        # Real part
                        f_real = interp1d(
                            time_src,
                            jones[:, f, a, i, j].real,
                            kind=method,
                            bounds_error=False,
                            fill_value='extrapolate'
                        )

                        # Imaginary part
                        f_imag = interp1d(
                            time_src,
                            jones[:, f, a, i, j].imag,
                            kind=method,
                            bounds_error=False,
                            fill_value='extrapolate'
                        )

                        jones_interp[f, a, i, j] = f_real(time_tgt) + 1j * f_imag(time_tgt)

        return jones_interp

    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def interpolate_jones_time_array(
    jones: np.ndarray,
    time_src: np.ndarray,
    time_tgt: np.ndarray,
    method: InterpMethod = 'linear'
) -> np.ndarray:
    """
    Interpolate Jones matrices to multiple target times.

    Parameters
    ----------
    jones : ndarray
        Source Jones matrices (n_time_src, n_ant, 2, 2) or (n_time_src, n_freq, n_ant, 2, 2)
    time_src : ndarray
        Source time points (n_time_src,)
    time_tgt : ndarray
        Target time points (n_time_tgt,)
    method : str
        Interpolation method: 'nearest', 'linear', 'cubic'

    Returns
    -------
    jones_interp : ndarray
        Interpolated Jones (n_time_tgt, n_ant, 2, 2) or (n_time_tgt, n_freq, n_ant, 2, 2)
    """
    n_time_src = jones.shape[0]
    n_time_tgt = len(time_tgt)

    if n_time_src == 1:
        # Only one time point, replicate
        return np.tile(jones, (n_time_tgt,) + (1,) * (jones.ndim - 1))

    if n_time_src == n_time_tgt and np.allclose(time_src, time_tgt):
        # Times match
        return jones

    if method == 'nearest':
        # Find nearest times
        jones_interp = np.zeros((n_time_tgt,) + jones.shape[1:], dtype=np.complex128)
        for t_idx, t in enumerate(time_tgt):
            src_idx = np.argmin(np.abs(time_src - t))
            jones_interp[t_idx] = jones[src_idx]
        return jones_interp

    elif method in ('linear', 'cubic'):
        # Handle both (n_time, n_ant, 2, 2) and (n_time, n_freq, n_ant, 2, 2)
        if jones.ndim == 4:
            # (n_time_src, n_ant, 2, 2)
            n_ant = jones.shape[1]
            jones_interp = np.zeros((n_time_tgt, n_ant, 2, 2), dtype=np.complex128)

            for a in range(n_ant):
                for i in range(2):
                    for j in range(2):
                        # Real part
                        f_real = interp1d(
                            time_src,
                            jones[:, a, i, j].real,
                            kind=method,
                            bounds_error=False,
                            fill_value='extrapolate'
                        )

                        # Imaginary part
                        f_imag = interp1d(
                            time_src,
                            jones[:, a, i, j].imag,
                            kind=method,
                            bounds_error=False,
                            fill_value='extrapolate'
                        )

                        jones_interp[:, a, i, j] = f_real(time_tgt) + 1j * f_imag(time_tgt)

        elif jones.ndim == 5:
            # (n_time_src, n_freq, n_ant, 2, 2)
            n_freq, n_ant = jones.shape[1:3]
            jones_interp = np.zeros((n_time_tgt, n_freq, n_ant, 2, 2), dtype=np.complex128)

            for f in range(n_freq):
                for a in range(n_ant):
                    for i in range(2):
                        for j in range(2):
                            # Real part
                            f_real = interp1d(
                                time_src,
                                jones[:, f, a, i, j].real,
                                kind=method,
                                bounds_error=False,
                                fill_value='extrapolate'
                            )

                            # Imaginary part
                            f_imag = interp1d(
                                time_src,
                                jones[:, f, a, i, j].imag,
                                kind=method,
                                bounds_error=False,
                                fill_value='extrapolate'
                            )

                            jones_interp[:, f, a, i, j] = f_real(time_tgt) + 1j * f_imag(time_tgt)

        else:
            raise ValueError(f"Unexpected Jones shape for time interpolation: {jones.shape}")

        return jones_interp

    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def interpolate_jones_freq(
    jones: np.ndarray,
    freq_src: np.ndarray,
    freq_tgt: np.ndarray,
    method: InterpMethod = 'linear'
) -> np.ndarray:
    """
    Interpolate Jones matrices in frequency dimension.

    Parameters
    ----------
    jones : ndarray
        Source Jones matrices (n_freq_src, n_ant, 2, 2)
    freq_src : ndarray
        Source frequency points (n_freq_src,)
    freq_tgt : ndarray
        Target frequency points (n_freq_tgt,)
    method : str
        Interpolation method: 'nearest', 'linear', 'cubic'

    Returns
    -------
    jones_interp : ndarray
        Interpolated Jones (n_freq_tgt, n_ant, 2, 2)
    """
    n_freq_src, n_ant = jones.shape[:2]
    n_freq_tgt = len(freq_tgt)

    if n_freq_src == 1:
        # Only one frequency, replicate
        return np.tile(jones, (n_freq_tgt, 1, 1, 1))

    if n_freq_src == n_freq_tgt and np.allclose(freq_src, freq_tgt):
        # Frequencies match
        return jones

    if method == 'nearest':
        # Find nearest frequencies
        jones_interp = np.zeros((n_freq_tgt, n_ant, 2, 2), dtype=np.complex128)
        for f_idx, f in enumerate(freq_tgt):
            src_idx = np.argmin(np.abs(freq_src - f))
            jones_interp[f_idx] = jones[src_idx]

        return jones_interp

    elif method in ('linear', 'cubic'):
        # Interpolate real and imaginary parts separately
        jones_interp = np.zeros((n_freq_tgt, n_ant, 2, 2), dtype=np.complex128)

        for a in range(n_ant):
            for i in range(2):
                for j in range(2):
                    # Real part
                    f_real = interp1d(
                        freq_src,
                        jones[:, a, i, j].real,
                        kind=method,
                        bounds_error=False,
                        fill_value='extrapolate'
                    )

                    # Imaginary part
                    f_imag = interp1d(
                        freq_src,
                        jones[:, a, i, j].imag,
                        kind=method,
                        bounds_error=False,
                        fill_value='extrapolate'
                    )

                    jones_interp[:, a, i, j] = f_real(freq_tgt) + 1j * f_imag(freq_tgt)

        return jones_interp

    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def interpolate_jones_to_chunk(
    jones: np.ndarray,
    time_src: np.ndarray,
    freq_src: np.ndarray,
    time_tgt: float,
    freq_tgt: np.ndarray,
    method_time: InterpMethod = 'linear',
    method_freq: InterpMethod = 'linear'
) -> np.ndarray:
    """
    Interpolate Jones to match a specific data chunk.

    This is used when applying pre-solved Jones with different solution
    intervals to a chunk with its own time/frequency grid.

    Parameters
    ----------
    jones : ndarray
        Source Jones (n_time_src, n_freq_src, n_ant, 2, 2)
    time_src : ndarray
        Source time points (n_time_src,)
    freq_src : ndarray
        Source frequency points (n_freq_src,)
    time_tgt : float
        Target time (scalar, e.g., midpoint of chunk)
    freq_tgt : ndarray
        Target frequencies (n_freq_tgt,)
    method_time : str
        Time interpolation method
    method_freq : str
        Frequency interpolation method

    Returns
    -------
    jones_interp : ndarray
        Interpolated Jones (n_freq_tgt, n_ant, 2, 2)
    """
    # Interpolate in time first
    jones_time = interpolate_jones_time(jones, time_src, time_tgt, method_time)

    # Then interpolate in frequency
    jones_interp = interpolate_jones_freq(jones_time, freq_src, freq_tgt, method_freq)

    return jones_interp


def get_chunk_time_center(time_range: Tuple[float, float]) -> float:
    """
    Get center time of a chunk.

    Parameters
    ----------
    time_range : tuple
        (t_start, t_end)

    Returns
    -------
    t_center : float
        Center time
    """
    return (time_range[0] + time_range[1]) / 2.0


def create_source_time_grid(n_sol_time: int, time_range_total: Tuple[float, float]) -> np.ndarray:
    """
    Create time grid for source Jones solutions.

    Returns midpoints of each solution interval.

    Parameters
    ----------
    n_sol_time : int
        Number of solution intervals
    time_range_total : tuple
        Total (t_min, t_max)

    Returns
    -------
    time_grid : ndarray
        Time points (n_sol_time,) at midpoints of intervals
    """
    t_min, t_max = time_range_total

    if n_sol_time == 1:
        return np.array([(t_min + t_max) / 2.0])

    edges = np.linspace(t_min, t_max, n_sol_time + 1)
    midpoints = (edges[:-1] + edges[1:]) / 2.0

    return midpoints


def create_source_freq_grid(n_sol_freq: int, freq_array: np.ndarray) -> np.ndarray:
    """
    Create frequency grid for source Jones solutions.

    Returns center frequencies of each solution interval.

    Parameters
    ----------
    n_sol_freq : int
        Number of solution intervals in frequency
    freq_array : ndarray
        Full frequency array

    Returns
    -------
    freq_grid : ndarray
        Frequency points (n_sol_freq,) at centers of intervals
    """
    n_freq = len(freq_array)

    if n_sol_freq == 1:
        return np.array([np.median(freq_array)])

    n_chan_per_chunk = n_freq // n_sol_freq
    freq_centers = []

    for i in range(n_sol_freq):
        f_start = i * n_chan_per_chunk
        f_end = f_start + n_chan_per_chunk if i < n_sol_freq - 1 else n_freq
        freq_centers.append(np.median(freq_array[f_start:f_end]))

    return np.array(freq_centers)


def fill_flagged_from_valid(
    jones: np.ndarray,
    weights: np.ndarray,
    time: np.ndarray = None,
    freq: np.ndarray = None,
    method: InterpMethod = 'linear'
) -> np.ndarray:
    """
    Fill flagged (weight=0) Jones chunks by interpolating from valid (weight>0) neighbors.

    Interpolates in both time and frequency dimensions to fill bad chunks.

    Parameters
    ----------
    jones : ndarray
        Jones matrices, shape (n_time, n_freq, n_ant, 2, 2)
    weights : ndarray
        Weights array, shape (n_time, n_freq), 1.0=valid, 0.0=flagged
    time : ndarray, optional
        Time coordinates (n_time,), for proper interpolation
    freq : ndarray, optional
        Frequency coordinates (n_freq,), for proper interpolation
    method : str
        Interpolation method: 'nearest', 'linear', 'cubic'

    Returns
    -------
    jones_filled : ndarray
        Jones with flagged chunks filled by interpolation
    """
    jones_filled = jones.copy()
    n_time, n_freq, n_ant = jones.shape[:3]

    # If all weights are 1, nothing to do
    if np.all(weights > 0):
        return jones_filled

    # Time interpolation: for each freq, interpolate across time
    if n_time > 1 and time is not None:
        for f in range(n_freq):
            # Find valid time points for this freq
            valid_t = weights[:, f] > 0
            if np.sum(valid_t) == 0:
                # No valid points at this freq, skip
                continue
            if np.sum(valid_t) == n_time:
                # All valid, skip
                continue

            valid_time = time[valid_t]

            # Interpolate each antenna and matrix element
            for a in range(n_ant):
                for i in range(2):
                    for j in range(2):
                        valid_vals = jones[valid_t, f, a, i, j]

                        # Interpolate real and imaginary separately
                        if method == 'nearest':
                            # Find nearest valid point for each invalid point
                            for t in range(n_time):
                                if weights[t, f] == 0:
                                    idx = np.argmin(np.abs(valid_time - time[t]))
                                    jones_filled[t, f, a, i, j] = valid_vals[idx]
                        else:
                            # Use scipy interp1d
                            if len(valid_time) >= 2:
                                f_real = interp1d(valid_time, valid_vals.real, kind=method,
                                                bounds_error=False, fill_value='extrapolate')
                                f_imag = interp1d(valid_time, valid_vals.imag, kind=method,
                                                bounds_error=False, fill_value='extrapolate')

                                # Fill invalid points
                                for t in range(n_time):
                                    if weights[t, f] == 0:
                                        jones_filled[t, f, a, i, j] = f_real(time[t]) + 1j * f_imag(time[t])
                            elif len(valid_time) == 1:
                                # Only one valid point, use it for all invalid
                                for t in range(n_time):
                                    if weights[t, f] == 0:
                                        jones_filled[t, f, a, i, j] = valid_vals[0]

    # Frequency interpolation: for each time, interpolate across freq
    if n_freq > 1 and freq is not None:
        for t in range(n_time):
            # Find valid freq points for this time
            valid_f = weights[t, :] > 0
            if np.sum(valid_f) == 0:
                # No valid points at this time, skip
                continue
            if np.sum(valid_f) == n_freq:
                # All valid, skip
                continue

            valid_freq = freq[valid_f]

            # Interpolate each antenna and matrix element
            for a in range(n_ant):
                for i in range(2):
                    for j in range(2):
                        valid_vals = jones_filled[t, valid_f, a, i, j]

                        # Interpolate real and imaginary separately
                        if method == 'nearest':
                            # Find nearest valid point for each invalid point
                            for f in range(n_freq):
                                if weights[t, f] == 0:
                                    idx = np.argmin(np.abs(valid_freq - freq[f]))
                                    jones_filled[t, f, a, i, j] = valid_vals[idx]
                        else:
                            # Use scipy interp1d
                            if len(valid_freq) >= 2:
                                f_real = interp1d(valid_freq, valid_vals.real, kind=method,
                                                bounds_error=False, fill_value='extrapolate')
                                f_imag = interp1d(valid_freq, valid_vals.imag, kind=method,
                                                bounds_error=False, fill_value='extrapolate')

                                # Fill invalid points
                                for f in range(n_freq):
                                    if weights[t, f] == 0:
                                        jones_filled[t, f, a, i, j] = f_real(freq[f]) + 1j * f_imag(freq[f])
                            elif len(valid_freq) == 1:
                                # Only one valid point, use it for all invalid
                                for f in range(n_freq):
                                    if weights[t, f] == 0:
                                        jones_filled[t, f, a, i, j] = valid_vals[0]

    return jones_filled


def log_interpolation_info(
    jones_type: str,
    source_shape: Tuple[int, int, int],
    target_time: float,
    target_freq_shape: int,
    method_time: str,
    method_freq: str
):
    """
    Log interpolation operation details.

    Parameters
    ----------
    jones_type : str
        Type of Jones being interpolated
    source_shape : tuple
        (n_time_src, n_freq_src, n_ant)
    target_time : float
        Target time
    target_freq_shape : int
        Number of target frequencies
    method_time : str
        Time interpolation method
    method_freq : str
        Frequency interpolation method
    """
    n_time_src, n_freq_src, n_ant = source_shape

    logger.debug(f"  Interpolating {jones_type}: "
                f"({n_time_src}t, {n_freq_src}f, {n_ant}a) -> "
                f"(1t, {target_freq_shape}f, {n_ant}a), "
                f"methods: time={method_time}, freq={method_freq}")


__all__ = [
    'interpolate_jones_time',
    'interpolate_jones_time_array',
    'interpolate_jones_freq',
    'interpolate_jones_to_chunk',
    'fill_flagged_from_valid',
    'get_chunk_time_center',
    'create_source_time_grid',
    'create_source_freq_grid',
    'log_interpolation_info',
    'InterpMethod',
]
