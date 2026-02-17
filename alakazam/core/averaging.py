"""
ALAKAZAM Averaging.

Flag-aware averaging per baseline. Numba JIT for performance.
Handles both time-only and time+freq averaging.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
from numba import njit, prange
from typing import Tuple


@njit(cache=True)
def _build_baseline_map(ant1, ant2, n_ant):
    """Build baseline index map. Returns (bl_map, ant1_out, ant2_out, n_bl)."""
    n_row = len(ant1)
    n_bl_max = n_ant * (n_ant - 1) // 2 + n_ant
    bl_map = -np.ones((n_ant, n_ant), dtype=np.int32)
    ant1_out = np.zeros(n_bl_max, dtype=np.int32)
    ant2_out = np.zeros(n_bl_max, dtype=np.int32)
    n_bl = 0

    for row in range(n_row):
        a1 = min(ant1[row], ant2[row])
        a2 = max(ant1[row], ant2[row])
        if a1 == a2:
            continue  # skip autocorrelations
        if bl_map[a1, a2] < 0:
            bl_map[a1, a2] = n_bl
            ant1_out[n_bl] = a1
            ant2_out[n_bl] = a2
            n_bl += 1

    return bl_map, ant1_out[:n_bl], ant2_out[:n_bl], n_bl


@njit(parallel=True, cache=True)
def average_per_baseline_time_only(
    vis, flags, ant1, ant2, n_ant
) -> Tuple:
    """Average visibilities per baseline over time, keeping frequency axis.

    vis:   (n_row, n_freq, 2, 2) complex128
    flags: (n_row, n_freq, 2, 2) bool
    ant1:  (n_row,) int32
    ant2:  (n_row,) int32

    Returns: (vis_avg, flags_avg, ant1_out, ant2_out)
        vis_avg:   (n_bl, n_freq, 2, 2)
        flags_avg: (n_bl, n_freq, 2, 2)
    """
    n_row = vis.shape[0]
    n_freq = vis.shape[1]
    bl_map, a1_out, a2_out, n_bl = _build_baseline_map(ant1, ant2, n_ant)

    vis_sum = np.zeros((n_bl, n_freq, 2, 2), dtype=np.complex128)
    counts = np.zeros((n_bl, n_freq, 2, 2), dtype=np.float64)

    for row in range(n_row):
        a1 = min(ant1[row], ant2[row])
        a2 = max(ant1[row], ant2[row])
        if a1 == a2:
            continue
        bl_idx = bl_map[a1, a2]
        if bl_idx < 0:
            continue
        for f in range(n_freq):
            for i in range(2):
                for j in range(2):
                    if not flags[row, f, i, j]:
                        vis_sum[bl_idx, f, i, j] += vis[row, f, i, j]
                        counts[bl_idx, f, i, j] += 1.0

    vis_avg = np.zeros((n_bl, n_freq, 2, 2), dtype=np.complex128)
    flags_avg = np.zeros((n_bl, n_freq, 2, 2), dtype=np.bool_)

    for bl in prange(n_bl):
        for f in range(n_freq):
            for i in range(2):
                for j in range(2):
                    if counts[bl, f, i, j] > 0:
                        vis_avg[bl, f, i, j] = vis_sum[bl, f, i, j] / counts[bl, f, i, j]
                    else:
                        flags_avg[bl, f, i, j] = True

    return vis_avg, flags_avg, a1_out, a2_out


@njit(parallel=True, cache=True)
def average_per_baseline_full(
    vis, flags, ant1, ant2, n_ant
) -> Tuple:
    """Average visibilities per baseline over time AND frequency.

    vis:   (n_row, n_freq, 2, 2) complex128
    flags: (n_row, n_freq, 2, 2) bool

    Returns: (vis_avg, flags_avg, ant1_out, ant2_out)
        vis_avg:   (n_bl, 2, 2)
        flags_avg: (n_bl, 2, 2)
    """
    n_row = vis.shape[0]
    n_freq = vis.shape[1]
    bl_map, a1_out, a2_out, n_bl = _build_baseline_map(ant1, ant2, n_ant)

    vis_sum = np.zeros((n_bl, 2, 2), dtype=np.complex128)
    counts = np.zeros((n_bl, 2, 2), dtype=np.float64)

    for row in range(n_row):
        a1 = min(ant1[row], ant2[row])
        a2 = max(ant1[row], ant2[row])
        if a1 == a2:
            continue
        bl_idx = bl_map[a1, a2]
        if bl_idx < 0:
            continue
        for f in range(n_freq):
            for i in range(2):
                for j in range(2):
                    if not flags[row, f, i, j]:
                        vis_sum[bl_idx, i, j] += vis[row, f, i, j]
                        counts[bl_idx, i, j] += 1.0

    vis_avg = np.zeros((n_bl, 2, 2), dtype=np.complex128)
    flags_avg = np.zeros((n_bl, 2, 2), dtype=np.bool_)

    for bl in prange(n_bl):
        for i in range(2):
            for j in range(2):
                if counts[bl, i, j] > 0:
                    vis_avg[bl, i, j] = vis_sum[bl, i, j] / counts[bl, i, j]
                else:
                    flags_avg[bl, i, j] = True

    return vis_avg, flags_avg, a1_out, a2_out
