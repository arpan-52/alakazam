"""ALAKAZAM Averaging.

Average visibilities over time and/or frequency for solver input.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
from numba import njit, prange
import logging

logger = logging.getLogger("alakazam")


@njit(parallel=True, cache=True)
def average_per_baseline_full(
    vis: np.ndarray,      # (n_row, n_chan, 2, 2) complex128
    flags: np.ndarray,    # (n_row, n_chan, 2, 2) bool
    ant1: np.ndarray,     # (n_row,) int32
    ant2: np.ndarray,
    n_ant: int,
) -> tuple:
    """Average all rows per unique baseline (time+freq average).

    Returns: (avg_vis, avg_flags, out_ant1, out_ant2)
      avg_vis: (n_bl, 2, 2)
      avg_flags: (n_bl, 2, 2) bool — True if all flagged
    """
    # Build baseline index map
    n_bl = (n_ant * (n_ant - 1)) // 2
    bl_map = np.full((n_ant, n_ant), -1, dtype=np.int64)
    out_ant1 = np.empty(n_bl, dtype=np.int32)
    out_ant2 = np.empty(n_bl, dtype=np.int32)
    idx = 0
    for i in range(n_ant):
        for j in range(i + 1, n_ant):
            bl_map[i, j] = idx
            out_ant1[idx] = i
            out_ant2[idx] = j
            idx += 1

    avg_vis   = np.zeros((n_bl, 2, 2), dtype=np.complex128)
    avg_wt    = np.zeros((n_bl, 2, 2), dtype=np.float64)
    avg_flags = np.ones((n_bl, 2, 2), dtype=np.bool_)

    n_row  = vis.shape[0]
    n_chan = vis.shape[1]

    for r in prange(n_row):
        a1 = ant1[r]
        a2 = ant2[r]
        if a1 >= a2:
            continue
        bl = bl_map[a1, a2]
        if bl < 0:
            continue
        for c in range(n_chan):
            for p in range(2):
                for q in range(2):
                    if not flags[r, c, p, q]:
                        avg_vis[bl, p, q]   += vis[r, c, p, q]
                        avg_wt[bl, p, q]    += 1.0
                        avg_flags[bl, p, q]  = False

    for bl in prange(n_bl):
        for p in range(2):
            for q in range(2):
                if avg_wt[bl, p, q] > 0:
                    avg_vis[bl, p, q] /= avg_wt[bl, p, q]

    return avg_vis, avg_flags, out_ant1, out_ant2


@njit(parallel=True, cache=True)
def average_per_baseline_time_only(
    vis: np.ndarray,      # (n_row, n_chan, 2, 2) complex128
    flags: np.ndarray,
    ant1: np.ndarray,
    ant2: np.ndarray,
    n_ant: int,
) -> tuple:
    """Average all rows per unique baseline in time only (keep freq axis).

    Returns: (avg_vis, avg_flags, out_ant1, out_ant2)
      avg_vis: (n_bl, n_chan, 2, 2)
    """
    n_bl = (n_ant * (n_ant - 1)) // 2
    bl_map = np.full((n_ant, n_ant), -1, dtype=np.int64)
    out_ant1 = np.empty(n_bl, dtype=np.int32)
    out_ant2 = np.empty(n_bl, dtype=np.int32)
    idx = 0
    for i in range(n_ant):
        for j in range(i + 1, n_ant):
            bl_map[i, j] = idx
            out_ant1[idx] = i
            out_ant2[idx] = j
            idx += 1

    n_chan = vis.shape[1]
    avg_vis   = np.zeros((n_bl, n_chan, 2, 2), dtype=np.complex128)
    avg_wt    = np.zeros((n_bl, n_chan, 2, 2), dtype=np.float64)
    avg_flags = np.ones((n_bl, n_chan, 2, 2), dtype=np.bool_)

    n_row = vis.shape[0]
    for r in prange(n_row):
        a1 = ant1[r]
        a2 = ant2[r]
        if a1 >= a2:
            continue
        bl = bl_map[a1, a2]
        if bl < 0:
            continue
        for c in range(n_chan):
            for p in range(2):
                for q in range(2):
                    if not flags[r, c, p, q]:
                        avg_vis[bl, c, p, q]   += vis[r, c, p, q]
                        avg_wt[bl, c, p, q]    += 1.0
                        avg_flags[bl, c, p, q]  = False

    for bl in prange(n_bl):
        for c in range(n_chan):
            for p in range(2):
                for q in range(2):
                    if avg_wt[bl, c, p, q] > 0:
                        avg_vis[bl, c, p, q] /= avg_wt[bl, c, p, q]

    return avg_vis, avg_flags, out_ant1, out_ant2
