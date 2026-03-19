"""ALAKAZAM v1 Averaging.

Operates on RAW format (n_row, n_chan, n_corr) — no 2x2 conversion needed.
Two modes:
  average_full     → (n_bl, n_corr)          time+freq averaged
  average_time_only → (n_bl, n_chan, n_corr)  time averaged, freq kept

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _build_bl_map(n_ant):
    n_bl = (n_ant * (n_ant - 1)) // 2
    bl_map = np.full((n_ant, n_ant), -1, dtype=np.int64)
    a1 = np.empty(n_bl, dtype=np.int32)
    a2 = np.empty(n_bl, dtype=np.int32)
    idx = 0
    for i in range(n_ant):
        for j in range(i + 1, n_ant):
            bl_map[i, j] = idx
            a1[idx] = i; a2[idx] = j
            idx += 1
    return bl_map, a1, a2, n_bl


@njit(parallel=True, cache=True)
def average_per_baseline_full(vis, flags, ant1, ant2, n_ant):
    """Average time+freq → (n_bl, n_corr).
    vis: (n_row, n_chan, n_corr), flags: (n_row, n_chan, n_corr)."""
    bl_map, oa1, oa2, n_bl = _build_bl_map(n_ant)
    n_corr = vis.shape[2]
    avg = np.zeros((n_bl, n_corr), dtype=np.complex128)
    wt = np.zeros((n_bl, n_corr), dtype=np.float64)

    nr, nc = vis.shape[0], vis.shape[1]
    for r in prange(nr):
        a1, a2 = ant1[r], ant2[r]
        if a1 >= a2: continue
        bl = bl_map[a1, a2]
        if bl < 0: continue
        for c in range(nc):
            for p in range(n_corr):
                if not flags[r, c, p]:
                    avg[bl, p] += vis[r, c, p]
                    wt[bl, p] += 1.0

    for bl in prange(n_bl):
        for p in range(n_corr):
            if wt[bl, p] > 0:
                avg[bl, p] /= wt[bl, p]

    return avg, oa1, oa2


@njit(parallel=True, cache=True)
def average_per_baseline_time_only(vis, flags, ant1, ant2, n_ant):
    """Average time only, keep freq → (n_bl, n_chan, n_corr).
    vis: (n_row, n_chan, n_corr), flags: (n_row, n_chan, n_corr)."""
    bl_map, oa1, oa2, n_bl = _build_bl_map(n_ant)
    nc = vis.shape[1]
    n_corr = vis.shape[2]
    avg = np.zeros((n_bl, nc, n_corr), dtype=np.complex128)
    wt = np.zeros((n_bl, nc, n_corr), dtype=np.float64)

    nr = vis.shape[0]
    for r in prange(nr):
        a1, a2 = ant1[r], ant2[r]
        if a1 >= a2: continue
        bl = bl_map[a1, a2]
        if bl < 0: continue
        for c in range(nc):
            for p in range(n_corr):
                if not flags[r, c, p]:
                    avg[bl, c, p] += vis[r, c, p]
                    wt[bl, c, p] += 1.0

    for bl in prange(n_bl):
        for c in range(nc):
            for p in range(n_corr):
                if wt[bl, c, p] > 0:
                    avg[bl, c, p] /= wt[bl, c, p]

    return avg, oa1, oa2


@njit(cache=True)
def accumulate_baselines_freqdep(vf, mf, ff, ant1, ant2, n_ant,
                                  sum_v, sum_m, count, bl_map):
    """Accumulate per-baseline sums for freq-dependent running average (Tier 3).
    vf/mf: (n_row, n_chan, n_corr), ff: (n_row, n_chan, n_corr) bool.
    sum_v/sum_m/count: (n_bl, n_chan, n_corr) — mutated in-place."""
    nr = vf.shape[0]
    nc = vf.shape[1]
    np_ = vf.shape[2]
    for r in range(nr):
        a1, a2 = ant1[r], ant2[r]
        if a1 >= a2:
            continue
        bl = bl_map[a1, a2]
        if bl < 0:
            continue
        for c in range(nc):
            for p in range(np_):
                if not ff[r, c, p]:
                    sum_v[bl, c, p] += vf[r, c, p]
                    sum_m[bl, c, p] += mf[r, c, p]
                    count[bl, c, p] += 1.0


@njit(cache=True)
def accumulate_baselines_full(vf, mf, ff, ant1, ant2, n_ant,
                               sum_v, sum_m, count, bl_map):
    """Accumulate per-baseline sums for full (time+freq) running average (Tier 3).
    vf/mf: (n_row, n_chan, n_corr), ff: (n_row, n_chan, n_corr) bool.
    sum_v/sum_m/count: (n_bl, n_corr) — mutated in-place."""
    nr = vf.shape[0]
    nc = vf.shape[1]
    np_ = vf.shape[2]
    for r in range(nr):
        a1, a2 = ant1[r], ant2[r]
        if a1 >= a2:
            continue
        bl = bl_map[a1, a2]
        if bl < 0:
            continue
        for c in range(nc):
            for p in range(np_):
                if not ff[r, c, p]:
                    sum_v[bl, p] += vf[r, c, p]
                    sum_m[bl, p] += mf[r, c, p]
                    count[bl, p] += 1.0
