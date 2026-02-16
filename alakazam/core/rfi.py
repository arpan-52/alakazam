"""
ALAKAZAM RFI Flagging.

MAD (Median Absolute Deviation) based outlier detection per baseline.
Numba JIT for performance.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
from numba import njit, prange
from typing import Tuple, Dict
import logging

logger = logging.getLogger("alakazam")

MAD_TO_SIGMA = 1.4826  # MAD × 1.4826 ≈ σ for Gaussian


@njit(cache=True)
def _median_1d(arr):
    """Median of 1D array."""
    s = np.sort(arr)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 0:
        return 0.5 * (s[n // 2 - 1] + s[n // 2])
    return s[n // 2]


@njit(parallel=True, cache=True)
def flag_rfi_mad(vis, flags, threshold=5.0):
    """Flag RFI using MAD per baseline per polarization.

    vis:       (n_bl, n_freq, 2, 2) or (n_bl, 2, 2) complex128
    flags:     same shape, bool
    threshold: sigma threshold

    Returns: updated flags (same shape, bool)
    """
    new_flags = flags.copy()

    if vis.ndim == 4:
        n_bl, n_freq = vis.shape[0], vis.shape[1]
        for bl in prange(n_bl):
            for pi in range(2):
                for pj in range(2):
                    # Collect unflagged amplitudes across frequency
                    amps = np.empty(n_freq, dtype=np.float64)
                    count = 0
                    for f in range(n_freq):
                        if not flags[bl, f, pi, pj]:
                            amps[count] = np.abs(vis[bl, f, pi, pj])
                            count += 1

                    if count < 3:
                        for f in range(n_freq):
                            new_flags[bl, f, pi, pj] = True
                        continue

                    amps_valid = amps[:count]
                    med = _median_1d(amps_valid)
                    deviations = np.abs(amps_valid - med)
                    mad = _median_1d(deviations) * MAD_TO_SIGMA

                    if mad < 1e-20:
                        continue

                    thresh = threshold * mad
                    for f in range(n_freq):
                        if not flags[bl, f, pi, pj]:
                            if np.abs(np.abs(vis[bl, f, pi, pj]) - med) > thresh:
                                new_flags[bl, f, pi, pj] = True

    elif vis.ndim == 3:
        n_bl = vis.shape[0]
        for pi in range(2):
            for pj in range(2):
                amps = np.empty(n_bl, dtype=np.float64)
                count = 0
                for bl in range(n_bl):
                    if not flags[bl, pi, pj]:
                        amps[count] = np.abs(vis[bl, pi, pj])
                        count += 1

                if count < 3:
                    continue

                amps_valid = amps[:count]
                med = _median_1d(amps_valid)
                deviations = np.abs(amps_valid - med)
                mad = _median_1d(deviations) * MAD_TO_SIGMA

                if mad < 1e-20:
                    continue

                thresh = threshold * mad
                for bl in range(n_bl):
                    if not flags[bl, pi, pj]:
                        if np.abs(np.abs(vis[bl, pi, pj]) - med) > thresh:
                            new_flags[bl, pi, pj] = True

    return new_flags


def flag_rfi(vis, flags, threshold=5.0):
    """Flag RFI with logging. Wrapper around flag_rfi_mad.

    Returns: (new_flags, stats_dict)
    """
    n_before = int(np.sum(flags))
    new_flags = flag_rfi_mad(
        np.ascontiguousarray(vis, dtype=np.complex128),
        np.ascontiguousarray(flags, dtype=bool),
        threshold,
    )
    n_after = int(np.sum(new_flags))
    n_new = n_after - n_before
    total = new_flags.size
    stats = {
        "n_before": n_before,
        "n_after": n_after,
        "n_new": n_new,
        "fraction": n_after / total if total > 0 else 0.0,
    }
    return new_flags, stats
