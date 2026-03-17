"""ALAKAZAM v1 RFI Flagging.

MAD-based threshold flagging on visibility amplitudes.
Flags on ALL 4 correlations independently — any correlation
that exceeds the threshold flags that entire sample (all 4 corrs).
Respects existing flags — only adds new flags, never removes.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
import logging

logger = logging.getLogger("alakazam")


def flag_rfi(vis: np.ndarray, flags: np.ndarray,
             threshold: float = 5.0) -> np.ndarray:
    """Flag visibilities with amplitudes > threshold * MAD above median.

    Checks all 4 correlations (pp, pq, qp, qq) independently.
    If ANY correlation at a given (row, chan) is an outlier,
    ALL 4 correlations at that (row, chan) are flagged.

    vis:   (n_row, n_chan, 2, 2) complex128
    flags: (n_row, n_chan, 2, 2) bool — existing flags (respected)

    Returns updated flags array.
    """
    n_new_total = 0

    # Check each correlation independently
    bad_any = np.zeros(vis.shape[:2], dtype=bool)  # (n_row, n_chan)

    for p in range(2):
        for q in range(2):
            amp = np.abs(vis[..., p, q])
            existing = flags[..., p, q]
            amp_clean = amp[~existing]

            if len(amp_clean) < 10:
                continue

            med = np.median(amp_clean)
            mad = np.median(np.abs(amp_clean - med))
            if mad < 1e-30:
                continue

            cutoff = med + threshold * 1.4826 * mad
            bad = amp > cutoff
            # Only flag samples not already flagged
            bad = bad & (~existing)
            bad_any |= bad

    # Flag all 4 correlations wherever any was bad
    if bad_any.any():
        n_new_total = int(bad_any.sum())
        flags[bad_any] = True  # broadcasts to all 4 corrs
        logger.debug(f"RFI: flagged {n_new_total} new samples "
                     f"(threshold={threshold})")

    return flags
