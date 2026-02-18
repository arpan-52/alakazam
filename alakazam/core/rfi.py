"""ALAKAZAM RFI Flagging.

Simple MAD-based threshold flagging on visibility amplitudes.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
import logging

logger = logging.getLogger("alakazam")


def flag_rfi(
    vis: np.ndarray,    # (..., 2, 2) complex128
    flags: np.ndarray,  # (..., 2, 2) bool — existing flags (in-place update)
    threshold: float = 5.0,
) -> np.ndarray:
    """Flag visibilities with amplitudes > threshold * MAD above median.

    Returns updated flags array.
    """
    amp = np.abs(vis[..., 0, 0])  # use pp correlation as proxy
    amp_clean = amp[~flags[..., 0, 0]]

    if len(amp_clean) < 10:
        return flags

    med = np.median(amp_clean)
    mad = np.median(np.abs(amp_clean - med))
    if mad < 1e-30:
        return flags

    bad = amp > (med + threshold * 1.4826 * mad)
    flags[bad] = True
    n_new = int(bad.sum())
    if n_new > 0:
        logger.debug(f"RFI: flagged {n_new} new samples (threshold={threshold})")
    return flags
