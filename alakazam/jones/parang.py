"""ALAKAZAM Parallactic Angle.

Compute parallactic angles from MS metadata and convert to Jones matrices.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
from numba import njit, prange
from typing import Optional

from .algebra import FeedBasis


def compute_parallactic_angles(
    ms_path: str,
    unique_times: np.ndarray,
    field: Optional[str] = None,
) -> np.ndarray:
    """Compute parallactic angle per antenna per timestep.

    Parameters
    ----------
    ms_path      : str   — path to Measurement Set
    unique_times : (n_time,) float64  — MJD seconds
    field        : str, optional — field name (uses first field if None)

    Returns
    -------
    parang : (n_time, n_ant) float64 — radians
    """
    from casacore.tables import table
    from casacore.measures import measures
    from casacore.quanta import quantity

    dm = measures()

    ant_tab = table(f"{ms_path}::ANTENNA", readonly=True, ack=False)
    ant_pos = ant_tab.getcol("POSITION")  # (n_ant, 3) ITRF metres
    n_ant   = ant_tab.nrows()
    ant_tab.close()

    field_tab  = table(f"{ms_path}::FIELD", readonly=True, ack=False)
    field_names = list(field_tab.getcol("NAME"))
    fid = 0
    if field is not None and field in field_names:
        fid = field_names.index(field)
    phase_dir = field_tab.getcol("PHASE_DIR")[fid]  # (1, 2) or (n_poly, 2)
    field_tab.close()

    src_dir = dm.direction(
        "J2000",
        quantity(phase_dir[0, 0], "rad"),
        quantity(phase_dir[0, 1], "rad"),
    )

    n_time = len(unique_times)
    parang = np.zeros((n_time, n_ant), dtype=np.float64)

    for t in range(n_time):
        dm.do_frame(dm.epoch("UTC", quantity(unique_times[t] / 86400.0, "d")))
        for a in range(n_ant):
            pos = dm.position(
                "ITRF",
                quantity(ant_pos[a, 0], "m"),
                quantity(ant_pos[a, 1], "m"),
                quantity(ant_pos[a, 2], "m"),
            )
            dm.do_frame(pos)
            parang[t, a] = dm.posangle(src_dir, dm.direction("ZENITH")).get_value("rad")

    return parang


@njit(parallel=True, cache=True)
def parang_to_jones_linear(parang_ant):
    """Parallactic angle → Jones rotation for LINEAR feeds.

    parang_ant: (n_ant,) float64 — radians
    Returns:    (n_ant, 2, 2) complex128
    """
    n_ant = parang_ant.shape[0]
    P = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    for a in prange(n_ant):
        c = np.cos(parang_ant[a])
        s = np.sin(parang_ant[a])
        P[a, 0, 0] =  c
        P[a, 0, 1] = -s
        P[a, 1, 0] =  s
        P[a, 1, 1] =  c
    return P


@njit(parallel=True, cache=True)
def parang_to_jones_circular(parang_ant):
    """Parallactic angle → Jones for CIRCULAR feeds.

    parang_ant: (n_ant,) float64 — radians
    Returns:    (n_ant, 2, 2) complex128
    """
    n_ant = parang_ant.shape[0]
    P = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    for a in prange(n_ant):
        P[a, 0, 0] = np.exp(-1j * parang_ant[a])
        P[a, 1, 1] = np.exp( 1j * parang_ant[a])
    return P


def parang_to_jones(parang_ant: np.ndarray, feed_basis: FeedBasis) -> np.ndarray:
    """Dispatch to linear or circular parang Jones builder."""
    if feed_basis == FeedBasis.CIRCULAR:
        return parang_to_jones_circular(parang_ant)
    return parang_to_jones_linear(parang_ant)
