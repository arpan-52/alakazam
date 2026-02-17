"""
ALAKAZAM Parallactic Angle Computation.

Computes parallactic angles and constructs P-Jones matrices
for both LINEAR and CIRCULAR feeds.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
from numba import njit, prange
from .algebra import FeedBasis

def compute_parallactic_angles(
    ms_path: str,
    unique_times: np.ndarray,
    field: Optional[str] = None,
) -> np.ndarray:
    """Compute parallactic angle per antenna per timestep.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set.
    unique_times : ndarray (n_time,)
        Unique timestamps in MJD seconds.
    field : str, optional
        Field name (uses first field if None).

    Returns
    -------
    parang : ndarray (n_time, n_ant) in radians
    """
    from casacore.tables import table
    from casacore.measures import measures
    from casacore.quanta import quantity

    dm = measures()

    # Antenna positions
    ant_tab = table(f"{ms_path}::ANTENNA", readonly=True, ack=False)
    ant_pos = ant_tab.getcol("POSITION")  # (n_ant, 3) ITRF metres
    n_ant = ant_tab.nrows()
    ant_tab.close()

    # Source direction
    field_tab = table(f"{ms_path}::FIELD", readonly=True, ack=False)
    if field is not None:
        field_names = list(field_tab.getcol("NAME"))
        if field in field_names:
            fid = field_names.index(field)
        else:
            fid = 0
    else:
        fid = 0
    phase_dir = field_tab.getcol("PHASE_DIR")[fid]  # (1, 2) or (n_poly, 2)
    field_tab.close()

    ra_rad = phase_dir[0, 0]
    dec_rad = phase_dir[0, 1]

    src_dir = dm.direction("J2000",
                           quantity(ra_rad, "rad"),
                           quantity(dec_rad, "rad"))

    n_time = len(unique_times)
    parang = np.zeros((n_time, n_ant), dtype=np.float64)

    for t_idx in range(n_time):
        epoch = dm.epoch("UTC", quantity(unique_times[t_idx] / 86400.0, "d"))
        dm.do_frame(epoch)

        for a_idx in range(n_ant):
            pos = dm.position(
                "ITRF",
                quantity(ant_pos[a_idx, 0], "m"),
                quantity(ant_pos[a_idx, 1], "m"),
                quantity(ant_pos[a_idx, 2], "m"),
            )
            dm.do_frame(pos)
            parang[t_idx, a_idx] = dm.posangle(src_dir, dm.direction("ZENITH")).get_value("rad")

    return parang


@njit(parallel=True, cache=True)
def parang_to_jones_linear(parang_ant):
    """Convert parallactic angles to Jones rotation matrices for LINEAR feeds.

    parang_ant: (n_ant,) float64  — radians
    Returns: (n_ant, 2, 2) complex128
    """
    n_ant = parang_ant.shape[0]
    P = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    for a in prange(n_ant):
        c = np.cos(parang_ant[a])
        s = np.sin(parang_ant[a])
        P[a, 0, 0] = c
        P[a, 0, 1] = -s
        P[a, 1, 0] = s
        P[a, 1, 1] = c
    return P


@njit(parallel=True, cache=True)
def parang_to_jones_circular(parang_ant):
    """Convert parallactic angles to Jones for CIRCULAR feeds.

    parang_ant: (n_ant,) float64  — radians
    Returns: (n_ant, 2, 2) complex128
    """
    n_ant = parang_ant.shape[0]
    P = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    for a in prange(n_ant):
        P[a, 0, 0] = np.exp(-1j * parang_ant[a])
        P[a, 1, 1] = np.exp(1j * parang_ant[a])
    return P


def parang_to_jones(parang_ant: np.ndarray, feed_basis: FeedBasis) -> np.ndarray:
    """Dispatch to linear or circular parang Jones builder."""
    if feed_basis == FeedBasis.CIRCULAR:
        return parang_to_jones_circular(parang_ant)
    return parang_to_jones_linear(parang_ant)


# ---------------------------------------------------------------------------
# Compute model visibility (forward RIME)
# ---------------------------------------------------------------------------
