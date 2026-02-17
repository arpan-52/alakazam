"""
ALAKAZAM Jones Matrix Operations.

All 2x2 Jones algebra: multiply, inverse, apply, unapply, delay_to_jones.
Single FeedBasis enum. Parallactic angle computation and application.
All hot-path functions are numba JIT compiled.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
from numba import njit, prange
from enum import Enum
from typing import Tuple, Optional
import logging

logger = logging.getLogger("alakazam")


# ---------------------------------------------------------------------------
# Feed basis
# ---------------------------------------------------------------------------

class FeedBasis(Enum):
    LINEAR = "LINEAR"      # XX, XY, YX, YY  (corr_type 9-12)
    CIRCULAR = "CIRCULAR"  # RR, RL, LR, LL  (corr_type 5-8)


def detect_feed_basis(ms_path: str) -> FeedBasis:
    """Detect feed basis from MS POLARIZATION subtable."""
    from casacore.tables import table
    pol_tab = table(f"{ms_path}::POLARIZATION", readonly=True, ack=False)
    corr_type = pol_tab.getcol("CORR_TYPE")[0]
    pol_tab.close()
    if corr_type[0] in (5, 6, 7, 8):
        return FeedBasis.CIRCULAR
    return FeedBasis.LINEAR


def corr_labels(basis: FeedBasis) -> Tuple[str, str, str, str]:
    """Return (pp, pq, qp, qq) labels for the given basis."""
    if basis == FeedBasis.CIRCULAR:
        return ("RR", "RL", "LR", "LL")
    return ("XX", "XY", "YX", "YY")


# ---------------------------------------------------------------------------
# Basic 2x2 algebra  (numba)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _mat2x2_mul(A, B):
    """Multiply two 2x2 complex matrices."""
    C = np.zeros((2, 2), dtype=np.complex128)
    C[0, 0] = A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0]
    C[0, 1] = A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1]
    C[1, 0] = A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0]
    C[1, 1] = A[1, 0] * B[0, 1] + A[1, 1] * B[1, 1]
    return C


@njit(cache=True)
def _mat2x2_inv(A):
    """Invert a 2x2 complex matrix."""
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if np.abs(det) < 1e-30:
        return np.full((2, 2), np.nan + 0j, dtype=np.complex128)
    inv_det = 1.0 / det
    B = np.zeros((2, 2), dtype=np.complex128)
    B[0, 0] = A[1, 1] * inv_det
    B[0, 1] = -A[0, 1] * inv_det
    B[1, 0] = -A[1, 0] * inv_det
    B[1, 1] = A[0, 0] * inv_det
    return B


@njit(cache=True)
def _mat2x2_herm(A):
    """Hermitian conjugate of a 2x2 complex matrix."""
    B = np.zeros((2, 2), dtype=np.complex128)
    B[0, 0] = np.conj(A[0, 0])
    B[0, 1] = np.conj(A[1, 0])
    B[1, 0] = np.conj(A[0, 1])
    B[1, 1] = np.conj(A[1, 1])
    return B


# ---------------------------------------------------------------------------
# Batch operations on (n, 2, 2) arrays
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def jones_multiply(J1, J2):
    """Element-wise multiply arrays of Jones matrices: result[k] = J1[k] @ J2[k].

    J1, J2: (n, 2, 2) complex128
    Returns: (n, 2, 2) complex128
    """
    n = J1.shape[0]
    out = np.empty((n, 2, 2), dtype=np.complex128)
    for k in prange(n):
        out[k] = _mat2x2_mul(J1[k], J2[k])
    return out


@njit(parallel=True, cache=True)
def jones_inverse(J):
    """Element-wise invert array of Jones matrices.

    J: (n, 2, 2) complex128
    Returns: (n, 2, 2) complex128
    """
    n = J.shape[0]
    out = np.empty((n, 2, 2), dtype=np.complex128)
    for k in prange(n):
        out[k] = _mat2x2_inv(J[k])
    return out


@njit(parallel=True, cache=True)
def jones_herm(J):
    """Element-wise Hermitian conjugate.

    J: (n, 2, 2) complex128
    Returns: (n, 2, 2) complex128
    """
    n = J.shape[0]
    out = np.empty((n, 2, 2), dtype=np.complex128)
    for k in prange(n):
        out[k] = _mat2x2_herm(J[k])
    return out


# ---------------------------------------------------------------------------
# Apply / unapply Jones to visibilities
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def jones_apply(J, vis, ant1, ant2):
    """Apply Jones matrices to visibilities: V' = J_i V J_j^H.

    J:    (n_ant, 2, 2) complex128
    vis:  (n_bl, 2, 2) complex128
    ant1: (n_bl,) int32
    ant2: (n_bl,) int32
    Returns: (n_bl, 2, 2) complex128
    """
    n_bl = vis.shape[0]
    out = np.empty((n_bl, 2, 2), dtype=np.complex128)
    for bl in prange(n_bl):
        Ji = J[ant1[bl]]
        Jj_H = _mat2x2_herm(J[ant2[bl]])
        out[bl] = _mat2x2_mul(_mat2x2_mul(Ji, vis[bl]), Jj_H)
    return out


@njit(parallel=True, cache=True)
def jones_unapply(J, vis, ant1, ant2):
    """Unapply (correct) Jones: V' = J_i^{-1} V J_j^{-H}.

    J:    (n_ant, 2, 2) complex128
    vis:  (n_bl, 2, 2) complex128
    ant1: (n_bl,) int32
    ant2: (n_bl,) int32
    Returns: (n_bl, 2, 2) complex128
    """
    n_bl = vis.shape[0]
    out = np.empty((n_bl, 2, 2), dtype=np.complex128)
    for bl in prange(n_bl):
        Ji_inv = _mat2x2_inv(J[ant1[bl]])
        Jj_inv_H = _mat2x2_herm(_mat2x2_inv(J[ant2[bl]]))
        out[bl] = _mat2x2_mul(_mat2x2_mul(Ji_inv, vis[bl]), Jj_inv_H)
    return out


@njit(parallel=True, cache=True)
def jones_apply_freq(J, vis, ant1, ant2):
    """Apply freq-dependent Jones: V'[bl,f] = J[ant1,f] V[bl,f] J[ant2,f]^H.

    J:    (n_ant, n_freq, 2, 2) complex128
    vis:  (n_bl, n_freq, 2, 2) complex128
    ant1: (n_bl,) int32
    ant2: (n_bl,) int32
    Returns: (n_bl, n_freq, 2, 2)
    """
    n_bl = vis.shape[0]
    n_freq = vis.shape[1]
    out = np.empty((n_bl, n_freq, 2, 2), dtype=np.complex128)
    for bl in prange(n_bl):
        a1 = ant1[bl]
        a2 = ant2[bl]
        for f in range(n_freq):
            Ji = J[a1, f]
            Jj_H = _mat2x2_herm(J[a2, f])
            out[bl, f] = _mat2x2_mul(_mat2x2_mul(Ji, vis[bl, f]), Jj_H)
    return out


@njit(parallel=True, cache=True)
def jones_unapply_freq(J, vis, ant1, ant2):
    """Unapply freq-dependent Jones: V'[bl,f] = J[ant1,f]^{-1} V[bl,f] J[ant2,f]^{-H}.

    J:    (n_ant, n_freq, 2, 2) complex128
    vis:  (n_bl, n_freq, 2, 2) complex128
    ant1: (n_bl,) int32
    ant2: (n_bl,) int32
    Returns: (n_bl, n_freq, 2, 2)
    """
    n_bl = vis.shape[0]
    n_freq = vis.shape[1]
    out = np.empty((n_bl, n_freq, 2, 2), dtype=np.complex128)
    for bl in prange(n_bl):
        a1 = ant1[bl]
        a2 = ant2[bl]
        for f in range(n_freq):
            Ji_inv = _mat2x2_inv(J[a1, f])
            Jj_inv_H = _mat2x2_herm(_mat2x2_inv(J[a2, f]))
            out[bl, f] = _mat2x2_mul(_mat2x2_mul(Ji_inv, vis[bl, f]), Jj_inv_H)
    return out


# ---------------------------------------------------------------------------
# Delay → Jones conversion
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def delay_to_jones(delay, freq):
    """Convert delays to frequency-dependent diagonal Jones matrices.

    delay: (n_ant, 2) float64  — delays in nanoseconds
    freq:  (n_freq,) float64   — frequencies in Hz
    Returns: (n_ant, n_freq, 2, 2) complex128
    """
    n_ant = delay.shape[0]
    n_freq = freq.shape[0]
    J = np.zeros((n_ant, n_freq, 2, 2), dtype=np.complex128)
    for a in prange(n_ant):
        tau_p = delay[a, 0] * 1e-9  # ns → seconds
        tau_q = delay[a, 1] * 1e-9
        for f in range(n_freq):
            J[a, f, 0, 0] = np.exp(-2j * np.pi * tau_p * freq[f])
            J[a, f, 1, 1] = np.exp(-2j * np.pi * tau_q * freq[f])
    return J


@njit(parallel=True, cache=True)
def crossdelay_to_jones(delay_pq, freq):
    """Convert cross-hand delays to freq-dependent diagonal Jones.

    delay_pq: (n_ant,) float64  — cross-delay in ns
    freq:     (n_freq,) float64 — Hz
    Returns:  (n_ant, n_freq, 2, 2) complex128
    """
    n_ant = delay_pq.shape[0]
    n_freq = freq.shape[0]
    J = np.zeros((n_ant, n_freq, 2, 2), dtype=np.complex128)
    for a in prange(n_ant):
        tau = delay_pq[a] * 1e-9
        for f in range(n_freq):
            J[a, f, 0, 0] = 1.0 + 0j
            J[a, f, 1, 1] = np.exp(-2j * np.pi * tau * freq[f])
    return J


@njit(cache=True)
def crossphase_to_jones(phi_pq, n_ant):
    """Convert cross-hand phases to diagonal Jones.

    phi_pq: (n_ant,) float64  — cross-hand phase in radians
    Returns: (n_ant, 2, 2) complex128
    """
    J = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    for a in range(n_ant):
        J[a, 0, 0] = 1.0 + 0j
        J[a, 1, 1] = np.exp(1j * phi_pq[a])
    return J


@njit(cache=True)
def gain_to_jones(amp, phase, n_ant):
    """Convert amplitude+phase to diagonal Jones.

    amp:   (n_ant, 2) float64
    phase: (n_ant, 2) float64 — radians
    Returns: (n_ant, 2, 2) complex128
    """
    J = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    for a in range(n_ant):
        J[a, 0, 0] = amp[a, 0] * np.exp(1j * phase[a, 0])
        J[a, 1, 1] = amp[a, 1] * np.exp(1j * phase[a, 1])
    return J


@njit(cache=True)
def leakage_to_jones(d_pq, d_qp, n_ant):
    """Convert leakage terms to Jones.

    d_pq: (n_ant,) complex128
    d_qp: (n_ant,) complex128
    Returns: (n_ant, 2, 2) complex128
    """
    J = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    for a in range(n_ant):
        J[a, 0, 0] = 1.0 + 0j
        J[a, 0, 1] = d_pq[a]
        J[a, 1, 0] = d_qp[a]
        J[a, 1, 1] = 1.0 + 0j
    return J


# ---------------------------------------------------------------------------
# Parallactic angle
# ---------------------------------------------------------------------------

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

@njit(parallel=True, cache=True)
def compute_residual_2x2(J, vis_obs, vis_model, ant1, ant2):
    """Compute residual: r = V_obs - J_i M J_j^H, flattened to real vector.

    J:         (n_ant, 2, 2) complex128
    vis_obs:   (n_bl, 2, 2) complex128
    vis_model: (n_bl, 2, 2) complex128
    ant1:      (n_bl,) int32
    ant2:      (n_bl,) int32
    Returns:   (n_bl * 8,) float64   [4 complex = 8 real per baseline]
    """
    n_bl = vis_obs.shape[0]
    residual = np.empty(n_bl * 8, dtype=np.float64)
    for bl in prange(n_bl):
        Ji = J[ant1[bl]]
        Jj_H = _mat2x2_herm(J[ant2[bl]])
        model = _mat2x2_mul(_mat2x2_mul(Ji, vis_model[bl]), Jj_H)
        diff = vis_obs[bl] - model
        idx = bl * 8
        residual[idx + 0] = diff[0, 0].real
        residual[idx + 1] = diff[0, 0].imag
        residual[idx + 2] = diff[0, 1].real
        residual[idx + 3] = diff[0, 1].imag
        residual[idx + 4] = diff[1, 0].real
        residual[idx + 5] = diff[1, 0].imag
        residual[idx + 6] = diff[1, 1].real
        residual[idx + 7] = diff[1, 1].imag
    return residual


@njit(parallel=True, cache=True)
def compute_residual_diag(J, vis_obs, vis_model, ant1, ant2):
    """Residual using only parallel-hand correlations (pp, qq).

    J:         (n_ant, 2, 2) complex128  — diagonal Jones
    vis_obs:   (n_bl, 2, 2) complex128
    vis_model: (n_bl, 2, 2) complex128
    ant1:      (n_bl,) int32
    ant2:      (n_bl,) int32
    Returns:   (n_bl * 4,) float64   [2 complex = 4 real per baseline]
    """
    n_bl = vis_obs.shape[0]
    residual = np.empty(n_bl * 4, dtype=np.float64)
    for bl in prange(n_bl):
        a1 = ant1[bl]
        a2 = ant2[bl]
        # pp: V_pp - J_i[0,0] * M_pp * conj(J_j[0,0])
        diff_pp = vis_obs[bl, 0, 0] - J[a1, 0, 0] * vis_model[bl, 0, 0] * np.conj(J[a2, 0, 0])
        # qq: V_qq - J_i[1,1] * M_qq * conj(J_j[1,1])
        diff_qq = vis_obs[bl, 1, 1] - J[a1, 1, 1] * vis_model[bl, 1, 1] * np.conj(J[a2, 1, 1])
        idx = bl * 4
        residual[idx + 0] = diff_pp.real
        residual[idx + 1] = diff_pp.imag
        residual[idx + 2] = diff_qq.real
        residual[idx + 3] = diff_qq.imag
    return residual


@njit(parallel=True, cache=True)
def compute_residual_cross(J, vis_obs, vis_model, ant1, ant2):
    """Residual using only cross-hand correlations (pq, qp).

    J:         (n_ant, 2, 2) complex128  — diagonal Jones (Xf, Kcross)
    vis_obs:   (n_bl, 2, 2) complex128
    vis_model: (n_bl, 2, 2) complex128
    ant1:      (n_bl,) int32
    ant2:      (n_bl,) int32
    Returns:   (n_bl * 4,) float64
    """
    n_bl = vis_obs.shape[0]
    residual = np.empty(n_bl * 4, dtype=np.float64)
    for bl in prange(n_bl):
        a1 = ant1[bl]
        a2 = ant2[bl]
        # pq: V_pq - J_i[0,0] * M_pq * conj(J_j[1,1])
        diff_pq = vis_obs[bl, 0, 1] - J[a1, 0, 0] * vis_model[bl, 0, 1] * np.conj(J[a2, 1, 1])
        # qp: V_qp - J_i[1,1] * M_qp * conj(J_j[0,0])
        diff_qp = vis_obs[bl, 1, 0] - J[a1, 1, 1] * vis_model[bl, 1, 0] * np.conj(J[a2, 0, 0])
        idx = bl * 4
        residual[idx + 0] = diff_pq.real
        residual[idx + 1] = diff_pq.imag
        residual[idx + 2] = diff_qp.real
        residual[idx + 3] = diff_qp.imag
    return residual


@njit(parallel=True, cache=True)
def compute_residual_diag_freq(J, vis_obs, vis_model, ant1, ant2):
    """Residual for freq-dependent diagonal Jones (K, Kcross-like).

    J:         (n_ant, n_freq, 2, 2) complex128
    vis_obs:   (n_bl, n_freq, 2, 2) complex128
    vis_model: (n_bl, n_freq, 2, 2) complex128
    ant1:      (n_bl,) int32
    ant2:      (n_bl,) int32
    Returns:   (n_bl * n_freq * 4,) float64
    """
    n_bl = vis_obs.shape[0]
    n_freq = vis_obs.shape[1]
    residual = np.empty(n_bl * n_freq * 4, dtype=np.float64)
    for bl in prange(n_bl):
        a1 = ant1[bl]
        a2 = ant2[bl]
        for f in range(n_freq):
            diff_pp = vis_obs[bl, f, 0, 0] - J[a1, f, 0, 0] * vis_model[bl, f, 0, 0] * np.conj(J[a2, f, 0, 0])
            diff_qq = vis_obs[bl, f, 1, 1] - J[a1, f, 1, 1] * vis_model[bl, f, 1, 1] * np.conj(J[a2, f, 1, 1])
            idx = (bl * n_freq + f) * 4
            residual[idx + 0] = diff_pp.real
            residual[idx + 1] = diff_pp.imag
            residual[idx + 2] = diff_qq.real
            residual[idx + 3] = diff_qq.imag
    return residual


@njit(parallel=True, cache=True)
def compute_residual_cross_freq(J, vis_obs, vis_model, ant1, ant2):
    """Residual for freq-dependent cross-hand Jones (Kcross).

    J:         (n_ant, n_freq, 2, 2) complex128
    vis_obs:   (n_bl, n_freq, 2, 2) complex128
    vis_model: (n_bl, n_freq, 2, 2) complex128
    ant1:      (n_bl,) int32
    ant2:      (n_bl,) int32
    Returns:   (n_bl * n_freq * 4,) float64
    """
    n_bl = vis_obs.shape[0]
    n_freq = vis_obs.shape[1]
    residual = np.empty(n_bl * n_freq * 4, dtype=np.float64)
    for bl in prange(n_bl):
        a1 = ant1[bl]
        a2 = ant2[bl]
        for f in range(n_freq):
            diff_pq = vis_obs[bl, f, 0, 1] - J[a1, f, 0, 0] * vis_model[bl, f, 0, 1] * np.conj(J[a2, f, 1, 1])
            diff_qp = vis_obs[bl, f, 1, 0] - J[a1, f, 1, 1] * vis_model[bl, f, 1, 0] * np.conj(J[a2, f, 0, 0])
            idx = (bl * n_freq + f) * 4
            residual[idx + 0] = diff_pq.real
            residual[idx + 1] = diff_pq.imag
            residual[idx + 2] = diff_qp.real
            residual[idx + 3] = diff_qp.imag
    return residual


# ---------------------------------------------------------------------------
# Compose multiple Jones chains
# ---------------------------------------------------------------------------

def compose_jones_chain(jones_list):
    """Compose list of Jones arrays: J_total = J_N ... J_2 J_1.

    jones_list : list of (n_ant, 2, 2) or (n_ant, n_freq, 2, 2) arrays
                 ordered [J_1, J_2, ..., J_N]  (first applied first)
    Returns: composed Jones in the highest-dimensional format present
    """
    if not jones_list:
        return None

    result = jones_list[0].copy()
    for J in jones_list[1:]:
        if result.ndim == 3 and J.ndim == 3:
            result = jones_multiply(J, result)
        elif result.ndim == 4 and J.ndim == 4:
            n_ant, n_freq = result.shape[:2]
            out = np.empty_like(result)
            for f in range(n_freq):
                out[:, f] = jones_multiply(J[:, f], result[:, f])
            result = out
        elif result.ndim == 3 and J.ndim == 4:
            # Broadcast: expand result to 4D
            n_ant, n_freq = J.shape[:2]
            out = np.empty_like(J)
            for f in range(n_freq):
                out[:, f] = jones_multiply(J[:, f], result)
            result = out
        elif result.ndim == 4 and J.ndim == 3:
            n_ant, n_freq = result.shape[:2]
            out = np.empty_like(result)
            for f in range(n_freq):
                out[:, f] = jones_multiply(J, result[:, f])
            result = out
    return result
