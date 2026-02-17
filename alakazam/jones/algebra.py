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
