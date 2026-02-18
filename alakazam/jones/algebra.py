"""ALAKAZAM Jones Algebra.

All 2x2 complex matrix operations, apply/unapply, residual kernels,
chain composition. Hot-path functions are numba JIT compiled.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
from numba import njit, prange
from enum import Enum
from typing import Tuple
import logging

logger = logging.getLogger("alakazam")


# ---------------------------------------------------------------------------
# Feed basis
# ---------------------------------------------------------------------------

class FeedBasis(Enum):
    LINEAR   = "LINEAR"    # XX XY YX YY  (corr_type 9-12)
    CIRCULAR = "CIRCULAR"  # RR RL LR LL  (corr_type 5-8)


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
# Scalar 2x2 ops (numba, called from within other kernels)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _mul22(A, B):
    C = np.zeros((2, 2), dtype=np.complex128)
    C[0, 0] = A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0]
    C[0, 1] = A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1]
    C[1, 0] = A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0]
    C[1, 1] = A[1, 0] * B[0, 1] + A[1, 1] * B[1, 1]
    return C


@njit(cache=True)
def _inv22(A):
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if np.abs(det) < 1e-30:
        return np.full((2, 2), np.nan + 0j, dtype=np.complex128)
    d = 1.0 / det
    B = np.zeros((2, 2), dtype=np.complex128)
    B[0, 0] =  A[1, 1] * d
    B[0, 1] = -A[0, 1] * d
    B[1, 0] = -A[1, 0] * d
    B[1, 1] =  A[0, 0] * d
    return B


@njit(cache=True)
def _herm22(A):
    B = np.zeros((2, 2), dtype=np.complex128)
    B[0, 0] = np.conj(A[0, 0])
    B[0, 1] = np.conj(A[1, 0])
    B[1, 0] = np.conj(A[0, 1])
    B[1, 1] = np.conj(A[1, 1])
    return B


# ---------------------------------------------------------------------------
# Batch operations on arrays of Jones matrices
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def jones_multiply(J1, J2):
    """Element-wise J1[k] @ J2[k].  Shape: (n, 2, 2)."""
    n = J1.shape[0]
    out = np.empty((n, 2, 2), dtype=np.complex128)
    for k in prange(n):
        out[k] = _mul22(J1[k], J2[k])
    return out


@njit(parallel=True, cache=True)
def jones_inverse(J):
    """Element-wise inverse. Shape: (n, 2, 2)."""
    n = J.shape[0]
    out = np.empty((n, 2, 2), dtype=np.complex128)
    for k in prange(n):
        out[k] = _inv22(J[k])
    return out


@njit(parallel=True, cache=True)
def jones_herm(J):
    """Element-wise Hermitian conjugate. Shape: (n, 2, 2)."""
    n = J.shape[0]
    out = np.empty((n, 2, 2), dtype=np.complex128)
    for k in prange(n):
        out[k] = _herm22(J[k])
    return out


# ---------------------------------------------------------------------------
# Apply / unapply — frequency-independent Jones
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def jones_apply(J, vis, ant1, ant2):
    """V' = J_i V J_j^H.

    J:    (n_ant, 2, 2)
    vis:  (n_bl, 2, 2)
    """
    n_bl = vis.shape[0]
    out = np.empty((n_bl, 2, 2), dtype=np.complex128)
    for bl in prange(n_bl):
        Ji   = J[ant1[bl]]
        JjH  = _herm22(J[ant2[bl]])
        out[bl] = _mul22(_mul22(Ji, vis[bl]), JjH)
    return out


@njit(parallel=True, cache=True)
def jones_unapply(J, vis, ant1, ant2):
    """V' = J_i^{-1} V J_j^{-H}.

    J:    (n_ant, 2, 2)
    vis:  (n_bl, 2, 2)
    """
    n_bl = vis.shape[0]
    out = np.empty((n_bl, 2, 2), dtype=np.complex128)
    for bl in prange(n_bl):
        Ji_inv  = _inv22(J[ant1[bl]])
        JjH_inv = _herm22(_inv22(J[ant2[bl]]))
        out[bl] = _mul22(_mul22(Ji_inv, vis[bl]), JjH_inv)
    return out


# ---------------------------------------------------------------------------
# Apply / unapply — frequency-dependent Jones
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def jones_apply_freq(J, vis, ant1, ant2):
    """V'[bl,f] = J[ant1,f] V[bl,f] J[ant2,f]^H.

    J:   (n_ant, n_freq, 2, 2)
    vis: (n_bl,  n_freq, 2, 2)
    """
    n_bl   = vis.shape[0]
    n_freq = vis.shape[1]
    out = np.empty((n_bl, n_freq, 2, 2), dtype=np.complex128)
    for bl in prange(n_bl):
        a1, a2 = ant1[bl], ant2[bl]
        for f in range(n_freq):
            out[bl, f] = _mul22(_mul22(J[a1, f], vis[bl, f]), _herm22(J[a2, f]))
    return out


@njit(parallel=True, cache=True)
def jones_unapply_freq(J, vis, ant1, ant2):
    """V'[bl,f] = J[ant1,f]^{-1} V[bl,f] J[ant2,f]^{-H}.

    J:   (n_ant, n_freq, 2, 2)
    vis: (n_bl,  n_freq, 2, 2)
    """
    n_bl   = vis.shape[0]
    n_freq = vis.shape[1]
    out = np.empty((n_bl, n_freq, 2, 2), dtype=np.complex128)
    for bl in prange(n_bl):
        a1, a2 = ant1[bl], ant2[bl]
        for f in range(n_freq):
            out[bl, f] = _mul22(
                _mul22(_inv22(J[a1, f]), vis[bl, f]),
                _herm22(_inv22(J[a2, f]))
            )
    return out


# ---------------------------------------------------------------------------
# Residual kernels
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def compute_residual_2x2(J, vis_obs, vis_model, ant1, ant2):
    """r = V_obs - J_i M J_j^H, flattened to real. Returns (n_bl*8,)."""
    n_bl = vis_obs.shape[0]
    r = np.empty(n_bl * 8, dtype=np.float64)
    for bl in prange(n_bl):
        pred = _mul22(_mul22(J[ant1[bl]], vis_model[bl]), _herm22(J[ant2[bl]]))
        d = vis_obs[bl] - pred
        b = bl * 8
        r[b+0] = d[0,0].real; r[b+1] = d[0,0].imag
        r[b+2] = d[0,1].real; r[b+3] = d[0,1].imag
        r[b+4] = d[1,0].real; r[b+5] = d[1,0].imag
        r[b+6] = d[1,1].real; r[b+7] = d[1,1].imag
    return r


@njit(parallel=True, cache=True)
def compute_residual_diag(J, vis_obs, vis_model, ant1, ant2):
    """Parallel-hand residual (pp, qq). Returns (n_bl*4,)."""
    n_bl = vis_obs.shape[0]
    r = np.empty(n_bl * 4, dtype=np.float64)
    for bl in prange(n_bl):
        a1, a2 = ant1[bl], ant2[bl]
        d_pp = vis_obs[bl,0,0] - J[a1,0,0] * vis_model[bl,0,0] * np.conj(J[a2,0,0])
        d_qq = vis_obs[bl,1,1] - J[a1,1,1] * vis_model[bl,1,1] * np.conj(J[a2,1,1])
        b = bl * 4
        r[b+0] = d_pp.real; r[b+1] = d_pp.imag
        r[b+2] = d_qq.real; r[b+3] = d_qq.imag
    return r


@njit(parallel=True, cache=True)
def compute_residual_cross(J, vis_obs, vis_model, ant1, ant2):
    """Cross-hand residual (pq, qp). Returns (n_bl*4,)."""
    n_bl = vis_obs.shape[0]
    r = np.empty(n_bl * 4, dtype=np.float64)
    for bl in prange(n_bl):
        a1, a2 = ant1[bl], ant2[bl]
        d_pq = vis_obs[bl,0,1] - J[a1,0,0] * vis_model[bl,0,1] * np.conj(J[a2,1,1])
        d_qp = vis_obs[bl,1,0] - J[a1,1,1] * vis_model[bl,1,0] * np.conj(J[a2,0,0])
        b = bl * 4
        r[b+0] = d_pq.real; r[b+1] = d_pq.imag
        r[b+2] = d_qp.real; r[b+3] = d_qp.imag
    return r


@njit(parallel=True, cache=True)
def compute_residual_diag_freq(J, vis_obs, vis_model, ant1, ant2):
    """Parallel-hand residual, freq-dependent Jones. Returns (n_bl*n_freq*4,)."""
    n_bl   = vis_obs.shape[0]
    n_freq = vis_obs.shape[1]
    r = np.empty(n_bl * n_freq * 4, dtype=np.float64)
    for bl in prange(n_bl):
        a1, a2 = ant1[bl], ant2[bl]
        for f in range(n_freq):
            d_pp = vis_obs[bl,f,0,0] - J[a1,f,0,0] * vis_model[bl,f,0,0] * np.conj(J[a2,f,0,0])
            d_qq = vis_obs[bl,f,1,1] - J[a1,f,1,1] * vis_model[bl,f,1,1] * np.conj(J[a2,f,1,1])
            b = (bl * n_freq + f) * 4
            r[b+0] = d_pp.real; r[b+1] = d_pp.imag
            r[b+2] = d_qq.real; r[b+3] = d_qq.imag
    return r


@njit(parallel=True, cache=True)
def compute_residual_cross_freq(J, vis_obs, vis_model, ant1, ant2):
    """Cross-hand residual, freq-dependent Jones. Returns (n_bl*n_freq*4,)."""
    n_bl   = vis_obs.shape[0]
    n_freq = vis_obs.shape[1]
    r = np.empty(n_bl * n_freq * 4, dtype=np.float64)
    for bl in prange(n_bl):
        a1, a2 = ant1[bl], ant2[bl]
        for f in range(n_freq):
            d_pq = vis_obs[bl,f,0,1] - J[a1,f,0,0] * vis_model[bl,f,0,1] * np.conj(J[a2,f,1,1])
            d_qp = vis_obs[bl,f,1,0] - J[a1,f,1,1] * vis_model[bl,f,1,0] * np.conj(J[a2,f,0,0])
            b = (bl * n_freq + f) * 4
            r[b+0] = d_pq.real; r[b+1] = d_pq.imag
            r[b+2] = d_qp.real; r[b+3] = d_qp.imag
    return r


# ---------------------------------------------------------------------------
# Chain composition
# ---------------------------------------------------------------------------

def compose_jones_chain(jones_list):
    """Compose [J1, J2, ..., JN] → J_total = J_N ... J_2 J_1.

    Handles mixing of (n_ant, 2, 2) and (n_ant, n_freq, 2, 2).
    """
    if not jones_list:
        return None

    result = jones_list[0].copy()
    for J in jones_list[1:]:
        r_nd = result.ndim
        j_nd = J.ndim

        if r_nd == 3 and j_nd == 3:
            result = jones_multiply(J, result)

        elif r_nd == 4 and j_nd == 4:
            n_ant, n_freq = result.shape[:2]
            out = np.empty_like(result)
            for f in range(n_freq):
                out[:, f] = jones_multiply(J[:, f], result[:, f])
            result = out

        elif r_nd == 3 and j_nd == 4:
            n_ant, n_freq = J.shape[:2]
            out = np.empty_like(J)
            for f in range(n_freq):
                out[:, f] = jones_multiply(J[:, f], result)
            result = out

        elif r_nd == 4 and j_nd == 3:
            n_ant, n_freq = result.shape[:2]
            out = np.empty_like(result)
            for f in range(n_freq):
                out[:, f] = jones_multiply(J, result[:, f])
            result = out

    return result
