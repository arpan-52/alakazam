"""ALAKAZAM v1 Jones Algebra.

All 2x2 complex matrix operations, apply/unapply, residual kernels,
chain composition.  Numba JIT for hot paths.

Universal jones schema: (n_ant, n_freq, n_time, 2, 2)
Per-cell operations work on (n_ant, 2, 2) or (n_ant, n_freq, 2, 2).
Apply/unapply work on visibility rows.

Diagonal-optimized paths for K, G, KC, CP: only touch (0,0) and (1,1).

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
from numba import njit, prange
from enum import Enum
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger("alakazam")


class FeedBasis(Enum):
    LINEAR = "LINEAR"
    CIRCULAR = "CIRCULAR"


def detect_feed_basis(ms_path: str) -> FeedBasis:
    from casacore.tables import table
    try:
        import os, sys, contextlib
        stderr_fd = sys.stderr.fileno()
        saved = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd); os.close(devnull)
        try:
            pol_tab = table(f"{ms_path}::POLARIZATION", readonly=True, ack=False)
            corr_type = pol_tab.getcol("CORR_TYPE")[0]
            pol_tab.close()
        finally:
            os.dup2(saved, stderr_fd); os.close(saved)
    except (AttributeError, OSError):
        pol_tab = table(f"{ms_path}::POLARIZATION", readonly=True, ack=False)
        corr_type = pol_tab.getcol("CORR_TYPE")[0]
        pol_tab.close()
    if corr_type[0] in (5, 6, 7, 8):
        return FeedBasis.CIRCULAR
    return FeedBasis.LINEAR


def corr_labels(basis: FeedBasis) -> Tuple[str, str, str, str]:
    if basis == FeedBasis.CIRCULAR:
        return ("RR", "RL", "LR", "LL")
    return ("XX", "XY", "YX", "YY")


# -------------------------------------------------------------------
# Scalar 2x2 ops
# -------------------------------------------------------------------

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
    B[0, 0] = A[1, 1] * d
    B[0, 1] = -A[0, 1] * d
    B[1, 0] = -A[1, 0] * d
    B[1, 1] = A[0, 0] * d
    return B


@njit(cache=True)
def _herm22(A):
    B = np.zeros((2, 2), dtype=np.complex128)
    B[0, 0] = np.conj(A[0, 0])
    B[0, 1] = np.conj(A[1, 0])
    B[1, 0] = np.conj(A[0, 1])
    B[1, 1] = np.conj(A[1, 1])
    return B


@njit(cache=True)
def _is_diagonal(A):
    return np.abs(A[0, 1]) < 1e-30 and np.abs(A[1, 0]) < 1e-30


# -------------------------------------------------------------------
# Batch ops on (n, 2, 2) arrays
# -------------------------------------------------------------------

@njit(parallel=True, cache=True)
def jones_multiply(J1, J2):
    n = J1.shape[0]
    out = np.empty((n, 2, 2), dtype=np.complex128)
    for k in prange(n):
        out[k] = _mul22(J1[k], J2[k])
    return out


@njit(parallel=True, cache=True)
def jones_inverse(J):
    n = J.shape[0]
    out = np.empty((n, 2, 2), dtype=np.complex128)
    for k in prange(n):
        out[k] = _inv22(J[k])
    return out


@njit(parallel=True, cache=True)
def jones_herm(J):
    n = J.shape[0]
    out = np.empty((n, 2, 2), dtype=np.complex128)
    for k in prange(n):
        out[k] = _herm22(J[k])
    return out


# -------------------------------------------------------------------
# Apply / unapply on row data  V' = J_i^{-1} V J_j^{-H}
# Full 2x2 version for D (leakage) and general Jones
# -------------------------------------------------------------------

@njit(parallel=True, cache=True)
def unapply_rows_full(J, vis, ant1, ant2):
    """J:(n_ant,2,2)  vis:(n_row,n_chan,2,2) -> corrected.
    Full 2x2 inverse — works for any Jones including off-diagonal."""
    n_row = vis.shape[0]
    n_chan = vis.shape[1]
    out = np.empty_like(vis)
    for r in prange(n_row):
        Ji_inv = _inv22(J[ant1[r]])
        JjH_inv = _herm22(_inv22(J[ant2[r]]))
        for c in range(n_chan):
            out[r, c] = _mul22(_mul22(Ji_inv, vis[r, c]), JjH_inv)
    return out


@njit(parallel=True, cache=True)
def unapply_rows_full_freqdep(J, vis, ant1, ant2):
    """J:(n_ant,n_freq,2,2)  vis:(n_row,n_freq,2,2) -> corrected.
    Full 2x2 freq-dependent."""
    n_row = vis.shape[0]
    n_freq = vis.shape[1]
    out = np.empty_like(vis)
    for r in prange(n_row):
        a1, a2 = ant1[r], ant2[r]
        for f in range(n_freq):
            Ji_inv = _inv22(J[a1, f])
            JjH_inv = _herm22(_inv22(J[a2, f]))
            out[r, f] = _mul22(_mul22(Ji_inv, vis[r, f]), JjH_inv)
    return out


# -------------------------------------------------------------------
# Apply / unapply — diagonal-optimized for K, G, KC, CP
# Only touches (0,0) and (1,1), skips off-diagonal
# -------------------------------------------------------------------

@njit(parallel=True, cache=True)
def unapply_rows_diag(J, vis, ant1, ant2):
    """J:(n_ant,2,2) diagonal  vis:(n_row,n_chan,2,2) -> corrected.
    Only uses J[a,0,0] and J[a,1,1]. Much faster than full 2x2."""
    n_row = vis.shape[0]
    n_chan = vis.shape[1]
    out = np.empty_like(vis)
    for r in prange(n_row):
        a1, a2 = ant1[r], ant2[r]
        gi_p_inv = 1.0 / J[a1, 0, 0]
        gi_q_inv = 1.0 / J[a1, 1, 1]
        gj_p_conj_inv = 1.0 / np.conj(J[a2, 0, 0])
        gj_q_conj_inv = 1.0 / np.conj(J[a2, 1, 1])
        for c in range(n_chan):
            out[r, c, 0, 0] = gi_p_inv * vis[r, c, 0, 0] * gj_p_conj_inv
            out[r, c, 0, 1] = gi_p_inv * vis[r, c, 0, 1] * gj_q_conj_inv
            out[r, c, 1, 0] = gi_q_inv * vis[r, c, 1, 0] * gj_p_conj_inv
            out[r, c, 1, 1] = gi_q_inv * vis[r, c, 1, 1] * gj_q_conj_inv
    return out


@njit(parallel=True, cache=True)
def unapply_rows_diag_freqdep(J, vis, ant1, ant2):
    """J:(n_ant,n_freq,2,2) diagonal  vis:(n_row,n_freq,2,2) -> corrected."""
    n_row = vis.shape[0]
    n_freq = vis.shape[1]
    out = np.empty_like(vis)
    for r in prange(n_row):
        a1, a2 = ant1[r], ant2[r]
        for f in range(n_freq):
            gi_p_inv = 1.0 / J[a1, f, 0, 0]
            gi_q_inv = 1.0 / J[a1, f, 1, 1]
            gj_p_conj_inv = 1.0 / np.conj(J[a2, f, 0, 0])
            gj_q_conj_inv = 1.0 / np.conj(J[a2, f, 1, 1])
            out[r, f, 0, 0] = gi_p_inv * vis[r, f, 0, 0] * gj_p_conj_inv
            out[r, f, 0, 1] = gi_p_inv * vis[r, f, 0, 1] * gj_q_conj_inv
            out[r, f, 1, 0] = gi_q_inv * vis[r, f, 1, 0] * gj_p_conj_inv
            out[r, f, 1, 1] = gi_q_inv * vis[r, f, 1, 1] * gj_q_conj_inv
    return out


def is_diagonal_jones(J):
    """Check if a Jones array is diagonal (off-diag < threshold)."""
    return (np.max(np.abs(J[..., 0, 1])) < 1e-20 and
            np.max(np.abs(J[..., 1, 0])) < 1e-20)


def unapply_jones_to_rows(J, vis, ant1, ant2):
    """Smart dispatch: diagonal-optimized or full 2x2.

    J: (n_ant, 2, 2) or (n_ant, n_freq, 2, 2)
    vis: (n_row, n_chan, 2, 2)
    """
    diag = is_diagonal_jones(J)
    if J.ndim == 3:
        if diag:
            return unapply_rows_diag(J, vis, ant1, ant2)
        else:
            return unapply_rows_full(J, vis, ant1, ant2)
    elif J.ndim == 4:
        if diag:
            return unapply_rows_diag_freqdep(J, vis, ant1, ant2)
        else:
            return unapply_rows_full_freqdep(J, vis, ant1, ant2)
    else:
        raise ValueError(f"J has unexpected ndim={J.ndim}")


# -------------------------------------------------------------------
# Residual kernels — all work on per-cell averaged data
# -------------------------------------------------------------------

@njit(parallel=True, cache=True)
def compute_residual_2x2(J, vis_obs, vis_model, ant1, ant2):
    """Full 2x2 residual.  J:(n_ant,2,2)  vis:(n_bl,2,2).
    Returns (n_bl*8,) float64."""
    n_bl = vis_obs.shape[0]
    r = np.empty(n_bl * 8, dtype=np.float64)
    for bl in prange(n_bl):
        pred = _mul22(_mul22(J[ant1[bl]], vis_model[bl]),
                      _herm22(J[ant2[bl]]))
        d = vis_obs[bl] - pred
        b = bl * 8
        r[b] = d[0, 0].real;     r[b + 1] = d[0, 0].imag
        r[b + 2] = d[0, 1].real; r[b + 3] = d[0, 1].imag
        r[b + 4] = d[1, 0].real; r[b + 5] = d[1, 0].imag
        r[b + 6] = d[1, 1].real; r[b + 7] = d[1, 1].imag
    return r


@njit(parallel=True, cache=True)
def compute_residual_diag(J, vis_obs, vis_model, ant1, ant2):
    """Diagonal residual (pp,qq).  J:(n_ant,2,2)  vis:(n_bl,2,2).
    Returns (n_bl*4,)."""
    n_bl = vis_obs.shape[0]
    r = np.empty(n_bl * 4, dtype=np.float64)
    for bl in prange(n_bl):
        a1, a2 = ant1[bl], ant2[bl]
        d_pp = (vis_obs[bl, 0, 0]
                - J[a1, 0, 0] * vis_model[bl, 0, 0] * np.conj(J[a2, 0, 0]))
        d_qq = (vis_obs[bl, 1, 1]
                - J[a1, 1, 1] * vis_model[bl, 1, 1] * np.conj(J[a2, 1, 1]))
        b = bl * 4
        r[b] = d_pp.real;     r[b + 1] = d_pp.imag
        r[b + 2] = d_qq.real; r[b + 3] = d_qq.imag
    return r


@njit(parallel=True, cache=True)
def compute_residual_diag_freq(J, vis_obs, vis_model, ant1, ant2):
    """Diagonal freq-dependent.  J:(n_ant,n_f,2,2) vis:(n_bl,n_f,2,2).
    Returns (n_bl*n_f*4,)."""
    n_bl = vis_obs.shape[0]
    n_freq = vis_obs.shape[1]
    r = np.empty(n_bl * n_freq * 4, dtype=np.float64)
    for bl in prange(n_bl):
        a1, a2 = ant1[bl], ant2[bl]
        for f in range(n_freq):
            d_pp = (vis_obs[bl, f, 0, 0]
                    - J[a1, f, 0, 0] * vis_model[bl, f, 0, 0]
                    * np.conj(J[a2, f, 0, 0]))
            d_qq = (vis_obs[bl, f, 1, 1]
                    - J[a1, f, 1, 1] * vis_model[bl, f, 1, 1]
                    * np.conj(J[a2, f, 1, 1]))
            b = (bl * n_freq + f) * 4
            r[b] = d_pp.real;     r[b + 1] = d_pp.imag
            r[b + 2] = d_qq.real; r[b + 3] = d_qq.imag
    return r


@njit(parallel=True, cache=True)
def compute_residual_cross(J, vis_obs, vis_model, ant1, ant2):
    """Cross-hand (pq,qp).  J:(n_ant,2,2) vis:(n_bl,2,2).
    Returns (n_bl*4,)."""
    n_bl = vis_obs.shape[0]
    r = np.empty(n_bl * 4, dtype=np.float64)
    for bl in prange(n_bl):
        a1, a2 = ant1[bl], ant2[bl]
        d_pq = (vis_obs[bl, 0, 1]
                - J[a1, 0, 0] * vis_model[bl, 0, 1] * np.conj(J[a2, 1, 1]))
        d_qp = (vis_obs[bl, 1, 0]
                - J[a1, 1, 1] * vis_model[bl, 1, 0] * np.conj(J[a2, 0, 0]))
        b = bl * 4
        r[b] = d_pq.real;     r[b + 1] = d_pq.imag
        r[b + 2] = d_qp.real; r[b + 3] = d_qp.imag
    return r


@njit(parallel=True, cache=True)
def compute_residual_cross_freq(J, vis_obs, vis_model, ant1, ant2):
    """Cross-hand freq-dependent.  J:(n_ant,n_f,2,2) vis:(n_bl,n_f,2,2).
    Returns (n_bl*n_f*4,)."""
    n_bl = vis_obs.shape[0]
    n_freq = vis_obs.shape[1]
    r = np.empty(n_bl * n_freq * 4, dtype=np.float64)
    for bl in prange(n_bl):
        a1, a2 = ant1[bl], ant2[bl]
        for f in range(n_freq):
            d_pq = (vis_obs[bl, f, 0, 1]
                    - J[a1, f, 0, 0] * vis_model[bl, f, 0, 1]
                    * np.conj(J[a2, f, 1, 1]))
            d_qp = (vis_obs[bl, f, 1, 0]
                    - J[a1, f, 1, 1] * vis_model[bl, f, 1, 0]
                    * np.conj(J[a2, f, 0, 0]))
            b = (bl * n_freq + f) * 4
            r[b] = d_pq.real;     r[b + 1] = d_pq.imag
            r[b + 2] = d_qp.real; r[b + 3] = d_qp.imag
    return r


# -------------------------------------------------------------------
# Chain composition — compose list of per-antenna Jones
# -------------------------------------------------------------------

def compose_jones_chain(jones_list: List[np.ndarray]) -> Optional[np.ndarray]:
    """Compose [J1, J2, ..., JN] -> J_total = J_N ... J_2 J_1.

    Each J can be (n_ant, 2, 2) or (n_ant, n_freq, 2, 2).
    Mixed dims: freq-indep broadcasts to freq-dep.
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
