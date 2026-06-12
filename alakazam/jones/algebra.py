"""ALAKAZAM v1 Jones Algebra.

All 2x2 complex matrix operations, apply/unapply, chain composition.
Numba JIT for hot paths.  Residuals/Jacobians live in the boa C++ solver.

Universal jones schema: (n_ant, n_freq, n_time, 2, 2)
Per-cell operations work on (n_ant, 2, 2) or (n_ant, n_freq, 2, 2).
Apply/unapply work on visibility rows.

Diagonal-optimized paths for K, G, KC, CP: only touch (0,0) and (1,1).

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
from numba import njit, prange
from enum import Enum
from typing import List, Optional
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


# -------------------------------------------------------------------
# Apply / unapply on row data  V' = J_i^{-1} V J_j^{-H}
# Full 2x2 version for D (leakage) and general Jones
# -------------------------------------------------------------------

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
    """Dispatch: diagonal-optimized or full 2x2.

    J: (n_ant, n_freq, 2, 2) — always 4D.
    vis: (n_row, n_chan, 2, 2)
    """
    if is_diagonal_jones(J):
        return unapply_rows_diag_freqdep(J, vis, ant1, ant2)
    else:
        return unapply_rows_full_freqdep(J, vis, ant1, ant2)


# -------------------------------------------------------------------
# Chain composition — compose list of per-antenna Jones
# -------------------------------------------------------------------

def _ensure_4d(J):
    """(n_ant, 2, 2) -> (n_ant, 1, 2, 2). Already 4D? pass through."""
    if J.ndim == 3:
        return J[:, np.newaxis, :, :]
    return J


def compose_jones_chain(jones_list: List[np.ndarray]) -> Optional[np.ndarray]:
    """Compose [J1, J2, ..., JN] -> J_total = J1 J2 ... JN.

    J1 (the first list entry) is the OUTERMOST term. This matches the
    sequential preapply semantics: step 1 is solved on raw data, so its
    solution sits outermost in the data model
        V = J1 (J2 (... M ...) J2^H) J1^H,
    and each later solution nests inside the terms solved before it.
    Lists must therefore be built in solve order (parang/external first,
    then internal solutions in the order they were solved). For purely
    diagonal chains the order is immaterial; once a non-diagonal term
    (D, parang) is in the chain it is not.

    Each J is (n_ant, n_freq, 2, 2) or (n_ant, 2, 2).
    3D inputs are broadcast to match any 4D input's freq axis.
    Output is always 4D (n_ant, n_freq, 2, 2).
    """
    if not jones_list:
        return None

    # Find max freq dim
    n_freq = 1
    for J in jones_list:
        if J.ndim == 4 and J.shape[1] > n_freq:
            n_freq = J.shape[1]

    # Broadcast all to (n_ant, n_freq, 2, 2)
    def _broadcast(J):
        J = _ensure_4d(J)
        if J.shape[1] == 1 and n_freq > 1:
            return np.broadcast_to(J, (J.shape[0], n_freq, 2, 2)).copy()
        return J

    result = _broadcast(jones_list[0])
    for J in jones_list[1:]:
        J = _broadcast(J)
        out = np.empty_like(result)
        for f in range(n_freq):
            out[:, f] = jones_multiply(result[:, f], J[:, f])
        result = out

    return result
