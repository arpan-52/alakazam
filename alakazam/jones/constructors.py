"""ALAKAZAM v1 Jones Constructors (numba).

Convert native calibration parameters to Jones matrices.
Used in apply / interpolation paths (not in AD optimiser loop).

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------
# K  — parallel delay  diag(e^{-2pi i tau_p nu}, e^{-2pi i tau_q nu})
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def parallel_delay_to_jones(delay, freq):
    """delay:(n_ant,2) ns   freq:(n_freq,) Hz  ->  (n_ant,n_freq,2,2)"""
    n_ant = delay.shape[0]
    n_freq = freq.shape[0]
    J = np.zeros((n_ant, n_freq, 2, 2), dtype=np.complex128)
    for a in prange(n_ant):
        tau_p = delay[a, 0] * 1e-9
        tau_q = delay[a, 1] * 1e-9
        for f in range(n_freq):
            J[a, f, 0, 0] = np.exp(-2j * np.pi * tau_p * freq[f])
            J[a, f, 1, 1] = np.exp(-2j * np.pi * tau_q * freq[f])
    return J


# ---------------------------------------------------------------------------
# G  — gains  diag(g_p e^{i phi_p}, g_q e^{i phi_q})
# ---------------------------------------------------------------------------

@njit(cache=True)
def gains_to_jones(amp, phase):
    """amp:(n_ant,2)  phase:(n_ant,2) rad  ->  (n_ant,2,2)"""
    n_ant = amp.shape[0]
    J = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    for a in range(n_ant):
        J[a, 0, 0] = amp[a, 0] * np.exp(1j * phase[a, 0])
        J[a, 1, 1] = amp[a, 1] * np.exp(1j * phase[a, 1])
    return J


# ---------------------------------------------------------------------------
# D  — leakage  [[1, d_pq], [d_qp, 1]]
# ---------------------------------------------------------------------------

@njit(cache=True)
def leakage_to_jones(d_pq, d_qp):
    """d_pq:(n_ant,) complex  d_qp:(n_ant,) complex  ->  (n_ant,2,2)"""
    n_ant = d_pq.shape[0]
    J = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    for a in range(n_ant):
        J[a, 0, 0] = 1.0 + 0j
        J[a, 0, 1] = d_pq[a]
        J[a, 1, 0] = d_qp[a]
        J[a, 1, 1] = 1.0 + 0j
    return J


# ---------------------------------------------------------------------------
# KC — cross delay  diag(e^{-2pi i tau nu}, 1)  same tau all antennas
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def cross_delay_to_jones(tau_cross, freq, n_ant):
    """tau_cross: scalar ns   freq:(n_freq,) Hz  n_ant:int
    ->  (n_ant, n_freq, 2, 2)"""
    n_freq = freq.shape[0]
    J = np.zeros((n_ant, n_freq, 2, 2), dtype=np.complex128)
    tau = tau_cross * 1e-9
    for a in prange(n_ant):
        for f in range(n_freq):
            J[a, f, 0, 0] = np.exp(-2j * np.pi * tau * freq[f])
            J[a, f, 1, 1] = 1.0 + 0j
    return J


# ---------------------------------------------------------------------------
# CP — cross phase  diag(1, e^{i phi})  same phi all antennas
# ---------------------------------------------------------------------------

@njit(cache=True)
def cross_phase_to_jones(phi_cross, n_ant):
    """phi_cross: scalar rad   n_ant:int  ->  (n_ant, 2, 2)

    J = diag(1, e^{i phi})  — differential cross-hand phase.
    Same value for all antennas (single global parameter).
    """
    J = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    val = np.exp(1j * phi_cross)
    for a in range(n_ant):
        J[a, 0, 0] = 1.0 + 0j
        J[a, 1, 1] = val
    return J
