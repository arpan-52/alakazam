"""ALAKAZAM Jones Constructors.

Convert native calibration parameters to Jones matrices.
All hot-path functions are numba JIT compiled.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def delay_to_jones(delay, freq):
    """Convert delays → freq-dependent diagonal Jones.

    delay: (n_ant, 2) float64  — ns
    freq:  (n_freq,) float64   — Hz
    Returns: (n_ant, n_freq, 2, 2) complex128
    """
    n_ant  = delay.shape[0]
    n_freq = freq.shape[0]
    J = np.zeros((n_ant, n_freq, 2, 2), dtype=np.complex128)
    for a in prange(n_ant):
        tau_p = delay[a, 0] * 1e-9
        tau_q = delay[a, 1] * 1e-9
        for f in range(n_freq):
            J[a, f, 0, 0] = np.exp(-2j * np.pi * tau_p * freq[f])
            J[a, f, 1, 1] = np.exp(-2j * np.pi * tau_q * freq[f])
    return J


@njit(parallel=True, cache=True)
def crossdelay_to_jones(delay_pq, freq):
    """Convert cross-hand delay → freq-dependent Jones.

    delay_pq: (n_ant,) float64 — ns
    freq:     (n_freq,) float64 — Hz
    Returns:  (n_ant, n_freq, 2, 2) complex128
    """
    n_ant  = delay_pq.shape[0]
    n_freq = freq.shape[0]
    J = np.zeros((n_ant, n_freq, 2, 2), dtype=np.complex128)
    for a in prange(n_ant):
        tau = delay_pq[a] * 1e-9
        for f in range(n_freq):
            J[a, f, 0, 0] = 1.0 + 0j
            J[a, f, 1, 1] = np.exp(-2j * np.pi * tau * freq[f])
    return J


@njit(cache=True)
def crossphase_to_jones(phi_pq):
    """Convert cross-hand phases → diagonal Jones.

    phi_pq: (n_ant,) float64 — radians
    Returns: (n_ant, 2, 2) complex128
    """
    n_ant = phi_pq.shape[0]
    J = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    for a in range(n_ant):
        J[a, 0, 0] = 1.0 + 0j
        J[a, 1, 1] = np.exp(1j * phi_pq[a])
    return J


@njit(cache=True)
def gain_to_jones(amp, phase):
    """Convert amplitude + phase → diagonal Jones.

    amp:   (n_ant, 2) float64
    phase: (n_ant, 2) float64 — radians
    Returns: (n_ant, 2, 2) complex128
    """
    n_ant = amp.shape[0]
    J = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    for a in range(n_ant):
        J[a, 0, 0] = amp[a, 0] * np.exp(1j * phase[a, 0])
        J[a, 1, 1] = amp[a, 1] * np.exp(1j * phase[a, 1])
    return J


@njit(cache=True)
def leakage_to_jones(d_pq, d_qp):
    """Convert leakage terms → full Jones.

    d_pq: (n_ant,) complex128
    d_qp: (n_ant,) complex128
    Returns: (n_ant, 2, 2) complex128
    """
    n_ant = d_pq.shape[0]
    J = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    for a in range(n_ant):
        J[a, 0, 0] = 1.0 + 0j
        J[a, 0, 1] = d_pq[a]
        J[a, 1, 0] = d_qp[a]
        J[a, 1, 1] = 1.0 + 0j
    return J
