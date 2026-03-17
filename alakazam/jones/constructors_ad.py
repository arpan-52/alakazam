"""ALAKAZAM v1 Jones Constructors — AD-compatible.

Pure-Python / NumPy versions that JAX and PyTorch can trace through
for automatic differentiation inside the optimiser loop.

No numba — these are called inside jax.grad / torch.autograd.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np


def parallel_delay_to_jones_np(delay, freq):
    """delay:(n_ant,2) ns  freq:(n_freq,) Hz -> (n_ant,n_freq,2,2) complex128.

    Pure numpy — no numba.
    """
    n_ant = delay.shape[0]
    n_freq = freq.shape[0]
    tau_p = delay[:, 0:1] * 1e-9       # (n_ant, 1)
    tau_q = delay[:, 1:2] * 1e-9       # (n_ant, 1)
    freq_row = freq[np.newaxis, :]      # (1, n_freq)
    J = np.zeros((n_ant, n_freq, 2, 2), dtype=np.complex128)
    J[:, :, 0, 0] = np.exp(-2j * np.pi * tau_p * freq_row)
    J[:, :, 1, 1] = np.exp(-2j * np.pi * tau_q * freq_row)
    return J


def gains_to_jones_np(amp, phase):
    """amp:(n_ant,2)  phase:(n_ant,2) rad -> (n_ant,2,2) complex128."""
    n_ant = amp.shape[0]
    J = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    J[:, 0, 0] = amp[:, 0] * np.exp(1j * phase[:, 0])
    J[:, 1, 1] = amp[:, 1] * np.exp(1j * phase[:, 1])
    return J


def leakage_to_jones_np(d_pq, d_qp):
    """d_pq:(n_ant,) complex  d_qp:(n_ant,) complex -> (n_ant,2,2)."""
    n_ant = d_pq.shape[0]
    J = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    J[:, 0, 0] = 1.0
    J[:, 0, 1] = d_pq
    J[:, 1, 0] = d_qp
    J[:, 1, 1] = 1.0
    return J


def cross_delay_to_jones_np(tau_cross, freq, n_ant):
    """tau_cross: scalar ns  freq:(n_freq,) Hz  n_ant:int
    -> (n_ant,n_freq,2,2)."""
    n_freq = freq.shape[0]
    tau = tau_cross * 1e-9
    phase = np.exp(-2j * np.pi * tau * freq)   # (n_freq,)
    J = np.zeros((n_ant, n_freq, 2, 2), dtype=np.complex128)
    J[:, :, 0, 0] = phase[np.newaxis, :]
    J[:, :, 1, 1] = 1.0
    return J


def cross_phase_to_jones_np(phi_cross, n_ant):
    """phi_cross: scalar rad  n_ant:int -> (n_ant,2,2).

    J = diag(1, e^{i phi}) — differential cross-hand phase.
    """
    J = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    J[:, 0, 0] = 1.0
    J[:, 1, 1] = np.exp(1j * phi_cross)
    return J


# ---------------------------------------------------------------------------
# Residual functions (pure numpy) for scipy/JAX/torch wrappers
# ---------------------------------------------------------------------------

def residual_diag_np(J, vis_obs, vis_model, ant1, ant2):
    """Parallel-hand residual (pp, qq).  Returns (n_bl*4,) float64."""
    n_bl = vis_obs.shape[0]
    d_pp = (vis_obs[:, 0, 0]
            - J[ant1, 0, 0] * vis_model[:, 0, 0] * np.conj(J[ant2, 0, 0]))
    d_qq = (vis_obs[:, 1, 1]
            - J[ant1, 1, 1] * vis_model[:, 1, 1] * np.conj(J[ant2, 1, 1]))
    r = np.empty(n_bl * 4, dtype=np.float64)
    r[0::4] = d_pp.real
    r[1::4] = d_pp.imag
    r[2::4] = d_qq.real
    r[3::4] = d_qq.imag
    return r


def residual_diag_freq_np(J, vis_obs, vis_model, ant1, ant2):
    """Parallel-hand freq-dependent.  J:(n_ant,n_f,2,2)  vis:(n_bl,n_f,2,2).
    Returns (n_bl*n_f*4,) float64."""
    n_bl, n_freq = vis_obs.shape[:2]
    d_pp = (vis_obs[:, :, 0, 0]
            - J[ant1, :, 0, 0] * vis_model[:, :, 0, 0]
            * np.conj(J[ant2, :, 0, 0]))
    d_qq = (vis_obs[:, :, 1, 1]
            - J[ant1, :, 1, 1] * vis_model[:, :, 1, 1]
            * np.conj(J[ant2, :, 1, 1]))
    r = np.empty(n_bl * n_freq * 4, dtype=np.float64)
    r_view = r.reshape(n_bl, n_freq, 4)
    r_view[:, :, 0] = d_pp.real
    r_view[:, :, 1] = d_pp.imag
    r_view[:, :, 2] = d_qq.real
    r_view[:, :, 3] = d_qq.imag
    return r


def residual_2x2_np(J, vis_obs, vis_model, ant1, ant2):
    """Full 2x2 residual.  Returns (n_bl*8,) float64."""
    n_bl = vis_obs.shape[0]
    Ji = J[ant1]                        # (n_bl, 2, 2)
    JjH = np.conj(J[ant2]).transpose(0, 2, 1)
    pred = np.einsum("bij,bjk,bkl->bil", Ji, vis_model, JjH)
    d = vis_obs - pred
    r = np.empty(n_bl * 8, dtype=np.float64)
    r_view = r.reshape(n_bl, 8)
    r_view[:, 0] = d[:, 0, 0].real;  r_view[:, 1] = d[:, 0, 0].imag
    r_view[:, 2] = d[:, 0, 1].real;  r_view[:, 3] = d[:, 0, 1].imag
    r_view[:, 4] = d[:, 1, 0].real;  r_view[:, 5] = d[:, 1, 0].imag
    r_view[:, 6] = d[:, 1, 1].real;  r_view[:, 7] = d[:, 1, 1].imag
    return r


def residual_cross_np(J, vis_obs, vis_model, ant1, ant2):
    """Cross-hand (pq, qp).  Returns (n_bl*4,) float64."""
    n_bl = vis_obs.shape[0]
    d_pq = (vis_obs[:, 0, 1]
            - J[ant1, 0, 0] * vis_model[:, 0, 1] * np.conj(J[ant2, 1, 1]))
    d_qp = (vis_obs[:, 1, 0]
            - J[ant1, 1, 1] * vis_model[:, 1, 0] * np.conj(J[ant2, 0, 0]))
    r = np.empty(n_bl * 4, dtype=np.float64)
    r[0::4] = d_pq.real
    r[1::4] = d_pq.imag
    r[2::4] = d_qp.real
    r[3::4] = d_qp.imag
    return r


def residual_cross_freq_np(J, vis_obs, vis_model, ant1, ant2):
    """Cross-hand freq-dependent.  Returns (n_bl*n_f*4,)."""
    n_bl, n_freq = vis_obs.shape[:2]
    d_pq = (vis_obs[:, :, 0, 1]
            - J[ant1, :, 0, 0] * vis_model[:, :, 0, 1]
            * np.conj(J[ant2, :, 1, 1]))
    d_qp = (vis_obs[:, :, 1, 0]
            - J[ant1, :, 1, 1] * vis_model[:, :, 1, 0]
            * np.conj(J[ant2, :, 0, 0]))
    r = np.empty(n_bl * n_freq * 4, dtype=np.float64)
    r_view = r.reshape(n_bl, n_freq, 4)
    r_view[:, :, 0] = d_pq.real
    r_view[:, :, 1] = d_pq.imag
    r_view[:, :, 2] = d_qp.real
    r_view[:, :, 3] = d_qp.imag
    return r


# ---------------------------------------------------------------------------
# Sum-of-squares cost (scalar) — for L-BFGS-B style optimisers
# ---------------------------------------------------------------------------

def cost_diag_np(J, vis_obs, vis_model, ant1, ant2):
    """Scalar cost = 0.5 * sum(residual^2) for parallel hands."""
    r = residual_diag_np(J, vis_obs, vis_model, ant1, ant2)
    return 0.5 * np.dot(r, r)


def cost_diag_freq_np(J, vis_obs, vis_model, ant1, ant2):
    r = residual_diag_freq_np(J, vis_obs, vis_model, ant1, ant2)
    return 0.5 * np.dot(r, r)


def cost_2x2_np(J, vis_obs, vis_model, ant1, ant2):
    r = residual_2x2_np(J, vis_obs, vis_model, ant1, ant2)
    return 0.5 * np.dot(r, r)


def cost_cross_np(J, vis_obs, vis_model, ant1, ant2):
    r = residual_cross_np(J, vis_obs, vis_model, ant1, ant2)
    return 0.5 * np.dot(r, r)


def cost_cross_freq_np(J, vis_obs, vis_model, ant1, ant2):
    r = residual_cross_freq_np(J, vis_obs, vis_model, ant1, ant2)
    return 0.5 * np.dot(r, r)
