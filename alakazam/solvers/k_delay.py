"""
ALAKAZAM K (Delay) Solver.

Solves for antenna-based delays: K(ν) = diag(exp(-2πi τ_p ν), exp(-2πi τ_q ν))
Uses parallel-hand correlations (XX,YY / RR,LL).
Frequency axis is kept — never averaged.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
from numba import njit, prange
from typing import Dict, Optional, Tuple
from . import JonesSolver, build_antenna_graph, bfs_order
import logging

logger = logging.getLogger("alakazam")


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _fit_delay_from_phase(phase, freq):
    """Linear fit: phase = -2π τ ν → extract τ in nanoseconds.

    phase: (n_freq,) float64  — unwrapped phase in radians
    freq:  (n_freq,) float64  — Hz
    Returns: delay in nanoseconds
    """
    n = len(freq)
    if n < 2:
        return 0.0
    # Weighted least squares: φ = a · ν + b
    # We want a = -2π τ  →  τ = -a / (2π)
    sum_f = 0.0
    sum_p = 0.0
    sum_ff = 0.0
    sum_fp = 0.0
    for i in range(n):
        sum_f += freq[i]
        sum_p += phase[i]
        sum_ff += freq[i] * freq[i]
        sum_fp += freq[i] * phase[i]
    denom = n * sum_ff - sum_f * sum_f
    if abs(denom) < 1e-30:
        return 0.0
    slope = (n * sum_fp - sum_f * sum_p) / denom
    tau_sec = -slope / (2.0 * np.pi)
    return tau_sec * 1e9  # seconds → nanoseconds


@njit(cache=True)
def _unwrap_phase(phase):
    """Simple phase unwrapping."""
    n = len(phase)
    out = np.empty(n, dtype=np.float64)
    out[0] = phase[0]
    for i in range(1, n):
        diff = phase[i] - phase[i - 1]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        out[i] = out[i - 1] + diff
    return out


@njit(parallel=True, cache=True)
def _k_residual(delay_params, vis_obs, vis_model, ant1, ant2,
                n_ant, ref_ant, working_ants, freq):
    """Residual function for K delay solver.

    delay_params: flat real vector [(n_working-1) * 2]
    vis_obs:      (n_bl, n_freq, 2, 2) complex128
    vis_model:    (n_bl, n_freq, 2, 2) complex128
    freq:         (n_freq,) float64 Hz
    Returns:      (n_bl * n_freq * 4,) float64
    """
    n_bl = vis_obs.shape[0]
    n_freq = vis_obs.shape[1]
    n_working = len(working_ants)

    # Build delay array (n_ant, 2)
    delay = np.zeros((n_ant, 2), dtype=np.float64)
    idx = 0
    for w in range(n_working):
        a = working_ants[w]
        if a == ref_ant:
            continue
        delay[a, 0] = delay_params[idx]
        delay[a, 1] = delay_params[idx + 1]
        idx += 2

    # Compute residuals
    residual = np.empty(n_bl * n_freq * 4, dtype=np.float64)
    for bl in prange(n_bl):
        a1 = ant1[bl]
        a2 = ant2[bl]
        for f in range(n_freq):
            # K Jones elements
            k_a1_p = np.exp(-2j * np.pi * delay[a1, 0] * 1e-9 * freq[f])
            k_a1_q = np.exp(-2j * np.pi * delay[a1, 1] * 1e-9 * freq[f])
            k_a2_p = np.exp(-2j * np.pi * delay[a2, 0] * 1e-9 * freq[f])
            k_a2_q = np.exp(-2j * np.pi * delay[a2, 1] * 1e-9 * freq[f])

            # pp residual
            diff_pp = vis_obs[bl, f, 0, 0] - k_a1_p * vis_model[bl, f, 0, 0] * np.conj(k_a2_p)
            # qq residual
            diff_qq = vis_obs[bl, f, 1, 1] - k_a1_q * vis_model[bl, f, 1, 1] * np.conj(k_a2_q)

            r_idx = (bl * n_freq + f) * 4
            residual[r_idx + 0] = diff_pp.real
            residual[r_idx + 1] = diff_pp.imag
            residual[r_idx + 2] = diff_qq.real
            residual[r_idx + 3] = diff_qq.imag

    return residual


class KDelaySolver(JonesSolver):
    """K (Delay) solver."""

    jones_type = "K"
    can_avg_freq = False
    correlations = "parallel"
    ref_constraint = "identity"

    def n_params(self, n_working: int, n_freq: int = 1,
                 phase_only: bool = False) -> int:
        return (n_working - 1) * 2

    def chain_initial_guess(
        self, vis_avg, model_avg, ant1, ant2,
        n_ant, ref_ant, working_ants, freq=None, phase_only=False,
    ):
        """Chain solver: phase slope fit on ref baselines, BFS propagation."""
        if freq is None:
            raise ValueError("K solver requires freq array")

        delay = np.zeros((n_ant, 2), dtype=np.float64)
        graph = build_antenna_graph(ant1, ant2, n_ant)
        order = bfs_order(graph, ref_ant, working_ants)

        for (ant, parent, bl_idx, is_parent_ant1) in order:
            # Get visibility for this baseline
            if vis_avg.ndim == 4:
                V = vis_avg[bl_idx]   # (n_freq, 2, 2)
                M = model_avg[bl_idx]
            else:
                continue

            for pol in range(2):
                # ratio = V / M
                ratio = np.zeros(len(freq), dtype=np.complex128)
                for f in range(len(freq)):
                    m_val = M[f, pol, pol]
                    if np.abs(m_val) > 1e-20:
                        ratio[f] = V[f, pol, pol] / m_val
                    else:
                        ratio[f] = 0.0

                # Remove parent delay contribution
                parent_delay = delay[parent, pol] * 1e-9  # ns → s
                for f in range(len(freq)):
                    if is_parent_ant1:
                        # parent is ant1: V = K_parent * M * K_ant^H
                        # ratio = K_parent * K_ant^H = exp(-2πi(τ_par - τ_ant)ν)
                        ratio[f] *= np.exp(2j * np.pi * parent_delay * freq[f])
                    else:
                        # parent is ant2: V = K_ant * M * K_parent^H
                        # ratio = K_ant * K_parent^H = exp(-2πi(τ_ant - τ_par)ν)
                        ratio[f] *= np.exp(2j * np.pi * parent_delay * freq[f])

                # Extract phase and fit delay
                phase = np.angle(ratio)
                phase_uw = _unwrap_phase(phase)
                tau = _fit_delay_from_phase(phase_uw, freq)

                if is_parent_ant1:
                    # ratio after parent removal ≈ exp(+2πi τ_ant ν)
                    delay[ant, pol] = -tau
                else:
                    delay[ant, pol] = tau

        # Pack
        return self._pack_delay(delay, ref_ant, working_ants)

    def _pack_delay(self, delay, ref_ant, working_ants):
        """Pack delay array to parameter vector."""
        params = []
        for a in working_ants:
            if a == ref_ant:
                continue
            params.append(delay[a, 0])
            params.append(delay[a, 1])
        return np.array(params, dtype=np.float64)

    def _unpack_delay(self, params, n_ant, ref_ant, working_ants):
        """Unpack parameter vector to delay array."""
        delay = np.full((n_ant, 2), np.nan, dtype=np.float64)
        delay[ref_ant, :] = 0.0
        idx = 0
        for a in working_ants:
            if a == ref_ant:
                continue
            delay[a, 0] = params[idx]
            delay[a, 1] = params[idx + 1]
            idx += 2
        return delay

    def pack_params(self, jones, n_ant, ref_ant, working_ants, phase_only=False):
        # Extract delays from Jones (not typically called directly)
        raise NotImplementedError("Use _pack_delay for K solver")

    def unpack_params(self, params, n_ant, ref_ant, working_ants,
                      freq=None, phase_only=False):
        """Unpack to Jones matrices (n_ant, n_freq, 2, 2)."""
        delay = self._unpack_delay(params, n_ant, ref_ant, working_ants)
        if freq is None:
            raise ValueError("K solver requires freq for unpack_params")
        from ..jones import delay_to_jones
        return delay_to_jones(delay, freq)

    def residual_func(self, params, vis_avg, model_avg, ant1, ant2,
                      n_ant, ref_ant, working_ants, freq=None,
                      phase_only=False):
        if freq is None:
            raise ValueError("K solver requires freq")
        return _k_residual(params, vis_avg, model_avg, ant1, ant2,
                           n_ant, ref_ant, working_ants, freq)

    def get_native_params(self, params, n_ant, ref_ant, working_ants, freq=None):
        delay = self._unpack_delay(params, n_ant, ref_ant, working_ants)
        return {"delay": delay}
