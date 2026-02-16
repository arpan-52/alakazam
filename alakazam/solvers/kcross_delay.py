"""
ALAKAZAM Kcross (Cross-hand Delay) Solver.

Solves for Kcross(ν) = diag(1, exp(-2πi τ_pq ν)) per antenna.
Uses cross-hand correlations (XY,YX / RL,LR).
Frequency axis kept — never averaged.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
from numba import njit, prange
from typing import Dict, Optional
from . import JonesSolver, build_antenna_graph, bfs_order
from .k_delay import _unwrap_phase, _fit_delay_from_phase
import logging

logger = logging.getLogger("alakazam")


@njit(parallel=True, cache=True)
def _kcross_residual(params, vis_obs, vis_model, ant1, ant2,
                     n_ant, ref_ant, working_ants, freq):
    """Residual for Kcross solver.

    Packing: others=[τ_pq] per antenna (ns). Ref has τ=0.
    Kcross = diag(1, exp(-2πi τ ν))
    Cross-hand residuals only.
    """
    n_bl = vis_obs.shape[0]
    n_freq = vis_obs.shape[1]
    n_working = len(working_ants)

    # Unpack delays
    tau = np.zeros(n_ant, dtype=np.float64)
    idx = 0
    for w in range(n_working):
        a = working_ants[w]
        if a == ref_ant:
            continue
        tau[a] = params[idx]
        idx += 1

    # Cross-hand residuals
    residual = np.empty(n_bl * n_freq * 4, dtype=np.float64)
    for bl in prange(n_bl):
        a1 = ant1[bl]
        a2 = ant2[bl]
        for f in range(n_freq):
            # J[a,0,0] = 1, J[a,1,1] = exp(-2πi τ ν)
            J_a1_11 = np.exp(-2j * np.pi * tau[a1] * 1e-9 * freq[f])
            J_a2_11 = np.exp(-2j * np.pi * tau[a2] * 1e-9 * freq[f])

            # V_pq = 1 * M_pq * conj(J_j[1,1])
            diff_pq = vis_obs[bl, f, 0, 1] - vis_model[bl, f, 0, 1] * np.conj(J_a2_11)
            # V_qp = J_i[1,1] * M_qp * 1
            diff_qp = vis_obs[bl, f, 1, 0] - J_a1_11 * vis_model[bl, f, 1, 0]

            r_idx = (bl * n_freq + f) * 4
            residual[r_idx + 0] = diff_pq.real
            residual[r_idx + 1] = diff_pq.imag
            residual[r_idx + 2] = diff_qp.real
            residual[r_idx + 3] = diff_qp.imag

    return residual


class KcrossDelaySolver(JonesSolver):
    """Kcross (Cross-hand Delay) solver."""

    jones_type = "Kcross"
    can_avg_freq = False
    correlations = "cross"
    ref_constraint = "identity"

    def n_params(self, n_working, n_freq=1, phase_only=False):
        return n_working - 1  # one delay per non-ref antenna

    def chain_initial_guess(self, vis_avg, model_avg, ant1, ant2,
                            n_ant, ref_ant, working_ants, freq=None,
                            phase_only=False):
        """Chain solver: phase slope fit on cross-hand correlations."""
        if freq is None:
            raise ValueError("Kcross solver requires freq array")

        tau = np.zeros(n_ant, dtype=np.float64)
        graph = build_antenna_graph(ant1, ant2, n_ant)
        order = bfs_order(graph, ref_ant, working_ants)

        for (ant, parent, bl_idx, is_parent_ant1) in order:
            V = vis_avg[bl_idx]   # (n_freq, 2, 2)
            M = model_avg[bl_idx]

            # Use QP correlation: V_qp = exp(-2πiτ_ant ν) * M_qp * 1
            ratio = np.zeros(len(freq), dtype=np.complex128)
            for f in range(len(freq)):
                m_val = M[f, 1, 0]  # M_qp
                if np.abs(m_val) > 1e-20:
                    ratio[f] = V[f, 1, 0] / m_val
                else:
                    m_val2 = M[f, 0, 1]  # M_pq
                    if np.abs(m_val2) > 1e-20:
                        ratio[f] = np.conj(V[f, 0, 1] / m_val2)
                    else:
                        ratio[f] = 1.0

            # Remove parent delay contribution
            parent_tau = tau[parent] * 1e-9
            for f in range(len(freq)):
                if is_parent_ant1:
                    ratio[f] *= np.exp(2j * np.pi * parent_tau * freq[f])
                else:
                    ratio[f] *= np.exp(2j * np.pi * parent_tau * freq[f])

            phase = np.angle(ratio)
            phase_uw = _unwrap_phase(phase)
            tau_fit = _fit_delay_from_phase(phase_uw, freq)

            tau[ant] = tau_fit

        return self._pack_delay(tau, ref_ant, working_ants)

    def _pack_delay(self, tau, ref_ant, working_ants):
        params = []
        for a in working_ants:
            if a == ref_ant:
                continue
            params.append(tau[a])
        return np.array(params, dtype=np.float64)

    def _unpack_delay(self, params, n_ant, ref_ant, working_ants):
        tau = np.full(n_ant, np.nan, dtype=np.float64)
        tau[ref_ant] = 0.0
        idx = 0
        for a in working_ants:
            if a == ref_ant:
                continue
            tau[a] = params[idx]
            idx += 1
        return tau

    def pack_params(self, jones, n_ant, ref_ant, working_ants, phase_only=False):
        raise NotImplementedError("Use _pack_delay for Kcross solver")

    def unpack_params(self, params, n_ant, ref_ant, working_ants,
                      freq=None, phase_only=False):
        tau = self._unpack_delay(params, n_ant, ref_ant, working_ants)
        if freq is None:
            raise ValueError("Kcross solver requires freq for unpack_params")
        from ..jones import crossdelay_to_jones
        return crossdelay_to_jones(tau, freq)

    def residual_func(self, params, vis_avg, model_avg, ant1, ant2,
                      n_ant, ref_ant, working_ants, freq=None,
                      phase_only=False):
        if freq is None:
            raise ValueError("Kcross solver requires freq")
        return _kcross_residual(params, vis_avg, model_avg, ant1, ant2,
                                n_ant, ref_ant, working_ants, freq)

    def get_native_params(self, params, n_ant, ref_ant, working_ants, freq=None):
        tau = self._unpack_delay(params, n_ant, ref_ant, working_ants)
        return {"cross_delay": tau}
