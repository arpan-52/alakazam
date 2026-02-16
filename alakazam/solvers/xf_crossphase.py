"""
ALAKAZAM Xf (Cross-hand Phase) Solver.

Solves for Xf = diag(1, exp(iφ_pq)) per antenna.
CORRECTED standard diagonal form (not the old off-diagonal form).
Uses cross-hand correlations (XY,YX / RL,LR).

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
from numba import njit, prange
from typing import Dict, Optional
from . import JonesSolver, build_antenna_graph, bfs_order
import logging

logger = logging.getLogger("alakazam")


@njit(parallel=True, cache=True)
def _xf_residual(params, vis_obs, vis_model, ant1, ant2,
                 n_ant, ref_ant, working_ants):
    """Residual for Xf cross-hand phase solver.

    Packing: others=[φ_pq] per antenna. Ref has φ=0 (no params).
    Xf = diag(1, exp(iφ))
    Cross-hand residuals only.
    """
    n_bl = vis_obs.shape[0]
    n_working = len(working_ants)

    # Build Jones (diagonal): J[a,0,0]=1, J[a,1,1]=exp(iφ)
    J_00 = np.ones(n_ant, dtype=np.complex128)
    J_11 = np.ones(n_ant, dtype=np.complex128)
    J_11[ref_ant] = 1.0 + 0j

    idx = 0
    for w in range(n_working):
        a = working_ants[w]
        if a == ref_ant:
            continue
        J_11[a] = np.exp(1j * params[idx])
        idx += 1

    # Cross-hand residuals: pq and qp
    # V_pq = J_i[0,0] * M_pq * conj(J_j[1,1])
    # V_qp = J_i[1,1] * M_qp * conj(J_j[0,0])
    residual = np.empty(n_bl * 4, dtype=np.float64)
    for bl in prange(n_bl):
        a1 = ant1[bl]
        a2 = ant2[bl]
        diff_pq = vis_obs[bl, 0, 1] - J_00[a1] * vis_model[bl, 0, 1] * np.conj(J_11[a2])
        diff_qp = vis_obs[bl, 1, 0] - J_11[a1] * vis_model[bl, 1, 0] * np.conj(J_00[a2])
        r_idx = bl * 4
        residual[r_idx + 0] = diff_pq.real
        residual[r_idx + 1] = diff_pq.imag
        residual[r_idx + 2] = diff_qp.real
        residual[r_idx + 3] = diff_qp.imag

    return residual


class XfCrossPhaseSolver(JonesSolver):
    """Xf (Cross-hand Phase) solver."""

    jones_type = "Xf"
    can_avg_freq = True
    correlations = "cross"
    ref_constraint = "identity"

    def n_params(self, n_working, n_freq=1, phase_only=False):
        return n_working - 1  # one phase per non-ref antenna

    def chain_initial_guess(self, vis_avg, model_avg, ant1, ant2,
                            n_ant, ref_ant, working_ants, freq=None,
                            phase_only=False):
        """Chain solver: extract cross-hand phase from ref baselines."""
        phi = np.zeros(n_ant, dtype=np.float64)
        graph = build_antenna_graph(ant1, ant2, n_ant)
        order = bfs_order(graph, ref_ant, working_ants)

        for (ant, parent, bl_idx, is_parent_ant1) in order:
            V = vis_avg[bl_idx]
            M = model_avg[bl_idx]

            # Use XY correlation: V_pq = J_i[0,0] * M_pq * conj(J_j[1,1])
            # For baseline to ref (φ_ref=0): V_pq = M_pq * conj(exp(iφ_ant))
            # → φ_ant = -angle(V_pq / M_pq)
            M_pq = M[0, 1]
            if np.abs(M_pq) > 1e-20:
                ratio = V[0, 1] / M_pq
                if is_parent_ant1:
                    # parent is ant1: V_pq = 1 * M_pq * conj(exp(iφ_ant))
                    # ratio = exp(-iφ_ant) → φ_ant = -angle(ratio)
                    phi[ant] = -np.angle(ratio) + phi[parent] * 0  # parent's effect already in data
                    # More precisely with parent known:
                    # Correct for parent: ratio' = ratio / (J_parent_00 * conj(J_parent_11))
                    # But J_00=1 for Xf, so: ratio' = ratio * exp(iφ_parent)
                    phi[ant] = -np.angle(ratio * np.exp(1j * phi[parent]))
                else:
                    # ant is ant1: V_pq = 1 * M_pq * conj(exp(iφ_parent))
                    # This gives φ_parent info — use qp instead
                    M_qp = M[1, 0]
                    if np.abs(M_qp) > 1e-20:
                        ratio_qp = V[1, 0] / M_qp
                        # V_qp = exp(iφ_ant) * M_qp * 1
                        phi[ant] = np.angle(ratio_qp / np.exp(1j * phi[parent]))
                    else:
                        phi[ant] = 0.0
            else:
                # Try qp
                M_qp = M[1, 0]
                if np.abs(M_qp) > 1e-20:
                    ratio_qp = V[1, 0] / M_qp
                    if not is_parent_ant1:
                        phi[ant] = np.angle(ratio_qp * np.exp(-1j * phi[parent]))
                    else:
                        phi[ant] = np.angle(ratio_qp * np.exp(-1j * phi[parent]))
                else:
                    phi[ant] = 0.0

        return self._pack_phase(phi, ref_ant, working_ants)

    def _pack_phase(self, phi, ref_ant, working_ants):
        params = []
        for a in working_ants:
            if a == ref_ant:
                continue
            params.append(phi[a])
        return np.array(params, dtype=np.float64)

    def _unpack_phase(self, params, n_ant, ref_ant, working_ants):
        phi = np.full(n_ant, np.nan, dtype=np.float64)
        phi[ref_ant] = 0.0
        idx = 0
        for a in working_ants:
            if a == ref_ant:
                continue
            phi[a] = params[idx]
            idx += 1
        return phi

    def pack_params(self, jones, n_ant, ref_ant, working_ants, phase_only=False):
        phi = np.zeros(n_ant, dtype=np.float64)
        for a in working_ants:
            phi[a] = np.angle(jones[a, 1, 1])
        return self._pack_phase(phi, ref_ant, working_ants)

    def unpack_params(self, params, n_ant, ref_ant, working_ants,
                      freq=None, phase_only=False):
        phi = self._unpack_phase(params, n_ant, ref_ant, working_ants)
        from ..jones import crossphase_to_jones
        return crossphase_to_jones(phi, n_ant)

    def residual_func(self, params, vis_avg, model_avg, ant1, ant2,
                      n_ant, ref_ant, working_ants, freq=None,
                      phase_only=False):
        return _xf_residual(params, vis_avg, model_avg, ant1, ant2,
                            n_ant, ref_ant, working_ants)

    def get_native_params(self, params, n_ant, ref_ant, working_ants, freq=None):
        phi = self._unpack_phase(params, n_ant, ref_ant, working_ants)
        return {"cross_phase": phi}
