"""
ALAKAZAM D (Polarization Leakage) Solver.

Solves for D = [[1, d_pq], [d_qp, 1]] per antenna.
Uses ALL FOUR correlations (pp, pq, qp, qq).
Proper chain initial guess from cross-hand data.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
from numba import njit, prange
from typing import Dict, Optional
from . import JonesSolver, build_antenna_graph, bfs_order
import logging

logger = logging.getLogger("alakazam")


@njit(parallel=True, cache=True)
def _d_residual(params, vis_obs, vis_model, ant1, ant2,
                n_ant, ref_ant, working_ants):
    """Residual for D leakage solver.

    Packing: ref=[Re(d_ref_qp), Im(d_ref_qp)] (d_ref_pq=0 by constraint),
             others=[Re(d_pq), Im(d_pq), Re(d_qp), Im(d_qp)] each.
    Full 2x2 RIME residual.
    """
    n_bl = vis_obs.shape[0]
    n_working = len(working_ants)

    # Unpack D matrices (n_ant, 2, 2)
    D = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    for a in range(n_ant):
        D[a, 0, 0] = 1.0 + 0j
        D[a, 1, 1] = 1.0 + 0j

    # Reference: d_pq = 0 (constraint), d_qp = free
    D[ref_ant, 0, 1] = 0.0 + 0j
    D[ref_ant, 1, 0] = params[0] + 1j * params[1]

    idx = 2
    for w in range(n_working):
        a = working_ants[w]
        if a == ref_ant:
            continue
        D[a, 0, 1] = params[idx] + 1j * params[idx + 1]      # d_pq
        D[a, 1, 0] = params[idx + 2] + 1j * params[idx + 3]  # d_qp
        idx += 4

    # Full 2x2 residuals: V_obs - D_i M D_j^H
    residual = np.empty(n_bl * 8, dtype=np.float64)
    for bl in prange(n_bl):
        a1 = ant1[bl]
        a2 = ant2[bl]

        # D_i * M
        Di = D[a1]
        M = vis_model[bl]
        DM = np.zeros((2, 2), dtype=np.complex128)
        DM[0, 0] = Di[0, 0] * M[0, 0] + Di[0, 1] * M[1, 0]
        DM[0, 1] = Di[0, 0] * M[0, 1] + Di[0, 1] * M[1, 1]
        DM[1, 0] = Di[1, 0] * M[0, 0] + Di[1, 1] * M[1, 0]
        DM[1, 1] = Di[1, 0] * M[0, 1] + Di[1, 1] * M[1, 1]

        # (D_i M) * D_j^H
        Dj_H = np.zeros((2, 2), dtype=np.complex128)
        Dj_H[0, 0] = np.conj(D[a2, 0, 0])
        Dj_H[0, 1] = np.conj(D[a2, 1, 0])
        Dj_H[1, 0] = np.conj(D[a2, 0, 1])
        Dj_H[1, 1] = np.conj(D[a2, 1, 1])

        model_pred = np.zeros((2, 2), dtype=np.complex128)
        model_pred[0, 0] = DM[0, 0] * Dj_H[0, 0] + DM[0, 1] * Dj_H[1, 0]
        model_pred[0, 1] = DM[0, 0] * Dj_H[0, 1] + DM[0, 1] * Dj_H[1, 1]
        model_pred[1, 0] = DM[1, 0] * Dj_H[0, 0] + DM[1, 1] * Dj_H[1, 0]
        model_pred[1, 1] = DM[1, 0] * Dj_H[0, 1] + DM[1, 1] * Dj_H[1, 1]

        diff = vis_obs[bl] - model_pred
        r_idx = bl * 8
        residual[r_idx + 0] = diff[0, 0].real
        residual[r_idx + 1] = diff[0, 0].imag
        residual[r_idx + 2] = diff[0, 1].real
        residual[r_idx + 3] = diff[0, 1].imag
        residual[r_idx + 4] = diff[1, 0].real
        residual[r_idx + 5] = diff[1, 0].imag
        residual[r_idx + 6] = diff[1, 1].real
        residual[r_idx + 7] = diff[1, 1].imag

    return residual


class DLeakageSolver(JonesSolver):
    """D (Polarization Leakage) solver."""

    jones_type = "D"
    can_avg_freq = True
    correlations = "full"
    ref_constraint = "leakage_zero"

    def n_params(self, n_working, n_freq=1, phase_only=False):
        return 2 + (n_working - 1) * 4  # ref: 2 (d_qp only), others: 4

    def chain_initial_guess(self, vis_avg, model_avg, ant1, ant2,
                            n_ant, ref_ant, working_ants, freq=None,
                            phase_only=False):
        """Chain solver using cross-hand correlations.

        d_i_pq ≈ (V_pq - M_pq) / M_qq   on baseline (ref, i)
        d_i_qp ≈ (V_qp - M_qp) / M_pp   on baseline (ref, i)
        """
        d_pq = np.zeros(n_ant, dtype=np.complex128)
        d_qp = np.zeros(n_ant, dtype=np.complex128)

        graph = build_antenna_graph(ant1, ant2, n_ant)
        order = bfs_order(graph, ref_ant, working_ants)

        for (ant, parent, bl_idx, is_parent_ant1) in order:
            V = vis_avg[bl_idx]  # (2, 2)
            M = model_avg[bl_idx]

            # For baselines directly to reference (parent == ref_ant):
            if parent == ref_ant:
                # d_ref_pq = 0 by constraint
                M_pp = M[0, 0]
                M_qq = M[1, 1]

                if np.abs(M_qq) > 1e-20:
                    if is_parent_ant1:
                        # ref is ant1: V_pq = D_ref[0,:] M D_ant[1,:]^H
                        # ≈ M_pq + d_ant_pq * M_qq (leading order)
                        d_pq[ant] = (V[0, 1] - M[0, 1]) / M_qq
                    else:
                        # ref is ant2: V_pq = D_ant[0,:] M D_ref[1,:]^H
                        d_pq[ant] = (V[0, 1] - M[0, 1]) / M_qq

                if np.abs(M_pp) > 1e-20:
                    if is_parent_ant1:
                        d_qp[ant] = (V[1, 0] - M[1, 0]) / M_pp
                    else:
                        d_qp[ant] = (V[1, 0] - M[1, 0]) / M_pp
            else:
                # Propagation through BFS — use parent's known leakage
                M_pp = M[0, 0]
                M_qq = M[1, 1]

                if np.abs(M_qq) > 1e-20:
                    raw = (V[0, 1] - M[0, 1]) / M_qq
                    if is_parent_ant1:
                        d_pq[ant] = raw - d_pq[parent]
                    else:
                        d_pq[ant] = raw - np.conj(d_pq[parent])

                if np.abs(M_pp) > 1e-20:
                    raw = (V[1, 0] - M[1, 0]) / M_pp
                    if is_parent_ant1:
                        d_qp[ant] = raw - d_qp[parent]
                    else:
                        d_qp[ant] = raw - np.conj(d_qp[parent])

        # Estimate d_ref_qp from first working antenna
        if len(order) > 0:
            first_ant = order[0][0]
            bl_idx = order[0][2]
            M = model_avg[bl_idx]
            V = vis_avg[bl_idx]
            M_pp = M[0, 0]
            if np.abs(M_pp) > 1e-20:
                # With d_first known, isolate d_ref_qp
                d_qp[ref_ant] = (V[1, 0] - M[1, 0]) / M_pp - d_qp[first_ant]

        return self._pack_leakage(d_pq, d_qp, ref_ant, working_ants)

    def _pack_leakage(self, d_pq, d_qp, ref_ant, working_ants):
        params = []
        # Reference: only d_qp (d_pq = 0 by constraint)
        params.append(d_qp[ref_ant].real)
        params.append(d_qp[ref_ant].imag)
        # Others
        for a in working_ants:
            if a == ref_ant:
                continue
            params.append(d_pq[a].real)
            params.append(d_pq[a].imag)
            params.append(d_qp[a].real)
            params.append(d_qp[a].imag)
        return np.array(params, dtype=np.float64)

    def _unpack_leakage(self, params, n_ant, ref_ant, working_ants):
        d_pq = np.full(n_ant, np.nan + 0j, dtype=np.complex128)
        d_qp = np.full(n_ant, np.nan + 0j, dtype=np.complex128)

        d_pq[ref_ant] = 0.0
        d_qp[ref_ant] = params[0] + 1j * params[1]

        idx = 2
        for a in working_ants:
            if a == ref_ant:
                continue
            d_pq[a] = params[idx] + 1j * params[idx + 1]
            d_qp[a] = params[idx + 2] + 1j * params[idx + 3]
            idx += 4

        return d_pq, d_qp

    def pack_params(self, jones, n_ant, ref_ant, working_ants, phase_only=False):
        d_pq = np.zeros(n_ant, dtype=np.complex128)
        d_qp = np.zeros(n_ant, dtype=np.complex128)
        for a in working_ants:
            d_pq[a] = jones[a, 0, 1]
            d_qp[a] = jones[a, 1, 0]
        return self._pack_leakage(d_pq, d_qp, ref_ant, working_ants)

    def unpack_params(self, params, n_ant, ref_ant, working_ants,
                      freq=None, phase_only=False):
        d_pq, d_qp = self._unpack_leakage(params, n_ant, ref_ant, working_ants)
        from ..jones import leakage_to_jones
        return leakage_to_jones(d_pq, d_qp, n_ant)

    def residual_func(self, params, vis_avg, model_avg, ant1, ant2,
                      n_ant, ref_ant, working_ants, freq=None,
                      phase_only=False):
        return _d_residual(params, vis_avg, model_avg, ant1, ant2,
                           n_ant, ref_ant, working_ants)

    def get_native_params(self, params, n_ant, ref_ant, working_ants, freq=None):
        d_pq, d_qp = self._unpack_leakage(params, n_ant, ref_ant, working_ants)
        return {"d_pq": d_pq, "d_qp": d_qp}
