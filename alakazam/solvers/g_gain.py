"""
ALAKAZAM G (Complex Gain) Solver.

Solves for diagonal complex gains: G = diag(g_p, g_q).
This is the ONLY diagonal gain solver. It serves as both traditional gain
(freq_interval=full) and bandpass (freq_interval=4MHz) depending on solint.

Supports full (amplitude+phase) and phase-only modes.
Uses parallel-hand correlations (XX,YY / RR,LL).

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
from numba import njit, prange
from typing import Dict, Optional, Tuple
from . import JonesSolver, build_antenna_graph, bfs_order
import logging

logger = logging.getLogger("alakazam")


@njit(parallel=True, cache=True)
def _g_residual_full(params, vis_obs, vis_model, ant1, ant2,
                     n_ant, ref_ant, working_ants):
    """Residual for full (amplitude + phase) G solve.

    Packing: ref=[A_ref_p, A_ref_q], others=[A_p, φ_p, A_q, φ_q] each.
    """
    n_bl = vis_obs.shape[0]
    n_working = len(working_ants)

    # Unpack gains (n_ant, 2) complex
    g = np.zeros((n_ant, 2), dtype=np.complex128)

    # Reference: amplitude only, phase=0
    g[ref_ant, 0] = params[0] + 0j  # A_ref_p
    g[ref_ant, 1] = params[1] + 0j  # A_ref_q

    idx = 2
    for w in range(n_working):
        a = working_ants[w]
        if a == ref_ant:
            continue
        A_p = params[idx]
        phi_p = params[idx + 1]
        A_q = params[idx + 2]
        phi_q = params[idx + 3]
        g[a, 0] = A_p * np.exp(1j * phi_p)
        g[a, 1] = A_q * np.exp(1j * phi_q)
        idx += 4

    # Residuals: pp and qq only
    residual = np.empty(n_bl * 4, dtype=np.float64)
    for bl in prange(n_bl):
        a1 = ant1[bl]
        a2 = ant2[bl]
        diff_pp = vis_obs[bl, 0, 0] - g[a1, 0] * vis_model[bl, 0, 0] * np.conj(g[a2, 0])
        diff_qq = vis_obs[bl, 1, 1] - g[a1, 1] * vis_model[bl, 1, 1] * np.conj(g[a2, 1])
        r_idx = bl * 4
        residual[r_idx + 0] = diff_pp.real
        residual[r_idx + 1] = diff_pp.imag
        residual[r_idx + 2] = diff_qq.real
        residual[r_idx + 3] = diff_qq.imag

    return residual


@njit(parallel=True, cache=True)
def _g_residual_phase_only(params, vis_obs, vis_model, ant1, ant2,
                           n_ant, ref_ant, working_ants):
    """Residual for phase-only G solve.

    Packing: others=[φ_p, φ_q] each. Ref has no params (identity).
    """
    n_bl = vis_obs.shape[0]
    n_working = len(working_ants)

    g = np.zeros((n_ant, 2), dtype=np.complex128)
    g[ref_ant, 0] = 1.0 + 0j
    g[ref_ant, 1] = 1.0 + 0j

    idx = 0
    for w in range(n_working):
        a = working_ants[w]
        if a == ref_ant:
            continue
        g[a, 0] = np.exp(1j * params[idx])
        g[a, 1] = np.exp(1j * params[idx + 1])
        idx += 2

    residual = np.empty(n_bl * 4, dtype=np.float64)
    for bl in prange(n_bl):
        a1 = ant1[bl]
        a2 = ant2[bl]
        diff_pp = vis_obs[bl, 0, 0] - g[a1, 0] * vis_model[bl, 0, 0] * np.conj(g[a2, 0])
        diff_qq = vis_obs[bl, 1, 1] - g[a1, 1] * vis_model[bl, 1, 1] * np.conj(g[a2, 1])
        r_idx = bl * 4
        residual[r_idx + 0] = diff_pp.real
        residual[r_idx + 1] = diff_pp.imag
        residual[r_idx + 2] = diff_qq.real
        residual[r_idx + 3] = diff_qq.imag

    return residual


class GGainSolver(JonesSolver):
    """G (Diagonal Gain) solver.

    Also serves as bandpass when freq_interval is not 'full'.
    The pipeline handles chunking — this solver just solves one chunk.
    """

    jones_type = "G"
    can_avg_freq = True
    correlations = "parallel"
    ref_constraint = "phase_zero"

    def n_params(self, n_working, n_freq=1, phase_only=False):
        if phase_only:
            return (n_working - 1) * 2
        return 2 + (n_working - 1) * 4  # ref: 2 amps, others: 4 each

    def chain_initial_guess(self, vis_avg, model_avg, ant1, ant2,
                            n_ant, ref_ant, working_ants, freq=None,
                            phase_only=False):
        """Chain solver: g_j = V/M on ref baselines, BFS propagation."""
        g = np.ones((n_ant, 2), dtype=np.complex128)
        graph = build_antenna_graph(ant1, ant2, n_ant)
        order = bfs_order(graph, ref_ant, working_ants)

        for (ant, parent, bl_idx, is_parent_ant1) in order:
            V = vis_avg[bl_idx]  # (2, 2)
            M = model_avg[bl_idx]

            for pol in range(2):
                m_val = M[pol, pol]
                if np.abs(m_val) < 1e-20:
                    g[ant, pol] = 1.0
                    continue

                ratio = V[pol, pol] / m_val

                # Remove parent contribution
                if is_parent_ant1:
                    # V = g_parent * M * conj(g_ant) → ratio / conj(g_parent)
                    # Then conj → g_ant
                    g_parent = g[parent, pol]
                    if np.abs(g_parent) > 1e-20:
                        g[ant, pol] = np.conj(ratio / g_parent)
                    else:
                        g[ant, pol] = 1.0
                else:
                    # V = g_ant * M * conj(g_parent)
                    g_parent = g[parent, pol]
                    if np.abs(g_parent) > 1e-20:
                        g[ant, pol] = ratio / np.conj(g_parent)
                    else:
                        g[ant, pol] = 1.0

        return self._pack_from_complex(g, ref_ant, working_ants, phase_only)

    def _pack_from_complex(self, g, ref_ant, working_ants, phase_only=False):
        """Pack complex gains to parameter vector."""
        params = []
        if not phase_only:
            params.append(np.abs(g[ref_ant, 0]))
            params.append(np.abs(g[ref_ant, 1]))

        for a in working_ants:
            if a == ref_ant:
                continue
            if phase_only:
                params.append(np.angle(g[a, 0]))
                params.append(np.angle(g[a, 1]))
            else:
                params.append(np.abs(g[a, 0]))
                params.append(np.angle(g[a, 0]))
                params.append(np.abs(g[a, 1]))
                params.append(np.angle(g[a, 1]))

        return np.array(params, dtype=np.float64)

    def pack_params(self, jones, n_ant, ref_ant, working_ants, phase_only=False):
        g = np.zeros((n_ant, 2), dtype=np.complex128)
        for a in working_ants:
            g[a, 0] = jones[a, 0, 0]
            g[a, 1] = jones[a, 1, 1]
        return self._pack_from_complex(g, ref_ant, working_ants, phase_only)

    def unpack_params(self, params, n_ant, ref_ant, working_ants,
                      freq=None, phase_only=False):
        """Unpack to (n_ant, 2, 2) Jones matrices."""
        J = np.full((n_ant, 2, 2), np.nan + 0j, dtype=np.complex128)

        if phase_only:
            J[ref_ant, 0, 0] = 1.0
            J[ref_ant, 0, 1] = 0.0
            J[ref_ant, 1, 0] = 0.0
            J[ref_ant, 1, 1] = 1.0
            idx = 0
            for a in working_ants:
                if a == ref_ant:
                    continue
                J[a, 0, 0] = np.exp(1j * params[idx])
                J[a, 0, 1] = 0.0
                J[a, 1, 0] = 0.0
                J[a, 1, 1] = np.exp(1j * params[idx + 1])
                idx += 2
        else:
            J[ref_ant, 0, 0] = params[0]
            J[ref_ant, 0, 1] = 0.0
            J[ref_ant, 1, 0] = 0.0
            J[ref_ant, 1, 1] = params[1]
            idx = 2
            for a in working_ants:
                if a == ref_ant:
                    continue
                A_p, phi_p = params[idx], params[idx + 1]
                A_q, phi_q = params[idx + 2], params[idx + 3]
                J[a, 0, 0] = A_p * np.exp(1j * phi_p)
                J[a, 0, 1] = 0.0
                J[a, 1, 0] = 0.0
                J[a, 1, 1] = A_q * np.exp(1j * phi_q)
                idx += 4

        return J

    def residual_func(self, params, vis_avg, model_avg, ant1, ant2,
                      n_ant, ref_ant, working_ants, freq=None,
                      phase_only=False):
        if phase_only:
            return _g_residual_phase_only(
                params, vis_avg, model_avg, ant1, ant2,
                n_ant, ref_ant, working_ants)
        return _g_residual_full(
            params, vis_avg, model_avg, ant1, ant2,
            n_ant, ref_ant, working_ants)

    def get_native_params(self, params, n_ant, ref_ant, working_ants, freq=None):
        """Extract amplitude and phase arrays."""
        amp = np.full((n_ant, 2), np.nan, dtype=np.float64)
        phase = np.full((n_ant, 2), np.nan, dtype=np.float64)

        # Determine if phase_only (heuristic: check param count)
        n_working = len(working_ants)
        expected_full = 2 + (n_working - 1) * 4
        phase_only = len(params) != expected_full

        if phase_only:
            amp[ref_ant] = [1.0, 1.0]
            phase[ref_ant] = [0.0, 0.0]
            idx = 0
            for a in working_ants:
                if a == ref_ant:
                    continue
                amp[a] = [1.0, 1.0]
                phase[a, 0] = params[idx]
                phase[a, 1] = params[idx + 1]
                idx += 2
        else:
            amp[ref_ant, 0] = params[0]
            amp[ref_ant, 1] = params[1]
            phase[ref_ant] = [0.0, 0.0]
            idx = 2
            for a in working_ants:
                if a == ref_ant:
                    continue
                amp[a, 0] = params[idx]
                phase[a, 0] = params[idx + 1]
                amp[a, 1] = params[idx + 2]
                phase[a, 1] = params[idx + 3]
                idx += 4

        return {"amplitude": amp, "phase": phase}
