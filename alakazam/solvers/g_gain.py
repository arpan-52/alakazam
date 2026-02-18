"""ALAKAZAM G (Gain) Solver.

Solves for per-antenna complex gain (amplitude + phase) per polarisation.
Frequency-independent diagonal Jones: J[a] = diag(g_p, g_q)

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import least_squares

from . import JonesSolver, build_antenna_graph, bfs_order
from ..jones.constructors import gain_to_jones
from ..jones.algebra import compute_residual_diag

logger = logging.getLogger("alakazam")


class GGainSolver(JonesSolver):
    jones_type = "G"

    def solve(
        self,
        vis_obs: np.ndarray,     # (n_bl, [n_chan,] 2, 2) or (n_bl, 2, 2) after averaging
        vis_model: np.ndarray,
        ant1: np.ndarray,
        ant2: np.ndarray,
        freqs: np.ndarray,
        n_ant: int,
        init_jones: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:

        # Average freq if vis is freq-resolved
        if vis_obs.ndim == 4:
            obs   = vis_obs.mean(axis=1)
            model = vis_model.mean(axis=1)
        else:
            obs, model = vis_obs, vis_model

        # BFS initial estimate from cross-multiplication
        adj   = build_antenna_graph(ant1, ant2, n_ant)
        order = bfs_order(adj, self.ref_ant)

        amp_init   = np.ones((n_ant, 2), dtype=np.float64)
        phase_init = np.zeros((n_ant, 2), dtype=np.float64)

        if init_jones is not None and init_jones.ndim == 3:
            amp_init   = np.abs(np.diagonal(init_jones, axis1=-2, axis2=-1))
            phase_init = np.angle(np.diagonal(init_jones, axis1=-2, axis2=-1))

        def _residual(x):
            amp_   = x[:n_ant * 2].reshape(n_ant, 2)
            if self.phase_only:
                ph = x[n_ant * 2:].reshape(n_ant, 2)
                amp_fixed = np.ones_like(amp_)
                J = gain_to_jones(amp_fixed, ph)
            else:
                ph = x[n_ant * 2:].reshape(n_ant, 2)
                J = gain_to_jones(amp_, ph)
            J[self.ref_ant] = np.eye(2, dtype=np.complex128)
            return compute_residual_diag(J, obs, model, ant1, ant2)

        if self.phase_only:
            x0 = np.concatenate([amp_init.flatten(), phase_init.flatten()])
        else:
            x0 = np.concatenate([amp_init.flatten(), phase_init.flatten()])

        result = least_squares(
            _residual, x0,
            method="lm",
            max_nfev=self.max_iter * len(x0),
            ftol=self.tol, xtol=self.tol, gtol=self.tol,
        )

        n2 = n_ant * 2
        amp   = result.x[:n2].reshape(n_ant, 2)
        phase = result.x[n2:].reshape(n_ant, 2)

        if self.phase_only:
            amp = np.ones((n_ant, 2), dtype=np.float64)

        amp[self.ref_ant]   = 1.0
        phase[self.ref_ant] = 0.0
        J = gain_to_jones(amp, phase)

        return {
            "jones":         J,
            "native_params": {"type": "G", "amp": amp, "phase": phase},
            "converged":     result.success,
            "n_iter":        result.nfev,
            "cost":          float(result.cost),
        }
