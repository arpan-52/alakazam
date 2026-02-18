"""ALAKAZAM Xf (Cross-Phase) Solver.

Solves for per-antenna cross-hand phase offset phi_pq.
Diagonal Jones: J = diag(1, exp(i phi_pq))

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import least_squares

from . import JonesSolver
from ..jones.constructors import crossphase_to_jones
from ..jones.algebra import compute_residual_cross

logger = logging.getLogger("alakazam")


class XfCrossPhaseSolver(JonesSolver):
    jones_type = "Xf"

    def solve(
        self,
        vis_obs: np.ndarray,
        vis_model: np.ndarray,
        ant1: np.ndarray,
        ant2: np.ndarray,
        freqs: np.ndarray,
        n_ant: int,
        init_jones: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:

        if vis_obs.ndim == 4:
            obs   = vis_obs.mean(axis=1)
            model = vis_model.mean(axis=1)
        else:
            obs, model = vis_obs, vis_model

        x0 = np.zeros(n_ant, dtype=np.float64)

        def _residual(x):
            phi = x.copy()
            phi[self.ref_ant] = 0.0
            J = crossphase_to_jones(phi)
            return compute_residual_cross(J, obs, model, ant1, ant2)

        result = least_squares(
            _residual, x0,
            method="lm",
            max_nfev=self.max_iter * len(x0),
            ftol=self.tol, xtol=self.tol, gtol=self.tol,
        )

        phi = result.x.copy()
        phi[self.ref_ant] = 0.0
        J = crossphase_to_jones(phi)

        return {
            "jones":         J,
            "native_params": {"type": "Xf", "phi_pq": phi},
            "converged":     result.success,
            "n_iter":        result.nfev,
            "cost":          float(result.cost),
        }
