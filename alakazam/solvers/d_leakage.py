"""ALAKAZAM D (Leakage) Solver.

Solves for per-antenna polarisation leakage terms d_pq and d_qp.
Full 2x2 Jones: J = [[1, d_pq], [d_qp, 1]]

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import least_squares

from . import JonesSolver
from ..jones.constructors import leakage_to_jones
from ..jones.algebra import compute_residual_2x2

logger = logging.getLogger("alakazam")


class DLeakageSolver(JonesSolver):
    jones_type = "D"

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

        # d_pq and d_qp as real+imag pairs: x = [Re(d_pq)..., Im(d_pq)..., Re(d_qp)..., Im(d_qp)...]
        x0 = np.zeros(n_ant * 4, dtype=np.float64)

        def _residual(x):
            d_pq = x[:n_ant] + 1j * x[n_ant:2*n_ant]
            d_qp = x[2*n_ant:3*n_ant] + 1j * x[3*n_ant:]
            d_pq[self.ref_ant] = 0.0
            d_qp[self.ref_ant] = 0.0
            J = leakage_to_jones(d_pq, d_qp)
            return compute_residual_2x2(J, obs, model, ant1, ant2)

        result = least_squares(
            _residual, x0,
            method="lm",
            max_nfev=self.max_iter * len(x0),
            ftol=self.tol, xtol=self.tol, gtol=self.tol,
        )

        d_pq = result.x[:n_ant] + 1j * result.x[n_ant:2*n_ant]
        d_qp = result.x[2*n_ant:3*n_ant] + 1j * result.x[3*n_ant:]
        d_pq[self.ref_ant] = 0.0
        d_qp[self.ref_ant] = 0.0
        J = leakage_to_jones(d_pq, d_qp)

        return {
            "jones":         J,
            "native_params": {"type": "D", "d_pq": d_pq, "d_qp": d_qp},
            "converged":     result.success,
            "n_iter":        result.nfev,
            "cost":          float(result.cost),
        }
