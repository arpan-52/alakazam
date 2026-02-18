"""ALAKAZAM Kcross (Cross-Delay) Solver.

Solves for per-antenna cross-hand delay tau_pq in nanoseconds.
Freq-dependent diagonal Jones: J[a,f] = diag(1, exp(-2πi tau_a ν_f))

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import least_squares

from . import JonesSolver
from ..jones.constructors import crossdelay_to_jones
from ..jones.algebra import compute_residual_cross_freq

logger = logging.getLogger("alakazam")


class KcrossDelaySolver(JonesSolver):
    jones_type = "Kcross"

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

        x0 = np.zeros(n_ant, dtype=np.float64)

        def _residual(x):
            tau = x.copy()
            tau[self.ref_ant] = 0.0
            J = crossdelay_to_jones(tau, freqs)
            return compute_residual_cross_freq(J, vis_obs, vis_model, ant1, ant2)

        result = least_squares(
            _residual, x0,
            method="lm",
            max_nfev=self.max_iter * len(x0),
            ftol=self.tol, xtol=self.tol, gtol=self.tol,
        )

        tau = result.x.copy()
        tau[self.ref_ant] = 0.0
        J = crossdelay_to_jones(tau, freqs)

        return {
            "jones":         J,
            "native_params": {"type": "Kcross", "delay_pq": tau},
            "converged":     result.success,
            "n_iter":        result.nfev,
            "cost":          float(result.cost),
        }
