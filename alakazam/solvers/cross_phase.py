"""ALAKAZAM v1 CP (Cross Phase) Solver.

Receives: (n_bl, 2, 2) — time+freq averaged.
Returns:  (n_ant, 2, 2).

Single global parameter phi_cross.
J = diag(1, e^{i phi}) same for all antennas.
Initial guess: mean cross-hand phase from corrected data.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations
import logging, time as _time
from typing import Any, Dict, Optional
import numpy as np
from scipy.optimize import least_squares
from . import (JonesSolver, initial_guess_cross_phase, solve_lbfgsb_jax)
from ..jones.constructors import cross_phase_to_jones
from ..jones.constructors_ad import cost_cross_np
from ..jones.algebra import compute_residual_cross

logger = logging.getLogger("alakazam")


class CrossPhaseSolver(JonesSolver):
    jones_type = "CP"

    def solve(self, vis_obs, vis_model, ant1, ant2, freqs, n_ant,
              init_jones=None):
        t0 = _time.time()

        if vis_obs.ndim == 4:
            obs = vis_obs.mean(axis=1); model = vis_model.mean(axis=1)
        else:
            obs, model = vis_obs, vis_model

        phi_init = initial_guess_cross_phase(obs, model, ant1, ant2)
        x0 = np.array([phi_init], dtype=np.float64)

        result = None
        if self.backend == "jax_scipy":
            result = self._solve_jax(x0, obs, model, ant1, ant2, n_ant)
        if result is None:
            result = self._solve_scipy_lm(x0, obs, model, ant1, ant2, n_ant)
        x_opt, cost, n_iter, conv = result

        phi = float(x_opt[0])
        J = cross_phase_to_jones(phi, n_ant)
        wall = _time.time() - t0
        return {"jones": J, "native_params": {"type": "CP", "phi_cross": phi},
                "converged": conv, "n_iter": n_iter, "cost": cost,
                "wall_time": wall, "solver_backend": self.backend}

    def _solve_scipy_lm(self, x0, obs, model, a1, a2, na):
        def _res(x):
            return compute_residual_cross(
                cross_phase_to_jones(float(x[0]), na), obs, model, a1, a2)
        r = least_squares(_res, x0, method="lm", max_nfev=self.max_iter*100,
                          ftol=self.tol, xtol=self.tol, gtol=self.tol)
        return r.x, float(r.cost), r.nfev, r.success

    def _solve_jax(self, x0, obs, model, a1, a2, na):
        try:
            import jax.numpy as jnp
            def cost(x):
                val = jnp.exp(1j * x[0])
                dp = obs[:, 0, 1] - model[:, 0, 1] * jnp.conj(val)
                dq = obs[:, 1, 0] - val * model[:, 1, 0]
                return 0.5*(jnp.sum(dp.real**2+dp.imag**2)+jnp.sum(dq.real**2+dq.imag**2))
            return solve_lbfgsb_jax(cost, x0, self.max_iter, self.tol)
        except ImportError:
            return None
