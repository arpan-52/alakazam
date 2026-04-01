"""ALAKAZAM v1 CP (Cross Phase) Solver.

Receives: (n_bl, 2, 2) — time+freq averaged.
Returns:  (n_ant, 2, 2).

Single global parameter phi_cross.
J = diag(1, e^{i phi}) same for all antennas.
Backends: ceres (default), scipy, jax.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations
import logging, os, time as _time
from typing import Any, Dict, Optional
import numpy as np
import pyceres
from . import (JonesSolver, initial_guess_cross_phase)
from ..jones.constructors import cross_phase_to_jones
from ..jones.algebra import compute_residual_cross

logger = logging.getLogger("alakazam")


# ======================================================================
# Ceres cost function — all baselines, one cross-hand pol
# ======================================================================

class _CrossPhaseCost(pyceres.CostFunction):
    """Cross-hand visibility residual for all baselines, one pol.

    pq: pred = model_pq * conj(exp(i*phi))
    qp: pred = exp(i*phi) * model_qp
    Single param block [phi] in radians.
    Vectorized over baselines.
    """

    def __init__(self, obs_cross, model_cross, is_pq=True):
        super().__init__()
        self.obs = obs_cross      # (n_bl,) complex
        self.model = model_cross  # (n_bl,) complex
        self.is_pq = is_pq
        self.set_num_residuals(len(obs_cross) * 2)
        self.set_parameter_block_sizes([1])

    def Evaluate(self, parameters, residuals, jacobians):
        phi = parameters[0][0]
        val = np.exp(1j * phi)
        if self.is_pq:
            pred = self.model * np.conj(val)
        else:
            pred = val * self.model
        diff = self.obs - pred
        residuals[0::2] = diff.real
        residuals[1::2] = diff.imag

        if jacobians is not None and jacobians[0] is not None:
            if self.is_pq:
                dp = self.model * (-1j) * np.conj(val)
            else:
                dp = 1j * val * self.model
            jacobians[0][0::2] = -dp.real
            jacobians[0][1::2] = -dp.imag
        return True


# ======================================================================
# Solver
# ======================================================================

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

        if self.backend == "ceres":
            result = self._solve_ceres(x0, obs, model, ant1, ant2, n_ant)
        elif self.backend == "scipy":
            result = self._solve_scipy_lm(x0, obs, model, ant1, ant2, n_ant)
        elif self.backend == "jax":
            result = self._solve_jax(x0, obs, model, ant1, ant2, n_ant)
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

        x_opt, cost, n_iter, conv = result

        phi = float(x_opt[0])
        J = cross_phase_to_jones(phi, n_ant)
        wall = _time.time() - t0
        return {"jones": J, "converged": conv, "n_iter": n_iter, "cost": cost,
                "wall_time": wall, "solver_backend": self.backend}

    # ------------------------------------------------------------------
    # Ceres LM
    # ------------------------------------------------------------------

    def _solve_ceres(self, x0, obs, model, a1, a2, na):
        from ._cpp_solvers import solve_cross_phase
        jones, cost, n_iter, conv = solve_cross_phase(
            obs.astype(np.complex128, copy=False),
            model.astype(np.complex128, copy=False),
            a1.astype(np.int32, copy=False),
            a2.astype(np.int32, copy=False),
            na, self.max_iter, self.tol,
            float(x0[0]))

        param = np.array([np.angle(jones[0, 1, 1])])
        return (param, float(cost), int(n_iter), bool(conv))

    # ------------------------------------------------------------------
    # scipy LM
    # ------------------------------------------------------------------

    def _solve_scipy_lm(self, x0, obs, model, a1, a2, na):
        from scipy.optimize import least_squares

        def _res(x):
            return compute_residual_cross(
                cross_phase_to_jones(float(x[0]), na), obs, model, a1, a2)

        r = least_squares(_res, x0, method="lm", max_nfev=self.max_iter*100,
                          ftol=self.tol, xtol=self.tol, gtol=self.tol)
        return r.x, float(r.cost), r.nfev, r.success

    # ------------------------------------------------------------------
    # jax BFGS
    # ------------------------------------------------------------------

    def _solve_jax(self, x0, obs, model, a1, a2, na):
        from . import solve_jax_bfgs
        import jax.numpy as jnp

        def cost(x):
            val = jnp.exp(1j * x[0])
            dp = obs[:, 0, 1] - model[:, 0, 1] * jnp.conj(val)
            dq = obs[:, 1, 0] - val * model[:, 1, 0]
            return 0.5 * (jnp.sum(dp.real**2 + dp.imag**2)
                          + jnp.sum(dq.real**2 + dq.imag**2))

        return solve_jax_bfgs(cost, x0, self.max_iter, self.tol)
