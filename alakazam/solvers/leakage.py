"""ALAKAZAM v1 D (Leakage) Solver.

Receives: (n_bl, 2, 2) — time+freq averaged.
Returns:  (n_ant, 2, 2).

Initial guess: cross/parallel ratio from corrected data.
Ref ant: d_pq[ref] = 0, d_qp[ref] FREE.

Feed-basis aware: uses appropriate initial guess for LINEAR vs CIRCULAR.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations
import logging, time as _time
from typing import Any, Dict, Optional
import numpy as np
from scipy.optimize import least_squares
from . import (JonesSolver, initial_guess_leakage, solve_jax_bfgs)
from ..jones.constructors import leakage_to_jones
from ..jones.algebra import compute_residual_2x2

logger = logging.getLogger("alakazam")


class LeakageSolver(JonesSolver):
    jones_type = "D"

    def solve(self, vis_obs, vis_model, ant1, ant2, freqs, n_ant,
              init_jones=None):
        t0 = _time.time()

        if vis_obs.ndim == 4:
            obs = vis_obs.mean(axis=1); model = vis_model.mean(axis=1)
        else:
            obs, model = vis_obs, vis_model

        logger.info(f"D solve: feed_basis={self.feed_basis}")

        # Initial guess from data (feed-basis aware)
        d_pq_init, d_qp_init = initial_guess_leakage(
            obs, model, ant1, ant2, n_ant, self.ref_ant,
            feed_basis=self.feed_basis)

        # Pack: [Re(d_pq), Im(d_pq), Re(d_qp), Im(d_qp)]
        x0 = np.concatenate([d_pq_init.real, d_pq_init.imag,
                              d_qp_init.real, d_qp_init.imag])

        result = None
        if self.backend == "jax":
            result = self._solve_jax(x0, obs, model, ant1, ant2, n_ant)
        if result is None:
            result = self._solve_scipy_lm(x0, obs, model, ant1, ant2, n_ant)
        x_opt, cost, n_iter, conv = result

        d_pq, d_qp = self._unpack(x_opt, n_ant)
        J = leakage_to_jones(d_pq, d_qp)
        wall = _time.time() - t0
        return {"jones": J, "converged": conv, "n_iter": n_iter, "cost": cost,
                "wall_time": wall, "solver_backend": self.backend}

    def _unpack(self, x, n_ant):
        d_pq = x[:n_ant] + 1j * x[n_ant:2*n_ant]
        d_qp = x[2*n_ant:3*n_ant] + 1j * x[3*n_ant:]
        d_pq[self.ref_ant] = 0.0
        return d_pq, d_qp

    def _solve_scipy_lm(self, x0, obs, model, a1, a2, na):
        ref = self.ref_ant
        def _res(x):
            dp = x[:na] + 1j*x[na:2*na]; dq = x[2*na:3*na] + 1j*x[3*na:]
            dp[ref] = 0.0
            return compute_residual_2x2(leakage_to_jones(dp, dq), obs, model, a1, a2)
        r = least_squares(_res, x0, method="lm", max_nfev=self.max_iter*len(x0),
                          ftol=self.tol, xtol=self.tol, gtol=self.tol)
        return r.x, float(r.cost), r.nfev, r.success

    def _solve_jax(self, x0, obs, model, a1, a2, na):
        ref = self.ref_ant
        try:
            import jax.numpy as jnp
            def cost(x):
                dpr = x[:na]; dpi = x[na:2*na]
                dqr = x[2*na:3*na]; dqi = x[3*na:]
                dpr = dpr.at[ref].set(0.0); dpi = dpi.at[ref].set(0.0)
                dp = dpr + 1j*dpi; dq = dqr + 1j*dqi
                J = jnp.zeros((na, 2, 2), dtype=jnp.complex128)
                J = J.at[:, 0, 0].set(1.0); J = J.at[:, 0, 1].set(dp)
                J = J.at[:, 1, 0].set(dq); J = J.at[:, 1, 1].set(1.0)
                Ji = J[a1]; JjH = jnp.conj(J[a2]).transpose(0, 2, 1)
                pred = jnp.einsum("bij,bjk,bkl->bil", Ji, model, JjH)
                d = obs - pred
                return 0.5 * jnp.sum(d.real**2 + d.imag**2)
            return solve_jax_bfgs(cost, x0, self.max_iter, self.tol)
        except ImportError:
            return None
