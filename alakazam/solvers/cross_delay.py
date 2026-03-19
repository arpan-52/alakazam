"""ALAKAZAM v1 KC (Cross Delay) Solver.

Receives: (n_bl, n_chan, 2, 2) — time-averaged, freq retained.
Returns:  (n_ant, 2, 2).

Single global parameter tau_cross.
J = diag(e^{-2pi i tau nu}, 1) same for all antennas.
Initial guess: cross-hand phase slope averaged over baselines.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations
import logging, time as _time
from typing import Any, Dict, Optional
import numpy as np
from scipy.optimize import least_squares
from . import JonesSolver, solve_jax_bfgs
from ..jones.constructors import cross_delay_to_jones
from ..jones.algebra import compute_residual_cross_freq

logger = logging.getLogger("alakazam")


class CrossDelaySolver(JonesSolver):
    jones_type = "KC"

    def solve(self, vis_obs, vis_model, ant1, ant2, freqs, n_ant,
              init_jones=None):
        t0 = _time.time()

        x0 = np.array([self._initial_estimate(vis_obs, vis_model, freqs)],
                       dtype=np.float64)

        result = None
        if self.backend == "jax":
            result = self._solve_jax(
                x0, vis_obs, vis_model, ant1, ant2, freqs, n_ant)
        if result is None:
            result = self._solve_scipy_lm(
                x0, vis_obs, vis_model, ant1, ant2, freqs, n_ant)
        x_opt, cost, n_iter, conv = result

        tau = float(x_opt[0])
        J_full = cross_delay_to_jones(tau, freqs, n_ant)
        J_mean = J_full.mean(axis=1)

        wall = _time.time() - t0
        return {"jones": J_mean, "converged": conv, "n_iter": n_iter, "cost": cost,
                "wall_time": wall, "solver_backend": self.backend}

    def _solve_scipy_lm(self, x0, vis_obs, vis_model, a1, a2, freqs, na):
        def _res(x):
            return compute_residual_cross_freq(
                cross_delay_to_jones(float(x[0]), freqs, na),
                vis_obs, vis_model, a1, a2)
        r = least_squares(_res, x0, method="lm", max_nfev=self.max_iter*100,
                          ftol=self.tol, xtol=self.tol, gtol=self.tol)
        return r.x, float(r.cost), r.nfev, r.success

    def _solve_jax(self, x0, vis_obs, vis_model, a1, a2, freqs, na):
        try:
            import jax.numpy as jnp
            def cost(x):
                tau = x[0] * 1e-9
                ph = jnp.exp(-2j * jnp.pi * tau * jnp.array(freqs))
                Jp = ph[jnp.newaxis, :]
                dp = vis_obs[:,:,0,1] - Jp[0,:][jnp.newaxis,:]*vis_model[:,:,0,1]
                dq = vis_obs[:,:,1,0] - vis_model[:,:,1,0]*jnp.conj(Jp[0,:][jnp.newaxis,:])
                return 0.5*(jnp.sum(dp.real**2+dp.imag**2)+jnp.sum(dq.real**2+dq.imag**2))
            return solve_jax_bfgs(cost, x0, self.max_iter, self.tol)
        except ImportError:
            return None

    def _initial_estimate(self, vis_obs, vis_model, freqs):
        if len(freqs) < 2:
            return 0.0
        ratio = vis_obs[:, :, 0, 1] / (vis_model[:, :, 0, 1] + 1e-30)
        avg = np.mean(ratio, axis=0)
        phase = np.unwrap(np.angle(avg))
        return -np.polyfit(freqs, phase, 1)[0] / (2 * np.pi) * 1e9
