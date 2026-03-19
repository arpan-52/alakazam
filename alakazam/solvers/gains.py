"""ALAKAZAM v1 G (Gains) Solver.

Receives: (n_bl, 2, 2) — time+freq averaged by flow.
Returns:  (n_ant, 2, 2).

Initial guess: BFS gain-ratio extraction from parallel hands.
Ref ant: phase[ref, :] = 0, amp FREE.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations
import logging, time as _time
from typing import Any, Dict, Optional
import numpy as np
from scipy.optimize import least_squares
from . import (JonesSolver, initial_guess_gain_bfs, solve_jax_bfgs)
from ..jones.constructors import gains_to_jones
from ..jones.algebra import compute_residual_diag

logger = logging.getLogger("alakazam")


class GainsSolver(JonesSolver):
    jones_type = "G"

    def solve(self, vis_obs, vis_model, ant1, ant2, freqs, n_ant,
              init_jones=None):
        t0 = _time.time()

        # If multi-channel somehow arrives, average
        if vis_obs.ndim == 4:
            obs = vis_obs.mean(axis=1); model = vis_model.mean(axis=1)
        else:
            obs, model = vis_obs, vis_model

        # Initial guess from data
        amp_init, phase_init = initial_guess_gain_bfs(
            obs, model, ant1, ant2, n_ant, self.ref_ant)
        if self.phase_only:
            amp_init = np.ones_like(amp_init)

        result = None
        if self.backend == "jax":
            result = self._solve_jax(
                amp_init, phase_init, obs, model, ant1, ant2, n_ant)
        if result is None:
            result = self._solve_scipy_lm(
                amp_init, phase_init, obs, model, ant1, ant2, n_ant)
        x_opt, cost, n_iter, conv = result

        amp, phase = self._unpack(x_opt, n_ant)
        J = gains_to_jones(amp, phase)
        wall = _time.time() - t0
        return {"jones": J, "converged": conv, "n_iter": n_iter, "cost": cost,
                "wall_time": wall, "solver_backend": self.backend}

    def _unpack(self, x, n_ant):
        n2 = n_ant * 2
        amp = x[:n2].reshape(n_ant, 2)
        phase = x[n2:].reshape(n_ant, 2)
        if self.phase_only:
            amp = np.ones((n_ant, 2), dtype=np.float64)
        phase[self.ref_ant, :] = 0.0
        return amp, phase

    def _pack(self, amp, phase):
        return np.concatenate([amp.flatten(), phase.flatten()])

    def _solve_scipy_lm(self, amp_init, phase_init, obs, model, a1, a2, na):
        ref, po = self.ref_ant, self.phase_only
        def _res(x):
            amp_ = x[:na*2].reshape(na, 2)
            ph = x[na*2:].reshape(na, 2)
            if po: amp_ = np.ones_like(amp_)
            ph[ref, :] = 0.0
            return compute_residual_diag(gains_to_jones(amp_, ph), obs, model, a1, a2)
        x0 = self._pack(amp_init, phase_init)
        r = least_squares(_res, x0, method="lm", max_nfev=self.max_iter*len(x0),
                          ftol=self.tol, xtol=self.tol, gtol=self.tol)
        return r.x, float(r.cost), r.nfev, r.success

    def _solve_jax(self, amp_init, phase_init, obs, model, a1, a2, na):
        ref, po = self.ref_ant, self.phase_only
        try:
            import jax.numpy as jnp
            def cost(x):
                amp_ = x[:na*2].reshape(na, 2)
                ph = x[na*2:].reshape(na, 2)
                if po: amp_ = jnp.ones_like(amp_)
                ph = ph.at[ref, :].set(0.0)
                gp = amp_[:, 0] * jnp.exp(1j * ph[:, 0])
                gq = amp_[:, 1] * jnp.exp(1j * ph[:, 1])
                dp = obs[:,0,0] - gp[a1]*model[:,0,0]*jnp.conj(gp[a2])
                dq = obs[:,1,1] - gq[a1]*model[:,1,1]*jnp.conj(gq[a2])
                return 0.5*(jnp.sum(dp.real**2+dp.imag**2)+jnp.sum(dq.real**2+dq.imag**2))
            return solve_jax_bfgs(cost, self._pack(amp_init, phase_init),
                                    self.max_iter, self.tol)
        except ImportError:
            return None
