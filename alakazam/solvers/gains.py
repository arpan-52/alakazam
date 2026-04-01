"""ALAKAZAM v1 G (Gains) Solver.

Receives: (n_bl, 2, 2) — time+freq averaged by flow.
Returns:  (n_ant, 2, 2).

Initial guess: BFS gain-ratio extraction from parallel hands.
Backends: ceres (default), scipy, jax.
Ref ant: phase[ref, :] = 0, amp FREE.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations
import logging, os, time as _time
from typing import Any, Dict, Optional
import numpy as np
import pyceres
from . import (JonesSolver, initial_guess_gain_bfs)
from ..jones.constructors import gains_to_jones
from ..jones.algebra import compute_residual_diag

logger = logging.getLogger("alakazam")


# ======================================================================
# Ceres cost function — per baseline per pol
# ======================================================================

class _GainCost(pyceres.CostFunction):
    """Complex visibility residual for one baseline, one pol.

    obs ≈ g_i * model * conj(g_j)  where g = amp * exp(i*phase).
    2 residuals (re, im), 4 param blocks of size 1:
    [amp_i], [phase_i], [amp_j], [phase_j].
    """

    def __init__(self, obs_val, model_val):
        super().__init__()
        self.obs = obs_val
        self.model = model_val
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([1, 1, 1, 1])

    def Evaluate(self, parameters, residuals, jacobians):
        amp_i = parameters[0][0]
        phi_i = parameters[1][0]
        amp_j = parameters[2][0]
        phi_j = parameters[3][0]
        gi = amp_i * np.exp(1j * phi_i)
        gj = amp_j * np.exp(1j * phi_j)
        M_cgj = self.model * np.conj(gj)
        pred = gi * M_cgj
        diff = self.obs - pred
        residuals[0] = diff.real
        residuals[1] = diff.imag
        if jacobians is not None:
            if jacobians[0] is not None:  # d/d(amp_i)
                dp = np.exp(1j * phi_i) * M_cgj
                jacobians[0][0] = -dp.real
                jacobians[0][1] = -dp.imag
            if jacobians[1] is not None:  # d/d(phase_i)
                dp = 1j * gi * M_cgj
                jacobians[1][0] = -dp.real
                jacobians[1][1] = -dp.imag
            if jacobians[2] is not None:  # d/d(amp_j)
                dp = gi * self.model * np.exp(-1j * phi_j)
                jacobians[2][0] = -dp.real
                jacobians[2][1] = -dp.imag
            if jacobians[3] is not None:  # d/d(phase_j)
                dp = gi * self.model * (-1j) * np.conj(gj)
                jacobians[3][0] = -dp.real
                jacobians[3][1] = -dp.imag
        return True


# ======================================================================
# Solver
# ======================================================================

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

        amp_init, phase_init = initial_guess_gain_bfs(
            obs, model, ant1, ant2, n_ant, self.ref_ant)
        if self.phase_only:
            amp_init = np.ones_like(amp_init)

        if self.backend == "ceres":
            result = self._solve_ceres(
                amp_init, phase_init, obs, model, ant1, ant2, n_ant)
        elif self.backend == "scipy":
            result = self._solve_scipy_lm(
                amp_init, phase_init, obs, model, ant1, ant2, n_ant)
        elif self.backend == "jax":
            result = self._solve_jax(
                amp_init, phase_init, obs, model, ant1, ant2, n_ant)
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

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

    # ------------------------------------------------------------------
    # Ceres LM — analytic Jacobians, sparse Cholesky
    # ------------------------------------------------------------------

    def _solve_ceres(self, amp_init, phase_init, obs, model, a1, a2, na):
        from ._cpp_solvers import solve_gains
        jones, cost, n_iter, conv = solve_gains(
            obs.astype(np.complex128, copy=False),
            model.astype(np.complex128, copy=False),
            a1.astype(np.int32, copy=False),
            a2.astype(np.int32, copy=False),
            na, self.ref_ant, self.max_iter, self.tol,
            amp_init, phase_init, self.phase_only)

        gp = jones[:, 0, 0]; gq = jones[:, 1, 1]
        amp_out = np.column_stack([np.abs(gp), np.abs(gq)])
        phase_out = np.column_stack([np.angle(gp), np.angle(gq)])
        phase_out[self.ref_ant, :] = 0.0
        if self.phase_only:
            amp_out = np.ones_like(amp_out)

        return (self._pack(amp_out, phase_out), float(cost), int(n_iter), bool(conv))

    # ------------------------------------------------------------------
    # scipy LM
    # ------------------------------------------------------------------

    def _solve_scipy_lm(self, amp_init, phase_init, obs, model, a1, a2, na):
        from scipy.optimize import least_squares
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

    # ------------------------------------------------------------------
    # jax BFGS
    # ------------------------------------------------------------------

    def _solve_jax(self, amp_init, phase_init, obs, model, a1, a2, na):
        from . import solve_jax_bfgs
        ref, po = self.ref_ant, self.phase_only
        import jax.numpy as jnp

        def cost(x):
            amp_ = x[:na*2].reshape(na, 2)
            ph = x[na*2:].reshape(na, 2)
            if po: amp_ = jnp.ones_like(amp_)
            ph = ph.at[ref, :].set(0.0)
            gp = amp_[:, 0] * jnp.exp(1j * ph[:, 0])
            gq = amp_[:, 1] * jnp.exp(1j * ph[:, 1])
            dp = obs[:, 0, 0] - gp[a1] * model[:, 0, 0] * jnp.conj(gp[a2])
            dq = obs[:, 1, 1] - gq[a1] * model[:, 1, 1] * jnp.conj(gq[a2])
            return 0.5 * (jnp.sum(dp.real**2 + dp.imag**2)
                          + jnp.sum(dq.real**2 + dq.imag**2))

        return solve_jax_bfgs(cost, self._pack(amp_init, phase_init),
                              self.max_iter, self.tol)
