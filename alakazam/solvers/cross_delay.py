"""ALAKAZAM v1 KC (Cross Delay) Solver.

Receives: (n_bl, n_chan, 2, 2) — time-averaged, freq retained.
Returns:  (n_ant, 2, 2).

Single global parameter tau_cross.
J = diag(e^{-2pi i tau nu}, 1) same for all antennas.
Backends: ceres (default), scipy, jax.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations
import logging, os, time as _time
from typing import Any, Dict, Optional
import numpy as np
import pyceres
from . import JonesSolver
from ..jones.constructors import cross_delay_to_jones
from ..jones.algebra import compute_residual_cross_freq

logger = logging.getLogger("alakazam")


# ======================================================================
# Ceres cost function — all baselines, one cross-hand pol
# ======================================================================

class _CrossDelayCost(pyceres.CostFunction):
    """Cross-hand visibility residual for all baselines, one pol (pq or qp).

    pq: pred = exp(-2πi τ ν) * model_pq
    qp: pred = exp(+2πi τ ν) * model_qp
    Single param block [tau] in nanoseconds.
    Vectorized over baselines and frequencies.
    """

    def __init__(self, obs_cross, model_cross, freqs, is_pq=True):
        super().__init__()
        self.obs = obs_cross        # (n_bl, n_freq) complex
        self.model = model_cross    # (n_bl, n_freq) complex
        self.is_pq = is_pq
        self.twopi_nu = 2.0 * np.pi * freqs * 1e-9
        self.set_num_residuals(obs_cross.shape[0] * obs_cross.shape[1] * 2)
        self.set_parameter_block_sizes([1])

    def Evaluate(self, parameters, residuals, jacobians):
        tau = parameters[0][0]
        phase = -self.twopi_nu * tau
        if self.is_pq:
            J00 = np.exp(1j * phase)
            pred = J00[None, :] * self.model
        else:
            pred = np.exp(-1j * phase)[None, :] * self.model

        diff = self.obs - pred
        residuals[0::2] = diff.ravel().real
        residuals[1::2] = diff.ravel().imag

        if jacobians is not None and jacobians[0] is not None:
            if self.is_pq:
                dp = pred * (-1j * self.twopi_nu[None, :])
            else:
                dp = pred * (1j * self.twopi_nu[None, :])
            flat = dp.ravel()
            jacobians[0][0::2] = -flat.real
            jacobians[0][1::2] = -flat.imag
        return True


# ======================================================================
# Solver
# ======================================================================

class CrossDelaySolver(JonesSolver):
    jones_type = "KC"

    def solve(self, vis_obs, vis_model, ant1, ant2, freqs, n_ant,
              init_jones=None):
        t0 = _time.time()

        x0 = np.array([self._initial_estimate(vis_obs, vis_model, freqs)],
                       dtype=np.float64)

        if self.backend == "ceres":
            result = self._solve_ceres(x0, vis_obs, vis_model, freqs)
        elif self.backend == "scipy":
            result = self._solve_scipy_lm(
                x0, vis_obs, vis_model, ant1, ant2, freqs, n_ant)
        elif self.backend == "jax":
            result = self._solve_jax(
                x0, vis_obs, vis_model, ant1, ant2, freqs, n_ant)
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

        x_opt, cost, n_iter, conv = result

        tau = float(x_opt[0])
        freq_mid = np.array([np.mean(freqs)])
        J = cross_delay_to_jones(tau, freq_mid, n_ant)[:, 0]  # (n_ant, 2, 2)

        # Store delay as (n_ant, 2): [tau_cross, 0] per antenna
        delay = np.zeros((n_ant, 2), dtype=np.float64)
        delay[:, 0] = tau

        wall = _time.time() - t0
        return {"jones": J, "delay": delay, "converged": conv, "n_iter": n_iter,
                "cost": cost, "wall_time": wall, "solver_backend": self.backend}

    # ------------------------------------------------------------------
    # Ceres LM
    # ------------------------------------------------------------------

    def _solve_ceres(self, x0, vis_obs, vis_model, freqs):
        param = np.array([x0[0]])
        prob = pyceres.Problem()
        prob.add_residual_block(
            _CrossDelayCost(vis_obs[:, :, 0, 1], vis_model[:, :, 0, 1],
                            freqs, is_pq=True),
            None, [param])
        prob.add_residual_block(
            _CrossDelayCost(vis_obs[:, :, 1, 0], vis_model[:, :, 1, 0],
                            freqs, is_pq=False),
            None, [param])

        from .parallel_delay import _ceres_opts
        opts = _ceres_opts(self.max_iter, self.tol)
        summary = pyceres.SolverSummary()
        pyceres.solve(opts, prob, summary)

        return (param, float(summary.final_cost),
                summary.num_successful_steps,
                summary.termination_type == pyceres.TerminationType.CONVERGENCE)

    # ------------------------------------------------------------------
    # scipy LM
    # ------------------------------------------------------------------

    def _solve_scipy_lm(self, x0, vis_obs, vis_model, a1, a2, freqs, na):
        from scipy.optimize import least_squares

        def _res(x):
            return compute_residual_cross_freq(
                cross_delay_to_jones(float(x[0]), freqs, na),
                vis_obs, vis_model, a1, a2)

        r = least_squares(_res, x0, method="lm", max_nfev=self.max_iter*100,
                          ftol=self.tol, xtol=self.tol, gtol=self.tol)
        return r.x, float(r.cost), r.nfev, r.success

    # ------------------------------------------------------------------
    # jax BFGS
    # ------------------------------------------------------------------

    def _solve_jax(self, x0, vis_obs, vis_model, a1, a2, freqs, na):
        from . import solve_jax_bfgs
        import jax.numpy as jnp

        def cost(x):
            tau = x[0] * 1e-9
            ph = jnp.exp(-2j * jnp.pi * tau * jnp.array(freqs))
            Jp = ph[jnp.newaxis, :]
            dp = vis_obs[:, :, 0, 1] - Jp[0, :][jnp.newaxis, :] * vis_model[:, :, 0, 1]
            dq = vis_obs[:, :, 1, 0] - vis_model[:, :, 1, 0] * jnp.conj(Jp[0, :][jnp.newaxis, :])
            return 0.5 * (jnp.sum(dp.real**2 + dp.imag**2)
                          + jnp.sum(dq.real**2 + dq.imag**2))

        return solve_jax_bfgs(cost, x0, self.max_iter, self.tol)

    # ------------------------------------------------------------------
    # Initial estimate — cross-hand phase slope
    # ------------------------------------------------------------------

    def _initial_estimate(self, vis_obs, vis_model, freqs):
        if len(freqs) < 2:
            return 0.0
        ratio = vis_obs[:, :, 0, 1] / (vis_model[:, :, 0, 1] + 1e-30)
        avg = np.mean(ratio, axis=0)
        phase = np.unwrap(np.angle(avg))
        return -np.polyfit(freqs, phase, 1)[0] / (2 * np.pi) * 1e9
