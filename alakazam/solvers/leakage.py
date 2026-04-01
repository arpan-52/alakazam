"""ALAKAZAM v1 D (Leakage) Solver.

Receives: (n_bl, 2, 2) — time+freq averaged.
Returns:  (n_ant, 2, 2).

Initial guess: cross/parallel ratio from corrected data.
Backends: ceres (default), scipy, jax.
Ref ant: d_pq[ref] = 0, d_qp[ref] FREE.

Feed-basis aware: uses appropriate initial guess for LINEAR vs CIRCULAR.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations
import logging, os, time as _time
from typing import Any, Dict, Optional
import numpy as np
import pyceres
from . import (JonesSolver, initial_guess_leakage)
from ..jones.constructors import leakage_to_jones
from ..jones.algebra import compute_residual_2x2

logger = logging.getLogger("alakazam")


# ======================================================================
# Ceres cost function — per baseline, full 2x2
# ======================================================================

class _LeakageCost(pyceres.CostFunction):
    """Full 2x2 visibility residual for one baseline.

    pred = J_i M J_j^H  where J = [[1, d_pq], [d_qp, 1]].
    8 residuals (re/im for each of 4 matrix elements).
    4 param blocks of size 2:
      [Re(d_pq_i), Im(d_pq_i)], [Re(d_qp_i), Im(d_qp_i)],
      [Re(d_pq_j), Im(d_pq_j)], [Re(d_qp_j), Im(d_qp_j)].
    """

    def __init__(self, obs_22, model_22):
        super().__init__()
        self.obs = obs_22
        self.model = model_22
        self.set_num_residuals(8)
        self.set_parameter_block_sizes([2, 2, 2, 2])

    def Evaluate(self, parameters, residuals, jacobians):
        dp_i = parameters[0][0] + 1j * parameters[0][1]
        dq_i = parameters[1][0] + 1j * parameters[1][1]
        dp_j = parameters[2][0] + 1j * parameters[2][1]
        dq_j = parameters[3][0] + 1j * parameters[3][1]

        Ji = np.array([[1.0, dp_i], [dq_i, 1.0]], dtype=np.complex128)
        Jj = np.array([[1.0, dp_j], [dq_j, 1.0]], dtype=np.complex128)
        JjH = Jj.conj().T
        pred = Ji @ self.model @ JjH
        diff = self.obs - pred

        residuals[0] = diff[0, 0].real; residuals[1] = diff[0, 0].imag
        residuals[2] = diff[0, 1].real; residuals[3] = diff[0, 1].imag
        residuals[4] = diff[1, 0].real; residuals[5] = diff[1, 0].imag
        residuals[6] = diff[1, 1].real; residuals[7] = diff[1, 1].imag

        if jacobians is not None:
            MJjH = self.model @ JjH
            JiM = Ji @ self.model

            if jacobians[0] is not None:  # d/d [Re(dp_i), Im(dp_i)]
                # dJi/d(Re dp_i) = [[0,1],[0,0]] → dpred row0 += MJjH[1,:]
                dps = [np.array([[MJjH[1, 0], MJjH[1, 1]], [0.0, 0.0]], dtype=np.complex128),
                       np.array([[1j*MJjH[1, 0], 1j*MJjH[1, 1]], [0.0, 0.0]], dtype=np.complex128)]
                _fill_jac_2(jacobians[0], dps)

            if jacobians[1] is not None:  # d/d [Re(dq_i), Im(dq_i)]
                dps = [np.array([[0.0, 0.0], [MJjH[0, 0], MJjH[0, 1]]], dtype=np.complex128),
                       np.array([[0.0, 0.0], [1j*MJjH[0, 0], 1j*MJjH[0, 1]]], dtype=np.complex128)]
                _fill_jac_2(jacobians[1], dps)

            if jacobians[2] is not None:  # d/d [Re(dp_j), Im(dp_j)]
                # dJjH/d(Re dp_j) = [[0,0],[1,0]], dJjH/d(Im dp_j) = [[0,0],[-1j,0]]
                dps = [np.array([[JiM[0, 1], 0.0], [JiM[1, 1], 0.0]], dtype=np.complex128),
                       np.array([[-1j*JiM[0, 1], 0.0], [-1j*JiM[1, 1], 0.0]], dtype=np.complex128)]
                _fill_jac_2(jacobians[2], dps)

            if jacobians[3] is not None:  # d/d [Re(dq_j), Im(dq_j)]
                # dJjH/d(Re dq_j) = [[0,1],[0,0]], dJjH/d(Im dq_j) = [[0,-1j],[0,0]]
                dps = [np.array([[0.0, JiM[0, 0]], [0.0, JiM[1, 0]]], dtype=np.complex128),
                       np.array([[0.0, -1j*JiM[0, 0]], [0.0, -1j*JiM[1, 0]]], dtype=np.complex128)]
                _fill_jac_2(jacobians[3], dps)
        return True


def _fill_jac_2(jac, dps):
    """Fill a Ceres Jacobian array for 8 residuals × 2 params (row-major)."""
    for idx, dp in enumerate(dps):
        jac[0*2 + idx] = -dp[0, 0].real; jac[1*2 + idx] = -dp[0, 0].imag
        jac[2*2 + idx] = -dp[0, 1].real; jac[3*2 + idx] = -dp[0, 1].imag
        jac[4*2 + idx] = -dp[1, 0].real; jac[5*2 + idx] = -dp[1, 0].imag
        jac[6*2 + idx] = -dp[1, 1].real; jac[7*2 + idx] = -dp[1, 1].imag


# ======================================================================
# Solver
# ======================================================================

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

        d_pq_init, d_qp_init = initial_guess_leakage(
            obs, model, ant1, ant2, n_ant, self.ref_ant,
            feed_basis=self.feed_basis)

        if self.backend == "ceres":
            result = self._solve_ceres(
                d_pq_init, d_qp_init, obs, model, ant1, ant2, n_ant)
        elif self.backend == "scipy":
            result = self._solve_scipy_lm(
                d_pq_init, d_qp_init, obs, model, ant1, ant2, n_ant)
        elif self.backend == "jax":
            result = self._solve_jax(
                d_pq_init, d_qp_init, obs, model, ant1, ant2, n_ant)
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

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

    def _pack(self, d_pq, d_qp):
        return np.concatenate([d_pq.real, d_pq.imag, d_qp.real, d_qp.imag])

    # ------------------------------------------------------------------
    # Ceres LM
    # ------------------------------------------------------------------

    def _solve_ceres(self, dpq_init, dqp_init, obs, model, a1, a2, na):
        from ._cpp_solvers import solve_leakage
        jones, cost, n_iter, conv = solve_leakage(
            obs.astype(np.complex128, copy=False),
            model.astype(np.complex128, copy=False),
            a1.astype(np.int32, copy=False),
            a2.astype(np.int32, copy=False),
            na, self.ref_ant, self.max_iter, self.tol,
            dpq_init.astype(np.complex128, copy=False),
            dqp_init.astype(np.complex128, copy=False))

        dpq_out = jones[:, 0, 1]
        dqp_out = jones[:, 1, 0]
        dpq_out[self.ref_ant] = 0.0

        return (self._pack(dpq_out, dqp_out), float(cost), int(n_iter), bool(conv))

    # ------------------------------------------------------------------
    # scipy LM
    # ------------------------------------------------------------------

    def _solve_scipy_lm(self, dpq_init, dqp_init, obs, model, a1, a2, na):
        from scipy.optimize import least_squares
        ref = self.ref_ant

        x0 = self._pack(dpq_init, dqp_init)

        def _res(x):
            dp = x[:na] + 1j * x[na:2*na]
            dq = x[2*na:3*na] + 1j * x[3*na:]
            dp[ref] = 0.0
            return compute_residual_2x2(leakage_to_jones(dp, dq), obs, model, a1, a2)

        r = least_squares(_res, x0, method="lm", max_nfev=self.max_iter*len(x0),
                          ftol=self.tol, xtol=self.tol, gtol=self.tol)
        return r.x, float(r.cost), r.nfev, r.success

    # ------------------------------------------------------------------
    # jax BFGS
    # ------------------------------------------------------------------

    def _solve_jax(self, dpq_init, dqp_init, obs, model, a1, a2, na):
        from . import solve_jax_bfgs
        ref = self.ref_ant
        import jax.numpy as jnp

        x0 = self._pack(dpq_init, dqp_init)

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
