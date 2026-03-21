"""ALAKAZAM v1 K (Parallel Delay) Solver.

Receives: (n_bl, n_chan, 2, 2) — time-averaged, freq retained.
Returns:  (n_ant, 2, 2) — one solution per cell.

Initial guess: FFT-based fringe fitting (8x zero-padding) -> BFS propagation.
Backends: ceres (default), scipy, jax.
Ref ant: delay[ref, :] = 0.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations
import logging, os, time as _time
from typing import Any, Dict, Optional
import numpy as np
import pyceres
from . import (JonesSolver, build_antenna_graph, bfs_order)
from ..jones.constructors import parallel_delay_to_jones
from ..jones.algebra import compute_residual_diag_freq

logger = logging.getLogger("alakazam")


# ======================================================================
# Ceres cost function — per baseline per pol
# ======================================================================

class _DelayCost(pyceres.CostFunction):
    """Complex visibility residual for one baseline, one pol.

    obs ≈ exp(-2πi (τ_i - τ_j) ν)  for model=1.
    Residual: obs - pred, split into real/imag → n_freq*2 residuals.
    Param blocks: [τ_i] (size 1), [τ_j] (size 1) in nanoseconds.
    Vectorized Evaluate — no Python loops over frequencies.
    """

    def __init__(self, obs, freqs):
        super().__init__()
        self.obs = obs
        self.freqs = freqs
        self.twopi_nu = 2.0 * np.pi * freqs * 1e-9
        self.set_num_residuals(len(freqs) * 2)
        self.set_parameter_block_sizes([1, 1])

    def Evaluate(self, parameters, residuals, jacobians):
        tau_i = parameters[0][0]
        tau_j = parameters[1][0]
        phase = -self.twopi_nu * (tau_i - tau_j)
        pred = np.exp(1j * phase)
        diff = self.obs - pred
        residuals[0::2] = diff.real
        residuals[1::2] = diff.imag
        if jacobians is not None:
            if jacobians[0] is not None:
                dp = pred * (-1j * self.twopi_nu)
                jacobians[0][0::2] = -dp.real
                jacobians[0][1::2] = -dp.imag
            if jacobians[1] is not None:
                dp = pred * (1j * self.twopi_nu)
                jacobians[1][0::2] = -dp.real
                jacobians[1][1::2] = -dp.imag
        return True


# ======================================================================
# Solver
# ======================================================================

class ParallelDelaySolver(JonesSolver):
    jones_type = "K"

    def solve(self, vis_obs, vis_model, ant1, ant2, freqs, n_ant,
              init_jones=None):
        t0 = _time.time()
        logger.debug(f"K solve: n_ant={n_ant} n_bl={vis_obs.shape[0]} "
                      f"n_freq={vis_obs.shape[1]} backend={self.backend}")

        delay_init = self._initial_estimate(
            vis_obs, vis_model, ant1, ant2, freqs, n_ant)

        if self.backend == "ceres":
            result = self._solve_ceres(
                delay_init, vis_obs, ant1, ant2, freqs, n_ant)
        elif self.backend == "scipy":
            result = self._solve_scipy_lm(
                delay_init, vis_obs, vis_model, ant1, ant2, freqs, n_ant)
        elif self.backend == "jax":
            result = self._solve_jaxopt(
                delay_init, vis_obs, ant1, ant2, freqs, n_ant)
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

        x_opt, cost, n_iter, conv = result

        delay = x_opt.reshape(n_ant, 2)
        delay[self.ref_ant, :] = 0.0
        freq_mid = np.array([np.mean(freqs)])
        J = parallel_delay_to_jones(delay, freq_mid)[:, 0]  # (n_ant, 2, 2)

        wall = _time.time() - t0
        return {
            "jones": J, "delay": delay, "converged": conv, "n_iter": n_iter,
            "cost": cost, "wall_time": wall,
            "solver_backend": self.backend,
        }

    # ------------------------------------------------------------------
    # Ceres LM — analytic Jacobians, sparse Cholesky
    # ------------------------------------------------------------------

    def _solve_ceres(self, delay_init, vis_obs, ant1, ant2, freqs, n_ant):
        obs_pp = vis_obs[:, :, 0, 0]
        obs_qq = vis_obs[:, :, 1, 1]

        delays_p = [np.array([delay_init[a, 0]]) for a in range(n_ant)]
        delays_q = [np.array([delay_init[a, 1]]) for a in range(n_ant)]

        prob = pyceres.Problem()
        for k in range(len(ant1)):
            a1, a2 = int(ant1[k]), int(ant2[k])
            prob.add_residual_block(
                _DelayCost(obs_pp[k], freqs), None,
                [delays_p[a1], delays_p[a2]])
            prob.add_residual_block(
                _DelayCost(obs_qq[k], freqs), None,
                [delays_q[a1], delays_q[a2]])

        prob.set_parameter_block_constant(delays_p[self.ref_ant])
        prob.set_parameter_block_constant(delays_q[self.ref_ant])

        opts = _ceres_opts(self.max_iter, self.tol)
        summary = pyceres.SolverSummary()
        pyceres.solve(opts, prob, summary)

        result = np.zeros((n_ant, 2))
        for a in range(n_ant):
            result[a, 0] = delays_p[a][0]
            result[a, 1] = delays_q[a][0]
        result[self.ref_ant, :] = 0.0

        return (result.flatten(), float(summary.final_cost),
                summary.num_successful_steps,
                summary.termination_type == pyceres.TerminationType.CONVERGENCE)

    # ------------------------------------------------------------------
    # scipy LM — finite-difference Jacobians
    # ------------------------------------------------------------------

    def _solve_scipy_lm(self, delay_init, vis_obs, vis_model,
                        ant1, ant2, freqs, n_ant):
        from scipy.optimize import least_squares
        ref = self.ref_ant

        def _residual(x):
            d = x.reshape(n_ant, 2); d[ref, :] = 0.0
            J = parallel_delay_to_jones(d, freqs)
            return compute_residual_diag_freq(J, vis_obs, vis_model, ant1, ant2)

        result = least_squares(_residual, delay_init.flatten(), method="lm",
                               max_nfev=self.max_iter * delay_init.size,
                               ftol=self.tol, xtol=self.tol, gtol=self.tol)
        return result.x, float(result.cost), result.nfev, result.success

    # ------------------------------------------------------------------
    # jaxopt LM — autodiff, f32 on GPU
    # ------------------------------------------------------------------

    def _solve_jaxopt(self, delay_init, vis_obs, ant1, ant2, freqs, n_ant):
        import jax
        import jax.numpy as jnp
        from jaxopt import LevenbergMarquardt
        from . import _jax_device

        device = _jax_device or jax.devices("cpu")[0]
        is_gpu = device.platform == "gpu"
        cdtype = jnp.complex64 if is_gpu else jnp.complex128
        fdtype = jnp.float32 if is_gpu else jnp.float64

        if not is_gpu:
            jax.config.update("jax_enable_x64", True)

        ref = self.ref_ant
        with jax.default_device(device):
            vobs_pp = jnp.array(vis_obs[:, :, 0, 0], dtype=cdtype)
            vobs_qq = jnp.array(vis_obs[:, :, 1, 1], dtype=cdtype)
            gf = jnp.array(freqs, dtype=fdtype)
            a1j = jnp.array(ant1)
            a2j = jnp.array(ant2)

            def res_fn(x):
                d = x.reshape(n_ant, 2)
                d = d.at[ref, :].set(0.0)
                ph = (-2.0 * jnp.pi * 1e-9) * gf[None, :]
                pp = jnp.exp(1j * d[a1j, 0:1] * ph) * jnp.exp(-1j * d[a2j, 0:1] * ph)
                qq = jnp.exp(1j * d[a1j, 1:2] * ph) * jnp.exp(-1j * d[a2j, 1:2] * ph)
                dp = vobs_pp - pp; dq = vobs_qq - qq
                return jnp.concatenate([dp.real.ravel(), dp.imag.ravel(),
                                        dq.real.ravel(), dq.imag.ravel()])

            x0 = jnp.array(delay_init.flatten(), dtype=fdtype)
            _ = jax.jit(res_fn)(x0).block_until_ready()

            tol = 1e-7 if is_gpu else self.tol
            solver = LevenbergMarquardt(res_fn, maxiter=self.max_iter,
                                        tol=tol, jit=True)
            state = solver.init_state(x0)
            x = x0
            stop_val = 1e-12 if is_gpu else 1e-24
            for i in range(self.max_iter):
                x, state = solver.update(x, state)
                if state.value < stop_val:
                    break
            x.block_until_ready()

            result = np.array(x, dtype=np.float64)
            result.reshape(n_ant, 2)[ref, :] = 0.0

        return result, float(state.value), i + 1, state.value < stop_val

    # ------------------------------------------------------------------
    # Initial estimate — FFT fringe fitting + BFS
    # ------------------------------------------------------------------

    def _initial_estimate(self, vis_obs, vis_model, ant1, ant2, freqs, n_ant):
        adj = build_antenna_graph(ant1, ant2, n_ant)
        order = bfs_order(adj, self.ref_ant)
        bl_delay = _bl_delay_fft(vis_obs, vis_model, ant1, ant2, freqs)
        delay = np.zeros((n_ant, 2), dtype=np.float64)
        solved = np.zeros(n_ant, dtype=bool)
        solved[self.ref_ant] = True
        for a in order:
            if a == self.ref_ant:
                continue
            for k, (a1, a2) in enumerate(zip(ant1, ant2)):
                if a1 == a and solved[a2]:
                    delay[a] = bl_delay[k] + delay[a2]; solved[a] = True; break
                if a2 == a and solved[a1]:
                    delay[a] = -bl_delay[k] + delay[a1]; solved[a] = True; break
        delay[self.ref_ant, :] = 0.0
        return delay


# ======================================================================
# Shared Ceres options
# ======================================================================

def _ceres_opts(max_iter, tol):
    opts = pyceres.SolverOptions()
    opts.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    opts.max_num_iterations = max_iter
    opts.function_tolerance = tol
    opts.gradient_tolerance = tol
    opts.parameter_tolerance = tol
    opts.minimizer_progress_to_stdout = False
    opts.num_threads = os.cpu_count()
    return opts


# ======================================================================
# FFT-based per-baseline delay estimate (vectorized, 8x zero-padding)
# ======================================================================

def _bl_delay_fft(vis_obs, vis_model, ant1, ant2, freqs):
    """FFT-based per-baseline delay estimate.

    Cross-multiply vis_obs * conj(vis_model) for each baseline,
    FFT along frequency axis with 8x zero-padding, find peak.
    Returns delays in nanoseconds, shape (n_bl, 2).
    """
    n_bl = vis_obs.shape[0]
    n_freq = len(freqs)
    delays = np.zeros((n_bl, 2), dtype=np.float64)
    if n_freq < 2:
        return delays

    df = freqs[1] - freqs[0]
    nfft = n_freq * 8
    delay_axis = np.fft.fftfreq(nfft, d=df)  # seconds

    for pol in range(2):
        # (n_bl, n_freq) cross-spectra — vectorized over baselines
        xspec = vis_obs[:, :, pol, pol] * np.conj(vis_model[:, :, pol, pol])

        # Zero flagged channels (model near-zero)
        bad = np.abs(vis_model[:, :, pol, pol]) < 1e-30
        xspec[bad] = 0.0
        good_count = np.sum(~bad, axis=1)

        # FFT with 8x zero-padding, vectorized over baselines
        spectra = np.fft.fft(xspec, n=nfft, axis=1)
        peak_idx = np.argmax(np.abs(spectra), axis=1)

        # Negate: FFT of obs*conj(model) peaks at -(τ_i - τ_j)
        delays[:, pol] = -delay_axis[peak_idx] * 1e9
        delays[good_count < 4, pol] = 0.0

    return delays
