"""ALAKAZAM v1 K (Parallel Delay) Solver.

Receives: (n_bl, n_chan, 2, 2) — time-averaged, freq retained.
Returns:  (n_ant, 2, 2) — one solution per cell.

Initial guess: per-baseline phase slope → BFS propagation from ref_ant.
Ref ant: delay[ref, :] = 0.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations
import logging, time as _time
from typing import Any, Dict, Optional
import numpy as np
from scipy.optimize import least_squares
from . import (JonesSolver, build_antenna_graph, bfs_order,
               solve_lbfgsb_jax)
from ..jones.constructors import parallel_delay_to_jones
from ..jones.constructors_ad import (parallel_delay_to_jones_np,
                                     cost_diag_freq_np)
from ..jones.algebra import compute_residual_diag_freq

logger = logging.getLogger("alakazam")


class ParallelDelaySolver(JonesSolver):
    jones_type = "K"

    def solve(self, vis_obs, vis_model, ant1, ant2, freqs, n_ant,
              init_jones=None):
        t0 = _time.time()
        logger.debug(f"K solve: n_ant={n_ant} n_bl={vis_obs.shape[0]} "
                      f"n_freq={vis_obs.shape[1]} backend={self.backend}")

        delay_init = self._initial_estimate(
            vis_obs, vis_model, ant1, ant2, freqs, n_ant)

        # Try JAX first (fast, exact gradients), fall back to scipy LM
        result = None
        if self.backend == "jax_scipy":
            result = self._solve_jax(
                delay_init, vis_obs, vis_model, ant1, ant2, freqs, n_ant)

        if result is None:
            logger.debug("K: using scipy LM")
            result = self._solve_scipy_lm(
                delay_init, vis_obs, vis_model, ant1, ant2, freqs, n_ant)

        x_opt, cost, n_iter, conv = result

        delay = x_opt.reshape(n_ant, 2)
        delay[self.ref_ant, :] = 0.0
        J = parallel_delay_to_jones(delay, freqs)
        # Collapse freq axis: take mean Jones across freq for the (n_ant, 2, 2) output
        # Actually K produces freq-dependent Jones but we need (n_ant, 2, 2) per cell.
        # The native params (delay) carry the full info; Jones is evaluated at freqs.
        # For the universal schema, flow.py stores delay in native_params and
        # reconstructs Jones at any freq during interpolation.
        # Return the mean across freq as the representative Jones.
        J_mean = J.mean(axis=1) if J.ndim == 4 else J

        wall = _time.time() - t0
        return {
            "jones": J_mean,  # (n_ant, 2, 2)
            "native_params": {"type": "K", "delay": delay},
            "converged": conv, "n_iter": n_iter,
            "cost": cost, "wall_time": wall,
            "solver_backend": self.backend,
        }

    def _solve_scipy_lm(self, delay_init, vis_obs, vis_model,
                        ant1, ant2, freqs, n_ant):
        ref = self.ref_ant
        def _residual(x):
            d = x.reshape(n_ant, 2); d[ref, :] = 0.0
            J = parallel_delay_to_jones(d, freqs)
            return compute_residual_diag_freq(J, vis_obs, vis_model, ant1, ant2)
        result = least_squares(_residual, delay_init.flatten(), method="lm",
                               max_nfev=self.max_iter * delay_init.size,
                               ftol=self.tol, xtol=self.tol, gtol=self.tol)
        return result.x, float(result.cost), result.nfev, result.success

    def _solve_jax(self, delay_init, vis_obs, vis_model,
                   ant1, ant2, freqs, n_ant):
        ref = self.ref_ant
        try:
            import jax.numpy as jnp
            def cost_fn_jax(x):
                d = x.reshape(n_ant, 2).at[ref, :].set(0.0)
                tau_p = d[:, 0:1] * 1e-9
                tau_q = d[:, 1:2] * 1e-9
                fr = jnp.array(freqs)[jnp.newaxis, :]
                Jp = jnp.exp(-2j * jnp.pi * tau_p * fr)
                Jq = jnp.exp(-2j * jnp.pi * tau_q * fr)
                dp = vis_obs[:,:,0,0] - Jp[ant1,:]*vis_model[:,:,0,0]*jnp.conj(Jp[ant2,:])
                dq = vis_obs[:,:,1,1] - Jq[ant1,:]*vis_model[:,:,1,1]*jnp.conj(Jq[ant2,:])
                return 0.5*(jnp.sum(dp.real**2+dp.imag**2)+jnp.sum(dq.real**2+dq.imag**2))
            return solve_lbfgsb_jax(cost_fn_jax, delay_init.flatten(),
                                    self.max_iter, self.tol)
        except ImportError:
            return None

    def _initial_estimate(self, vis_obs, vis_model, ant1, ant2, freqs, n_ant):
        """BFS phase-slope propagation from ref antenna."""
        adj = build_antenna_graph(ant1, ant2, n_ant)
        order = bfs_order(adj, self.ref_ant)
        bl_delay = _bl_delay_estimate(vis_obs, vis_model, ant1, ant2, freqs)
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


def _bl_delay_estimate(vis_obs, vis_model, ant1, ant2, freqs):
    """Per-baseline delay from phase slope across frequency."""
    n_bl = vis_obs.shape[0]
    delays = np.zeros((n_bl, 2), dtype=np.float64)
    if len(freqs) < 2:
        return delays
    for k in range(n_bl):
        for pol in range(2):
            ratio = vis_obs[k, :, pol, pol] * np.conj(vis_model[k, :, pol, pol] + 1e-30)
            phase = np.unwrap(np.angle(ratio))
            delays[k, pol] = -np.polyfit(freqs, phase, 1)[0] / (2 * np.pi) * 1e9
    return delays
