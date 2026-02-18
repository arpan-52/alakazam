"""ALAKAZAM K (Delay) Solver.

Solves for per-antenna delay in nanoseconds using fringe fitting.
Frequency-dependent diagonal Jones: J[a,f] = exp(-2πi τ_a ν_f)

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import least_squares

from . import JonesSolver, build_antenna_graph, bfs_order
from ..jones.constructors import delay_to_jones
from ..jones.algebra import compute_residual_diag_freq

logger = logging.getLogger("alakazam")


class KDelaySolver(JonesSolver):
    jones_type = "K"

    def solve(
        self,
        vis_obs: np.ndarray,     # (n_bl, n_chan, 2, 2)
        vis_model: np.ndarray,
        ant1: np.ndarray,
        ant2: np.ndarray,
        freqs: np.ndarray,
        n_ant: int,
        init_jones: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:

        n_bl, n_chan = vis_obs.shape[:2]

        # Initial delay estimate via phase slope across frequency
        delay_init = np.zeros((n_ant, 2), dtype=np.float64)
        adj = build_antenna_graph(ant1, ant2, n_ant)
        order = bfs_order(adj, self.ref_ant)

        # Simple phase-slope estimator per baseline
        phase_slope = np.zeros((n_ant, 2), dtype=np.float64)
        bl_delay = _estimate_delay_per_baseline(vis_obs, vis_model, ant1, ant2, freqs)

        for a in order:
            if a == self.ref_ant:
                phase_slope[a] = 0.0
                continue
            # Find baselines involving this antenna with already-solved antennas
            for k, (a1, a2) in enumerate(zip(ant1, ant2)):
                if a1 == a and phase_slope[a2, 0] != 0:
                    phase_slope[a] = bl_delay[k] + phase_slope[a2]
                    break
                if a2 == a and phase_slope[a1, 0] != 0:
                    phase_slope[a] = -bl_delay[k] + phase_slope[a1]
                    break

        delay_init = phase_slope

        def _residual(x):
            delay = x.reshape(n_ant, 2)
            delay[self.ref_ant] = 0.0
            J = delay_to_jones(delay, freqs)
            return compute_residual_diag_freq(J, vis_obs, vis_model, ant1, ant2)

        x0 = delay_init.flatten()
        result = least_squares(
            _residual, x0,
            method="lm",
            max_nfev=self.max_iter * len(x0),
            ftol=self.tol, xtol=self.tol, gtol=self.tol,
        )

        delay = result.x.reshape(n_ant, 2)
        delay[self.ref_ant] = 0.0
        J = delay_to_jones(delay, freqs)

        return {
            "jones":         J,
            "native_params": {"type": "K", "delay": delay},
            "converged":     result.success,
            "n_iter":        result.nfev,
            "cost":          float(result.cost),
        }


def _estimate_delay_per_baseline(vis_obs, vis_model, ant1, ant2, freqs):
    """Estimate delay per baseline from phase slope across frequency."""
    n_bl = vis_obs.shape[0]
    delays = np.zeros((n_bl, 2), dtype=np.float64)
    dfreq = freqs[-1] - freqs[0] if len(freqs) > 1 else 1.0

    for k in range(n_bl):
        ratio = vis_obs[k, :, 0, 0] * np.conj(vis_model[k, :, 0, 0] + 1e-30)
        phase = np.unwrap(np.angle(ratio))
        if len(freqs) > 1:
            slope = np.polyfit(freqs, phase, 1)[0]
            delays[k, 0] = -slope / (2 * np.pi) * 1e9  # ns
        ratio_q = vis_obs[k, :, 1, 1] * np.conj(vis_model[k, :, 1, 1] + 1e-30)
        phase_q = np.unwrap(np.angle(ratio_q))
        if len(freqs) > 1:
            slope_q = np.polyfit(freqs, phase_q, 1)[0]
            delays[k, 1] = -slope_q / (2 * np.pi) * 1e9

    return delays
