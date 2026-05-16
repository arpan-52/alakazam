"""ALAKAZAM v1 KC (Cross Delay) Solver.

Receives: (n_bl, n_chan, 2, 2) — time-averaged, freq retained.
Returns:  (n_ant, 2, 2).

Single global parameter tau_cross.
J = diag(e^{-2pi i tau nu}, 1) same for all antennas.
Backend: boa (Kokkos LM)

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations
import logging
import time as _time
from typing import Any, Dict

import numpy as np

from . import JonesSolver
from ..jones.constructors import cross_delay_to_jones

logger = logging.getLogger("alakazam")


class CrossDelaySolver(JonesSolver):
    jones_type = "KC"

    def solve(self, vis_obs, vis_model, ant1, ant2, freqs, n_ant,
              init_jones=None):
        t0 = _time.time()

        tau_init = self._initial_estimate(vis_obs, vis_model, freqs)

        from .boa_backend import import_boa, vis22f_to_boa, make_opts
        boa = import_boa()

        res = boa.solve_KC(
            vis22f_to_boa(vis_obs), vis22f_to_boa(vis_model),
            ant1.astype(np.int32, copy=False), ant2.astype(np.int32, copy=False),
            freqs.astype(np.float64, copy=False), n_ant, self.ref_ant,
            make_opts(self.max_iter, self.tol),
            np.array([tau_init], dtype=np.float64))

        tau = float(res["params"][0])
        freq_mid = np.array([np.mean(freqs)])
        J = cross_delay_to_jones(tau, freq_mid, n_ant)[:, 0]

        # Store delay as (n_ant, 2): [tau_cross, 0] per antenna
        delay = np.zeros((n_ant, 2), dtype=np.float64)
        delay[:, 0] = tau

        wall = _time.time() - t0
        return {
            "jones": J,
            "delay": delay,
            "converged": bool(res["converged"]),
            "n_iter": int(res["n_iter"]),
            "cost": float(res["cost"]),
            "wall_time": wall,
            "solver_backend": self.backend,
        }

    def _initial_estimate(self, vis_obs, vis_model, freqs):
        """Cross-hand phase slope for initial tau estimate."""
        if len(freqs) < 2:
            return 0.0
        ratio = vis_obs[:, :, 0, 1] / (vis_model[:, :, 0, 1] + 1e-30)
        avg = np.mean(ratio, axis=0)
        phase = np.unwrap(np.angle(avg))
        return -np.polyfit(freqs, phase, 1)[0] / (2 * np.pi) * 1e9
