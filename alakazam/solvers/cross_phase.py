"""ALAKAZAM v1 CP (Cross Phase) Solver.

Receives: (n_bl, 2, 2) — time+freq averaged.
Returns:  (n_ant, 2, 2).

Single global parameter phi_cross.
J = diag(1, e^{i phi}) same for all antennas.
Backend: boa (Kokkos LM)

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations
import logging
import time as _time
from typing import Any, Dict

import numpy as np

from . import JonesSolver, initial_guess_cross_phase
from ..jones.constructors import cross_phase_to_jones

logger = logging.getLogger("alakazam")


class CrossPhaseSolver(JonesSolver):
    jones_type = "CP"

    def solve(self, vis_obs, vis_model, ant1, ant2, freqs, n_ant,
              init_jones=None):
        t0 = _time.time()

        if vis_obs.ndim == 4:
            obs = vis_obs.mean(axis=1)
            model = vis_model.mean(axis=1)
        else:
            obs, model = vis_obs, vis_model

        phi_init = initial_guess_cross_phase(obs, model, ant1, ant2)

        from .boa_backend import import_boa, vis22_to_boa, make_opts
        boa = import_boa()
        freqs_empty = np.zeros(0, dtype=np.float64)

        res = boa.solve_CP(
            vis22_to_boa(obs), vis22_to_boa(model),
            ant1.astype(np.int32, copy=False), ant2.astype(np.int32, copy=False),
            freqs_empty, n_ant, self.ref_ant,
            make_opts(self.max_iter, self.tol),
            np.array([phi_init], dtype=np.float64))

        phi = float(res["params"][0])
        J = cross_phase_to_jones(phi, n_ant)

        wall = _time.time() - t0
        return {
            "jones": J,
            "converged": bool(res["converged"]),
            "n_iter": int(res["n_iter"]),
            "cost": float(res["cost"]),
            "wall_time": wall,
            "solver_backend": self.backend,
        }
