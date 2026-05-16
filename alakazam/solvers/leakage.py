"""ALAKAZAM v1 D (Leakage) Solver.

Receives: (n_bl, 2, 2) — time+freq averaged.
Returns:  (n_ant, 2, 2).

Initial guess: cross/parallel ratio from corrected data.
Backend: boa (Kokkos LM)
Ref ant: d_pq[ref] = 0, d_qp[ref] FREE.

Feed-basis aware: uses appropriate initial guess for LINEAR vs CIRCULAR.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations
import logging
import time as _time
from typing import Any, Dict

import numpy as np

from . import JonesSolver, initial_guess_leakage
from ..jones.constructors import leakage_to_jones

logger = logging.getLogger("alakazam")


class LeakageSolver(JonesSolver):
    jones_type = "D"

    def solve(self, vis_obs, vis_model, ant1, ant2, freqs, n_ant,
              init_jones=None):
        t0 = _time.time()

        if vis_obs.ndim == 4:
            obs = vis_obs.mean(axis=1)
            model = vis_model.mean(axis=1)
        else:
            obs, model = vis_obs, vis_model

        logger.info(f"D solve: feed_basis={self.feed_basis}")

        d_pq_init, d_qp_init = initial_guess_leakage(
            obs, model, ant1, ant2, n_ant, self.ref_ant,
            feed_basis=self.feed_basis)

        from .boa_backend import import_boa, vis22_to_boa, jones4_to_22, make_opts
        boa = import_boa()
        freqs_empty = np.zeros(0, dtype=np.float64)

        # Pack initial guess: [Re(d_pq), Im(d_pq), Re(d_qp), Im(d_qp)] per non-ref ant,
        # then [Re(d_qp_ref), Im(d_qp_ref)] for ref ant (only d_pq[ref]=0).
        init_params = []
        for a in range(n_ant):
            if a != self.ref_ant:
                init_params.extend([d_pq_init[a].real, d_pq_init[a].imag,
                                    d_qp_init[a].real, d_qp_init[a].imag])
        init_params.extend([d_qp_init[self.ref_ant].real, d_qp_init[self.ref_ant].imag])

        res = boa.solve_D(
            vis22_to_boa(obs), vis22_to_boa(model),
            ant1.astype(np.int32, copy=False), ant2.astype(np.int32, copy=False),
            freqs_empty, n_ant, self.ref_ant,
            make_opts(self.max_iter, self.tol),
            np.array(init_params, dtype=np.float64))

        J = jones4_to_22(res["jones"])  # (n_ant, 2, 2)
        dpq_out = J[:, 0, 1]
        dqp_out = J[:, 1, 0]
        dpq_out[self.ref_ant] = 0.0  # belt-and-suspenders

        wall = _time.time() - t0
        return {
            "jones": leakage_to_jones(dpq_out, dqp_out),
            "converged": bool(res["converged"]),
            "n_iter": int(res["n_iter"]),
            "cost": float(res["cost"]),
            "wall_time": wall,
            "solver_backend": self.backend,
        }
