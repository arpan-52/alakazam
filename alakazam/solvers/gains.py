"""ALAKAZAM v1 G (Gains) Solver.

Receives: (n_bl, 2, 2) — time+freq averaged by flow.
Returns:  (n_ant, 2, 2).

Initial guess: BFS gain-ratio extraction from parallel hands.
Backend: boa (Kokkos LM)

Two modes:
  full (default):  amps + phases solved; phase[ref]=0, ALL amps free
                   (including ref — required for the unit-model fluxscale
                   convention where gains absorb sqrt(flux)).
  phase_only:      only phases solved; amplitudes are fixed at 1 inside
                   the solver (not parameters); phase[ref]=0.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations
import logging
import time as _time
from typing import Any, Dict

import numpy as np

from . import JonesSolver, initial_guess_gain_bfs
from ..jones.constructors import gains_to_jones

logger = logging.getLogger("alakazam")


class GainsSolver(JonesSolver):
    jones_type = "G"

    def solve(self, vis_obs, vis_model, ant1, ant2, freqs, n_ant,
              init_jones=None):
        t0 = _time.time()

        # If multi-channel somehow arrives, average
        if vis_obs.ndim == 4:
            obs = vis_obs.mean(axis=1)
            model = vis_model.mean(axis=1)
        else:
            obs, model = vis_obs, vis_model

        amp_init, phase_init = initial_guess_gain_bfs(
            obs, model, ant1, ant2, n_ant, self.ref_ant)

        from .boa_backend import import_boa, vis22_to_boa, jones4_to_22, make_opts
        boa = import_boa()
        freqs_empty = np.zeros(0, dtype=np.float64)  # G is freq-independent

        if self.phase_only:
            # True phase-only mode: amplitudes are not parameters at all —
            # fixed at 1 inside the solver. Params: [phase_p, phase_q]
            # per non-ref antenna.
            init_params = []
            for a in range(n_ant):
                if a != self.ref_ant:
                    init_params.extend([phase_init[a, 0], phase_init[a, 1]])
            res = boa.solve_Gp(
                vis22_to_boa(obs), vis22_to_boa(model),
                ant1.astype(np.int32, copy=False), ant2.astype(np.int32, copy=False),
                freqs_empty, n_ant, self.ref_ant,
                make_opts(self.max_iter, self.tol),
                np.array(init_params, dtype=np.float64))
        else:
            # Full mode: [amp_p, phase_p, amp_q, phase_q] per non-ref antenna,
            # then [amp_p_ref, amp_q_ref] (ref phases fixed, amps free).
            init_params = []
            for a in range(n_ant):
                if a != self.ref_ant:
                    init_params.extend([amp_init[a, 0], phase_init[a, 0],
                                        amp_init[a, 1], phase_init[a, 1]])
            init_params.extend([amp_init[self.ref_ant, 0],
                                amp_init[self.ref_ant, 1]])
            res = boa.solve_G(
                vis22_to_boa(obs), vis22_to_boa(model),
                ant1.astype(np.int32, copy=False), ant2.astype(np.int32, copy=False),
                freqs_empty, n_ant, self.ref_ant,
                make_opts(self.max_iter, self.tol),
                np.array(init_params, dtype=np.float64))

        J = jones4_to_22(res["jones"])  # (n_ant, 2, 2)
        gp = J[:, 0, 0]
        gq = J[:, 1, 1]
        amp_out = np.column_stack([np.abs(gp), np.abs(gq)])
        phase_out = np.column_stack([np.angle(gp), np.angle(gq)])
        phase_out[self.ref_ant, :] = 0.0

        wall = _time.time() - t0
        return {
            "jones": gains_to_jones(amp_out, phase_out),
            "converged": bool(res["converged"]),
            "n_iter": int(res["n_iter"]),
            "cost": float(res["cost"]),
            "wall_time": wall,
            "solver_backend": self.backend,
        }
