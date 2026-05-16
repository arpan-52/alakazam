"""ALAKAZAM v1 K (Parallel Delay) Solver.

Receives: (n_bl, n_chan, 2, 2) — time-averaged, freq retained.
Returns:  (n_ant, 2, 2) — one solution per cell.

Initial guess: FFT-based fringe fitting (8x zero-padding) -> BFS propagation.
Backend: boa (Kokkos LM)
Ref ant: delay[ref, :] = 0.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations
import logging
import time as _time
from typing import Any, Dict

import numpy as np

from . import JonesSolver, build_antenna_graph, bfs_order
from ..jones.constructors import parallel_delay_to_jones

logger = logging.getLogger("alakazam")


class ParallelDelaySolver(JonesSolver):
    jones_type = "K"

    def solve(self, vis_obs, vis_model, ant1, ant2, freqs, n_ant,
              init_jones=None):
        t0 = _time.time()
        logger.debug(f"K solve: n_ant={n_ant} n_bl={vis_obs.shape[0]} "
                     f"n_freq={vis_obs.shape[1]} backend={self.backend}")

        delay_init, vis_obs_w, vis_mod_w = self._initial_estimate(
            vis_obs, vis_model, ant1, ant2, freqs, n_ant)

        from .boa_backend import (import_boa, vis22f_to_boa, make_opts,
                                  reconstruct_delay)
        boa = import_boa()

        # Pack FFT+BFS initial guess: [tau_p, tau_q] per non-ref antenna in ns
        init_params = []
        for a in range(n_ant):
            if a != self.ref_ant:
                init_params.extend([delay_init[a, 0], delay_init[a, 1]])

        res = boa.solve_K(
            vis22f_to_boa(vis_obs), vis22f_to_boa(vis_model),
            ant1.astype(np.int32, copy=False), ant2.astype(np.int32, copy=False),
            freqs.astype(np.float64, copy=False), n_ant, self.ref_ant,
            make_opts(self.max_iter, self.tol),
            np.array(init_params, dtype=np.float64))

        delay = reconstruct_delay(res["params"], n_ant, self.ref_ant)
        freq_mid = np.array([np.mean(freqs)])
        J = parallel_delay_to_jones(delay, freq_mid)[:, 0]  # (n_ant, 2, 2)

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

    def _initial_estimate(self, vis_obs, vis_model, ant1, ant2, freqs, n_ant):
        """FFT fringe fitting + BFS propagation for initial delay guess."""
        n_bl, n_freq = vis_obs.shape[:2]
        nfft = n_freq * 4
        df = freqs[1] - freqs[0]
        delay_axis = np.fft.fftfreq(nfft, d=df)
        vis_obs_w = vis_obs.copy()
        vis_mod_w = vis_model.copy()

        for pol in range(2):
            xsp = vis_obs[:, :, pol, pol] * np.conj(vis_model[:, :, pol, pol])
            bad = np.abs(vis_model[:, :, pol, pol]) < 1e-30
            xsp[bad] = 0.0
            n_good = (~bad).sum(axis=1).astype(np.float64)

            spectra = np.fft.fft(xsp, n=nfft, axis=1)
            peak_idx = np.argmax(np.abs(spectra), axis=1)
            tau_best = -delay_axis[peak_idx]

            derot_phase = (2.0 * np.pi * tau_best[:, np.newaxis]
                           * freqs[np.newaxis, :])
            xsp_d = xsp * np.exp(1j * derot_phase)

            phi_res = np.angle(xsp_d)
            phi_res[bad] = 0.0
            rms_sq = np.sum(phi_res ** 2, axis=1) / np.maximum(n_good, 1.0)

            noise_floor = float(np.median(rms_sq)) * 0.1 + 1e-4
            weight = 1.0 / (rms_sq + noise_floor)
            w_sqrt = np.sqrt(weight / weight.max())

            vis_obs_w[:, :, pol, pol] *= w_sqrt[:, np.newaxis]
            vis_mod_w[:, :, pol, pol] *= w_sqrt[:, np.newaxis]

        adj = build_antenna_graph(ant1, ant2, n_ant)
        order = bfs_order(adj, self.ref_ant)
        bl_delay = _bl_delay_fft(vis_obs_w, vis_mod_w, ant1, ant2, freqs)
        delay = np.zeros((n_ant, 2), dtype=np.float64)
        solved = np.zeros(n_ant, dtype=bool)
        solved[self.ref_ant] = True

        for a in order:
            if a == self.ref_ant:
                continue
            for k, (a1, a2) in enumerate(zip(ant1, ant2)):
                if a1 == a and solved[a2]:
                    delay[a] = bl_delay[k] + delay[a2]
                    solved[a] = True
                    break
                if a2 == a and solved[a1]:
                    delay[a] = -bl_delay[k] + delay[a1]
                    solved[a] = True
                    break

        delay[self.ref_ant, :] = 0.0
        return delay, vis_obs_w, vis_mod_w


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
    delay_axis = np.fft.fftfreq(nfft, d=df)

    for pol in range(2):
        xspec = vis_obs[:, :, pol, pol] * np.conj(vis_model[:, :, pol, pol])
        bad = np.abs(vis_model[:, :, pol, pol]) < 1e-30
        xspec[bad] = 0.0
        good_count = np.sum(~bad, axis=1)

        spectra = np.fft.fft(xspec, n=nfft, axis=1)
        peak_idx = np.argmax(np.abs(spectra), axis=1)

        delays[:, pol] = -delay_axis[peak_idx] * 1e9
        delays[good_count < 4, pol] = 0.0

    return delays
