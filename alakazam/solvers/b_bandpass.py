"""
B Bandpass Solver - Frequency-dependent complex gains.

Jones Matrix:
  B = diag(b_x(ν), b_y(ν)) where b(ν) = A(ν) × exp(iφ(ν))

Reference Constraint:
  φ_ref(ν) = 0 for all ν (phase zero for reference antenna)

Parameters:
  [A_x(ν), φ_x(ν), A_y(ν), φ_y(ν)] per antenna per channel

Averaging:
  - Time: YES (average over entire solint)
  - Freq: YES (can average to freq chunks per user solint)

Chain Initial Guess:
  Per channel: b_j(ν) = V_ref,j(ν) / M_ref,j(ν)
"""

import numpy as np
from scipy.optimize import least_squares
from numba import njit, prange
from typing import Dict, Any

from .base import JonesSolverBase, SolverMetadata
from .utils import find_ref_baselines


@njit(parallel=True, cache=True)
def _chain_bandpass_jit(
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    flags: np.ndarray,
    ref_ant: int,
    n_ant: int
) -> np.ndarray:
    """
    JIT-compiled chain solver for B bandpass.

    For each channel: b_j(ν) = V_ref,j(ν) / M_ref,j(ν)

    Parameters
    ----------
    vis_obs, vis_model : ndarray (n_bl, n_freq, 2, 2)
        Visibilities
    antenna1, antenna2 : ndarray (n_bl,)
        Antenna indices
    flags : ndarray (n_bl, n_freq, 2, 2)
        Flags
    ref_ant : int
        Reference antenna
    n_ant : int
        Number of antennas

    Returns
    -------
    gains : ndarray (n_ant, n_freq, 2, 2)
        Initial per-channel diagonal gains
    """
    n_bl, n_freq = vis_obs.shape[:2]
    gains = np.ones((n_ant, n_freq, 2, 2), dtype=np.complex128)

    # Process baselines in parallel
    for bl in prange(n_bl):
        a1 = antenna1[bl]
        a2 = antenna2[bl]

        # Check if this baseline contains reference
        if a1 == ref_ant:
            ant_j = a2
            flip = False
        elif a2 == ref_ant:
            ant_j = a1
            flip = True
        else:
            continue

        # For each channel and pol
        for ch in range(n_freq):
            for p in range(2):
                V = vis_obs[bl, ch, p, p]
                M = vis_model[bl, ch, p, p]
                F = flags[bl, ch, p, p]

                if F or np.abs(M) < 1e-10 or not np.isfinite(V):
                    continue

                # Chain solve
                if flip:
                    b_j = np.conj(V / M)
                else:
                    b_j = V / M

                gains[ant_j, ch, p, p] = b_j

    return gains


class BBandpassSolver(JonesSolverBase):
    """B bandpass solver."""

    metadata = SolverMetadata(
        jones_type='B',
        ref_constraint='phase_zero',
        can_avg_time=True,
        can_avg_freq=True,  # Can average to freq chunks
        description="Frequency-dependent complex gains (bandpass)"
    )

    def chain_initial_guess(
        self,
        vis_obs: np.ndarray,
        vis_model: np.ndarray,
        antenna1: np.ndarray,
        antenna2: np.ndarray,
        freq: np.ndarray,
        flags: np.ndarray,
        ref_ant: int,
        n_ant: int,
        **kwargs
    ) -> np.ndarray:
        """
        Chain solver for B bandpass (JIT-accelerated).

        For each channel: b_j(ν) = V_ref,j(ν) / M_ref,j(ν)

        Parameters
        ----------
        vis_obs, vis_model : ndarray (n_bl, n_freq, 2, 2)
            Time-averaged, freq may be chunked
        """
        n_freq = vis_obs.shape[1]

        if self.verbose:
            n_ref_bl = np.sum((antenna1 == ref_ant) | (antenna2 == ref_ant))
            print(f"[ALAKAZAM] Chain initial guess: {n_ref_bl} ref baselines, {n_freq} channels (JIT-accelerated)")

        # Call JIT-compiled chain solver
        gains_init = _chain_bandpass_jit(
            vis_obs, vis_model, antenna1, antenna2, flags, ref_ant, n_ant
        )

        return gains_init

    def residual(
        self,
        params: np.ndarray,
        vis_obs: np.ndarray,
        vis_model: np.ndarray,
        antenna1: np.ndarray,
        antenna2: np.ndarray,
        freq: np.ndarray,
        flags: np.ndarray,
        n_ant: int,
        ref_ant: int,
        **kwargs
    ) -> np.ndarray:
        """Compute residuals for B bandpass."""
        phase_only = kwargs.get('phase_only', False)
        n_freq = vis_obs.shape[1]

        gains = self.unpack_params(params, n_ant, ref_ant, n_freq=n_freq, phase_only=phase_only)

        residuals = _b_bandpass_residual(
            gains, vis_obs, vis_model, antenna1, antenna2, flags
        )

        return residuals

    def pack_params(
        self,
        jones: np.ndarray,
        ref_ant: int,
        **kwargs
    ) -> np.ndarray:
        """Pack bandpass to parameters."""
        phase_only = kwargs.get('phase_only', False)
        n_ant, n_freq = jones.shape[0], jones.shape[1]
        params = []

        for ant in range(n_ant):
            for ch in range(n_freq):
                b_x = jones[ant, ch, 0, 0]
                b_y = jones[ant, ch, 1, 1]

                if ant == ref_ant:
                    if not phase_only:
                        params.extend([np.abs(b_x), np.abs(b_y)])
                else:
                    if phase_only:
                        params.extend([np.angle(b_x), np.angle(b_y)])
                    else:
                        params.extend([
                            np.abs(b_x), np.angle(b_x),
                            np.abs(b_y), np.angle(b_y)
                        ])

        return np.array(params, dtype=np.float64)

    def unpack_params(
        self,
        params: np.ndarray,
        n_ant: int,
        ref_ant: int,
        **kwargs
    ) -> np.ndarray:
        """Unpack parameters to bandpass."""
        phase_only = kwargs.get('phase_only', False)
        n_freq = kwargs.get('n_freq', 1)
        
        gains = np.zeros((n_ant, n_freq, 2, 2), dtype=np.complex128)
        idx = 0

        for ant in range(n_ant):
            for ch in range(n_freq):
                if ant == ref_ant:
                    if phase_only:
                        gains[ant, ch, 0, 0] = 1.0
                        gains[ant, ch, 1, 1] = 1.0
                    else:
                        gains[ant, ch, 0, 0] = params[idx]
                        gains[ant, ch, 1, 1] = params[idx + 1]
                        idx += 2
                else:
                    if phase_only:
                        gains[ant, ch, 0, 0] = np.exp(1j * params[idx])
                        gains[ant, ch, 1, 1] = np.exp(1j * params[idx + 1])
                        idx += 2
                    else:
                        A_x, phi_x = params[idx], params[idx + 1]
                        A_y, phi_y = params[idx + 2], params[idx + 3]
                        gains[ant, ch, 0, 0] = A_x * np.exp(1j * phi_x)
                        gains[ant, ch, 1, 1] = A_y * np.exp(1j * phi_y)
                        idx += 4

        return gains

    def print_solution(
        self,
        jones: np.ndarray,
        working_ants: np.ndarray,
        ref_ant: int,
        convergence_info: Dict[str, Any]
    ):
        """Print B bandpass solution."""
        if not self.verbose:
            return

        n_freq = jones.shape[1]
        print(f"[ALAKAZAM] Cost: {convergence_info['cost_init']:.4e} -> {convergence_info['cost_final']:.4e}")
        print(f"[ALAKAZAM] Bandpass solved: {len(working_ants)} ants × {n_freq} channels")


@njit(cache=True)
def _b_bandpass_residual(
    gains: np.ndarray,
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    flags: np.ndarray
) -> np.ndarray:
    """Compute residuals for B bandpass."""
    n_bl = len(antenna1)
    n_freq = vis_obs.shape[1]

    n_unflagged = 0
    for bl in range(n_bl):
        for ch in range(n_freq):
            if not flags[bl, ch, 0, 0]:
                n_unflagged += 1
            if not flags[bl, ch, 1, 1]:
                n_unflagged += 1

    residuals = np.zeros(n_unflagged * 2, dtype=np.float64)
    res_idx = 0

    for bl in range(n_bl):
        a1 = antenna1[bl]
        a2 = antenna2[bl]

        for ch in range(n_freq):
            b1_x = gains[a1, ch, 0, 0]
            b1_y = gains[a1, ch, 1, 1]
            b2_x = gains[a2, ch, 0, 0]
            b2_y = gains[a2, ch, 1, 1]

            if not flags[bl, ch, 0, 0]:
                M_xx = vis_model[bl, ch, 0, 0]
                V_predicted = b1_x * M_xx * np.conj(b2_x)
                V_observed = vis_obs[bl, ch, 0, 0]

                residuals[res_idx] = (V_observed - V_predicted).real
                residuals[res_idx + 1] = (V_observed - V_predicted).imag
                res_idx += 2

            if not flags[bl, ch, 1, 1]:
                M_yy = vis_model[bl, ch, 1, 1]
                V_predicted = b1_y * M_yy * np.conj(b2_y)
                V_observed = vis_obs[bl, ch, 1, 1]

                residuals[res_idx] = (V_observed - V_predicted).real
                residuals[res_idx + 1] = (V_observed - V_predicted).imag
                res_idx += 2

    return residuals


__all__ = ['BBandpassSolver']
