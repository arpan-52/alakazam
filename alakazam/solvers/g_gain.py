"""
G Gain Solver - Time-variable complex gains.

Jones Matrix:
  G = diag(g_x, g_y) where g = A × exp(iφ)

Reference Constraint:
  φ_ref = 0 (phase zero for reference antenna)

Parameters:
  - Amplitude + Phase: [A_x, φ_x, A_y, φ_y] per antenna
  - Phase-only: [φ_x, φ_y] per antenna
  - Ref params: [A_x, A_y] (amplitude only, phase=0)

Averaging:
  - Time: YES (average over solint)
  - Freq: YES (average over solint)

Chain Initial Guess:
  g_j = V_ref,j / M_ref,j  (since φ_ref = 0, g_ref ≈ real)
"""

import numpy as np
from scipy.optimize import least_squares
from numba import njit, prange
from typing import Dict, Any

from .base import JonesSolverBase, SolverMetadata
from .utils import find_ref_baselines


@njit(parallel=True, cache=True)
def _chain_gains_jit(
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    flags: np.ndarray,
    ref_ant: int,
    n_ant: int
) -> np.ndarray:
    """
    JIT-compiled chain solver for G gains.

    For φ_ref = 0:
      g_j = V_ref,j / M_ref,j

    Parameters
    ----------
    vis_obs, vis_model : ndarray (n_bl, 2, 2)
        Visibilities
    antenna1, antenna2 : ndarray (n_bl,)
        Antenna indices
    flags : ndarray (n_bl, 2, 2)
        Flags
    ref_ant : int
        Reference antenna
    n_ant : int
        Number of antennas

    Returns
    -------
    gains : ndarray (n_ant, 2, 2)
        Initial diagonal gains
    """
    gains = np.ones((n_ant, 2, 2), dtype=np.complex128)
    n_bl = len(antenna1)

    # Find baselines to reference and solve
    for bl in prange(n_bl):
        a1 = antenna1[bl]
        a2 = antenna2[bl]

        # Check if this baseline contains reference
        if a1 == ref_ant:
            # Baseline is [ref, j]
            ant_j = a2
            flip = False
        elif a2 == ref_ant:
            # Baseline is [j, ref]
            ant_j = a1
            flip = True
        else:
            # Doesn't contain reference
            continue

        # Solve for each pol
        for p in range(2):
            V = vis_obs[bl, p, p]
            M = vis_model[bl, p, p]
            F = flags[bl, p, p]

            # Check if flagged or invalid
            if F or np.abs(M) < 1e-10 or not np.isfinite(V):
                continue

            # Chain solve
            if flip:
                # V = M × g_ref^H × g_j^H
                # g_j = (V / M)^* (assuming g_ref ≈ real)
                g_j = np.conj(V / M)
            else:
                # V = g_j × M × g_ref^H
                # g_j = V / M (assuming g_ref ≈ real)
                g_j = V / M

            gains[ant_j, p, p] = g_j

    return gains


class GGainSolver(JonesSolverBase):
    """G complex gain solver."""

    metadata = SolverMetadata(
        jones_type='G',
        ref_constraint='phase_zero',
        can_avg_time=True,
        can_avg_freq=True,
        description="Complex gains (amplitude + phase)"
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
        Chain solver for G gains (JIT-accelerated).

        For φ_ref = 0:
          g_j = V_ref,j / M_ref,j

        Parameters
        ----------
        vis_obs, vis_model : ndarray (n_bl, 2, 2)
            Observed and model visibilities (time+freq averaged)
        flags : ndarray (n_bl, 2, 2)
            Flags

        Returns
        -------
        gains_init : ndarray (n_ant, 2, 2)
            Initial diagonal gains
        """
        # Count reference baselines for logging
        if self.verbose:
            n_ref_bl = np.sum((antenna1 == ref_ant) | (antenna2 == ref_ant))
            print(f"[ALAKAZAM] Chain initial guess: {n_ref_bl} ref baselines (JIT-accelerated)")

        # Call JIT-compiled chain solver
        gains_init = _chain_gains_jit(
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
        """Compute residuals for G gain optimization."""
        phase_only = kwargs.get('phase_only', False)

        # Unpack gains
        gains = self.unpack_params(params, n_ant, ref_ant, phase_only=phase_only)

        # Compute residuals
        residuals = _g_gain_residual(
            gains, vis_obs, vis_model, antenna1, antenna2, flags
        )

        return residuals

    def pack_params(
        self,
        jones: np.ndarray,
        ref_ant: int,
        **kwargs
    ) -> np.ndarray:
        """
        Pack gains to parameter array.

        jones : ndarray (n_ant, 2, 2)
            Diagonal gain matrices

        For amplitude + phase:
          - Ref: [A_x, A_y]
          - Others: [A_x, φ_x, A_y, φ_y]

        For phase-only:
          - Ref: []
          - Others: [φ_x, φ_y]
        """
        phase_only = kwargs.get('phase_only', False)
        n_ant = jones.shape[0]
        params = []

        for ant in range(n_ant):
            g_x = jones[ant, 0, 0]
            g_y = jones[ant, 1, 1]

            if ant == ref_ant:
                if not phase_only:
                    # Ref: amplitude only
                    params.extend([np.abs(g_x), np.abs(g_y)])
            else:
                if phase_only:
                    # Phase only
                    params.extend([np.angle(g_x), np.angle(g_y)])
                else:
                    # Amplitude + phase
                    params.extend([
                        np.abs(g_x), np.angle(g_x),
                        np.abs(g_y), np.angle(g_y)
                    ])

        return np.array(params, dtype=np.float64)

    def unpack_params(
        self,
        params: np.ndarray,
        n_ant: int,
        ref_ant: int,
        **kwargs
    ) -> np.ndarray:
        """Unpack parameters to gain matrices."""
        phase_only = kwargs.get('phase_only', False)
        gains = np.zeros((n_ant, 2, 2), dtype=np.complex128)
        idx = 0

        for ant in range(n_ant):
            if ant == ref_ant:
                if phase_only:
                    # Ref: unity gains
                    gains[ant, 0, 0] = 1.0
                    gains[ant, 1, 1] = 1.0
                else:
                    # Ref: amplitude only, phase=0
                    gains[ant, 0, 0] = params[idx]
                    gains[ant, 1, 1] = params[idx + 1]
                    idx += 2
            else:
                if phase_only:
                    # Phase only
                    gains[ant, 0, 0] = np.exp(1j * params[idx])
                    gains[ant, 1, 1] = np.exp(1j * params[idx + 1])
                    idx += 2
                else:
                    # Amplitude + phase
                    A_x, phi_x = params[idx], params[idx + 1]
                    A_y, phi_y = params[idx + 2], params[idx + 3]
                    gains[ant, 0, 0] = A_x * np.exp(1j * phi_x)
                    gains[ant, 1, 1] = A_y * np.exp(1j * phi_y)
                    idx += 4

        return gains

    def print_solution(
        self,
        jones: np.ndarray,
        working_ants: np.ndarray,
        ref_ant: int,
        convergence_info: Dict[str, Any]
    ):
        """Print G gain solution."""
        if not self.verbose:
            return

        gains = jones  # (n_working, 2, 2)

        print(f"[ALAKAZAM] Cost: {convergence_info['cost_init']:.4e} -> {convergence_info['cost_final']:.4e}")
        print(f"[ALAKAZAM] Final gains:")

        for i in range(len(working_ants)):
            ant_full = working_ants[i]
            g_x = gains[i, 0, 0]
            g_y = gains[i, 1, 1]

            A_x, phi_x = np.abs(g_x), np.angle(g_x) * 180 / np.pi
            A_y, phi_y = np.abs(g_y), np.angle(g_y) * 180 / np.pi

            status = "[ref]" if ant_full == ref_ant else ""
            print(f"        Ant {ant_full:2d}: X={A_x:6.3f}∠{phi_x:+7.2f}°, Y={A_y:6.3f}∠{phi_y:+7.2f}°  {status}")


@njit(cache=True)
def _g_gain_residual(
    gains: np.ndarray,
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    flags: np.ndarray
) -> np.ndarray:
    """
    Compute residuals for G gain solver.

    Parameters
    ----------
    gains : ndarray (n_ant, 2, 2)
        Diagonal gain matrices
    vis_obs, vis_model : ndarray (n_bl, 2, 2)
        Visibilities
    antenna1, antenna2 : ndarray (n_bl,)
        Antenna indices
    flags : ndarray (n_bl, 2, 2)
        Flags

    Returns
    -------
    residuals : ndarray
        Flattened real residuals
    """
    n_bl = len(antenna1)

    # Count unflagged
    n_unflagged = 0
    for bl in range(n_bl):
        if not flags[bl, 0, 0]:
            n_unflagged += 1
        if not flags[bl, 1, 1]:
            n_unflagged += 1

    residuals = np.zeros(n_unflagged * 2, dtype=np.float64)
    res_idx = 0

    for bl in range(n_bl):
        a1 = antenna1[bl]
        a2 = antenna2[bl]

        g1_x = gains[a1, 0, 0]
        g1_y = gains[a1, 1, 1]
        g2_x = gains[a2, 0, 0]
        g2_y = gains[a2, 1, 1]

        # XX
        if not flags[bl, 0, 0]:
            M_xx = vis_model[bl, 0, 0]
            V_predicted = g1_x * M_xx * np.conj(g2_x)
            V_observed = vis_obs[bl, 0, 0]

            residuals[res_idx] = (V_observed - V_predicted).real
            residuals[res_idx + 1] = (V_observed - V_predicted).imag
            res_idx += 2

        # YY
        if not flags[bl, 1, 1]:
            M_yy = vis_model[bl, 1, 1]
            V_predicted = g1_y * M_yy * np.conj(g2_y)
            V_observed = vis_obs[bl, 1, 1]

            residuals[res_idx] = (V_observed - V_predicted).real
            residuals[res_idx + 1] = (V_observed - V_predicted).imag
            res_idx += 2

    return residuals


__all__ = ['GGainSolver']
