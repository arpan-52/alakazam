"""
Kcross Delay Solver - Crosshand frequency-dependent delays.

Jones Matrix:
  Kcross affects XY and YX correlations
  K_cross = [[1, exp(-2πi τ_xy ν)],
             [exp(-2πi τ_yx ν), 1]]

Reference Constraint:
  K_ref = I (identity)

Parameters:
  [τ_xy, τ_yx] per antenna in nanoseconds

Averaging:
  - Time: YES (average over solint)
  - Freq: NO (need full freq for phase slope fitting)

Chain Initial Guess:
  For each frequency:
    K_j(ν) = V_ref,j(ν) / M_ref,j(ν)  (since K_ref = I)
    φ_xy(ν) = angle(K_j[0,1](ν))
    φ_yx(ν) = angle(K_j[1,0](ν))
  Fit: τ_xy = -d(φ_xy)/d(ν) / (2π)
       τ_yx = -d(φ_yx)/d(ν) / (2π)
"""

import numpy as np
from scipy.optimize import least_squares
from numba import njit
from typing import Dict, Any

from .base import JonesSolverBase, SolverMetadata
from .utils import fit_phase_slope_weighted, find_ref_baselines


class KcrossDelaySolver(JonesSolverBase):
    """Kcross crosshand delay solver."""

    metadata = SolverMetadata(
        jones_type='Kcross',
        ref_constraint='identity',
        can_avg_time=True,
        can_avg_freq=False,  # Need full freq for delay fitting
        description="Crosshand delays from XY/YX phase slope across frequency"
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
        Chain solver for Kcross delays.

        For J_ref = I:
          K_j(ν) = V_ref,j(ν) / M_ref,j(ν)
          φ_xy(ν) = angle(K_j[0,1](ν))
          φ_yx(ν) = angle(K_j[1,0](ν))
          τ_xy = fit delay from φ_xy vs ν
          τ_yx = fit delay from φ_yx vs ν

        Returns
        -------
        delays_init : ndarray (n_ant, 2)
            Initial delays in nanoseconds [τ_xy, τ_yx]
        """
        n_freq = len(freq)
        delays_init = np.zeros((n_ant, 2), dtype=np.float64)

        # Find baselines to reference
        ref_baselines = find_ref_baselines(antenna1, antenna2, ref_ant)

        if self.verbose:
            print(f"[ALAKAZAM] Chain initial guess: {len(ref_baselines)} ref baselines")

        # For each baseline to reference
        for bl_idx, ant_j, flip in ref_baselines:
            # For each crosshand correlation (XY, YX)
            for corr_idx, (p1, p2) in enumerate([(0, 1), (1, 0)]):
                V = vis_obs[bl_idx, :, p1, p2]
                M = vis_model[bl_idx, :, p1, p2]
                F = flags[bl_idx, :, p1, p2]

                # Mask: unflagged + valid data
                mask = ~F & (np.abs(M) > 1e-10) & np.isfinite(V) & np.isfinite(M)

                if np.sum(mask) < 10:
                    continue  # Not enough data

                # Chain solve: K_j = V / M (since K_ref = I)
                if flip:
                    # V = M × K_ref^H × K_j^H = M × K_j^H
                    # K_j = (V / M)^*
                    K_j = np.conj(V[mask] / M[mask])
                else:
                    # V = K_j × M × K_ref^H = K_j × M
                    # K_j = V / M
                    K_j = V[mask] / M[mask]

                # Extract phase
                phase = np.angle(K_j)
                phase_unwrap = np.unwrap(phase)
                freq_valid = freq[mask]

                # Fit delay from phase slope
                delay_ns = fit_phase_slope_weighted(phase_unwrap, freq_valid)

                delays_init[ant_j, corr_idx] = delay_ns

        return delays_init

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
        """
        Compute residuals for Kcross delay optimization.

        Apply Kcross delays to model and compute complex residuals for XY/YX.
        """
        # Unpack delays
        delays = self.unpack_params(params, n_ant, ref_ant)

        # Compute residuals
        residuals = _kcross_delay_residual(
            delays, vis_obs, vis_model, antenna1, antenna2, freq, flags
        )

        return residuals

    def pack_params(
        self,
        jones: np.ndarray,
        ref_ant: int,
        **kwargs
    ) -> np.ndarray:
        """
        Pack delays to parameter array.

        jones : ndarray (n_ant, 2)
            Delays [τ_xy, τ_yx] in nanoseconds

        Returns params excluding reference antenna.
        """
        n_ant = jones.shape[0]
        params = []

        for ant in range(n_ant):
            if ant != ref_ant:
                params.extend([jones[ant, 0], jones[ant, 1]])

        return np.array(params, dtype=np.float64)

    def unpack_params(
        self,
        params: np.ndarray,
        n_ant: int,
        ref_ant: int,
        **kwargs
    ) -> np.ndarray:
        """
        Unpack parameters to delays.

        Returns delays (n_ant, 2) with ref=0.
        """
        delays = np.zeros((n_ant, 2), dtype=np.float64)
        idx = 0

        for ant in range(n_ant):
            if ant != ref_ant:
                delays[ant, 0] = params[idx]
                delays[ant, 1] = params[idx + 1]
                idx += 2

        return delays

    def print_solution(
        self,
        jones: np.ndarray,
        working_ants: np.ndarray,
        ref_ant: int,
        convergence_info: Dict[str, Any]
    ):
        """Print Kcross delay solution."""
        if not self.verbose:
            return

        delays = jones  # (n_working, 2)

        print(f"[ALAKAZAM] Cost: {convergence_info['cost_init']:.4e} -> {convergence_info['cost_final']:.4e}")
        print(f"[ALAKAZAM] Final crosshand delays (ns):")

        for i in range(len(working_ants)):
            ant_full = working_ants[i]
            status = "[ref]" if ant_full == ref_ant else ""
            print(f"        Ant {ant_full:2d}: XY={delays[i, 0]:+8.3f}, YX={delays[i, 1]:+8.3f}  {status}")


@njit(cache=True)
def _kcross_delay_residual(
    delays: np.ndarray,
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    freq: np.ndarray,
    flags: np.ndarray
) -> np.ndarray:
    """
    Compute residuals for Kcross delay solver.

    Numba-compiled for speed.

    Parameters
    ----------
    delays : ndarray (n_ant, 2)
        Delays [τ_xy, τ_yx] in nanoseconds
    vis_obs, vis_model : ndarray (n_bl, n_freq, 2, 2)
        Visibilities
    antenna1, antenna2 : ndarray (n_bl,)
        Antenna indices
    freq : ndarray (n_freq,)
        Frequencies in Hz
    flags : ndarray (n_bl, n_freq, 2, 2)
        Flags (True = flagged)

    Returns
    -------
    residuals : ndarray
        Flattened real residuals for unflagged XY/YX data
    """
    n_bl = len(antenna1)
    n_freq = len(freq)

    # Count unflagged crosshand data points
    n_unflagged = 0
    for bl in range(n_bl):
        for ch in range(n_freq):
            if not flags[bl, ch, 0, 1]:  # XY
                n_unflagged += 1
            if not flags[bl, ch, 1, 0]:  # YX
                n_unflagged += 1

    # Allocate residual array (2 values per unflagged correlation: real + imag)
    residuals = np.zeros(n_unflagged * 2, dtype=np.float64)
    res_idx = 0

    # For each baseline and frequency
    for bl in range(n_bl):
        a1 = antenna1[bl]
        a2 = antenna2[bl]

        for ch in range(n_freq):
            f = freq[ch]

            # Compute Jones matrices for crosshand
            # Kcross = [[1, exp(-2πi τ_xy ν)],
            #           [exp(-2πi τ_yx ν), 1]]

            phase_xy1 = -2.0 * np.pi * delays[a1, 0] * f * 1e-9
            phase_yx1 = -2.0 * np.pi * delays[a1, 1] * f * 1e-9
            phase_xy2 = -2.0 * np.pi * delays[a2, 0] * f * 1e-9
            phase_yx2 = -2.0 * np.pi * delays[a2, 1] * f * 1e-9

            K1_xy = np.exp(1j * phase_xy1)
            K1_yx = np.exp(1j * phase_yx1)
            K2_xy = np.exp(1j * phase_xy2)
            K2_yx = np.exp(1j * phase_yx2)

            # Apply Kcross to model: V' = K1 × M × K2^H
            # For XY correlation: V'_xy = K1_xy × M_xy + K1_xy × M_yy × K2_yx^*
            # Simplified for diagonal M (assuming model is parallel-hand only):
            # V'_xy ≈ K1_xy × M_xy × K2_xy^*

            # XY correlation
            if not flags[bl, ch, 0, 1]:
                M_xy = vis_model[bl, ch, 0, 1]
                V_predicted = K1_xy * M_xy * np.conj(K2_xy)
                V_observed = vis_obs[bl, ch, 0, 1]

                residuals[res_idx] = (V_observed - V_predicted).real
                residuals[res_idx + 1] = (V_observed - V_predicted).imag
                res_idx += 2

            # YX correlation
            if not flags[bl, ch, 1, 0]:
                M_yx = vis_model[bl, ch, 1, 0]
                V_predicted = K1_yx * M_yx * np.conj(K2_yx)
                V_observed = vis_obs[bl, ch, 1, 0]

                residuals[res_idx] = (V_observed - V_predicted).real
                residuals[res_idx + 1] = (V_observed - V_predicted).imag
                res_idx += 2

    return residuals


__all__ = ['KcrossDelaySolver']
