"""
K Delay Solver - Frequency-dependent antenna delays.

Jones Matrix:
  K = diag(exp(-2πi τ_x ν), exp(-2πi τ_y ν))

Reference Constraint:
  J_ref = I (identity)

Parameters:
  [τ_x, τ_y] per antenna in nanoseconds

Averaging:
  - Time: YES (average over solint)
  - Freq: NO (need full freq for phase slope fitting)

Chain Initial Guess:
  For each frequency:
    J_j(ν) = V_ref,j(ν) / M_ref,j(ν)  (since J_ref = I)
    φ_j(ν) = angle(J_j(ν))
  Fit: τ_j = -d(φ_j)/d(ν) / (2π)
"""

import numpy as np
from scipy.optimize import least_squares
from numba import njit, prange
from typing import Dict, Any

from .base import JonesSolverBase, SolverMetadata
from .utils import fit_phase_slope_weighted, find_ref_baselines


@njit(cache=True)
def _fit_delay_jit(phase: np.ndarray, freq: np.ndarray) -> float:
    """
    JIT-compiled delay fitting from phase slope.

    Linear fit: φ = -2π × τ × ν
    Return: τ in nanoseconds
    """
    n = len(phase)
    if n < 3:
        return 0.0

    # Weighted linear fit
    freq_mean = np.mean(freq)
    phase_mean = np.mean(phase)

    numerator = np.sum((freq - freq_mean) * (phase - phase_mean))
    denominator = np.sum((freq - freq_mean) ** 2)

    if denominator < 1e-20:
        return 0.0

    # Slope in rad/Hz
    slope = numerator / denominator

    # Convert to delay in nanoseconds
    # φ = -2π × τ × ν  →  τ = -slope / (2π)
    delay_s = -slope / (2.0 * np.pi)
    delay_ns = delay_s * 1e9

    return delay_ns


@njit(parallel=True, cache=True)
def _chain_delays_jit(
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    freq: np.ndarray,
    flags: np.ndarray,
    ref_ant: int,
    n_ant: int
) -> np.ndarray:
    """
    JIT-compiled chain solver for K delays.

    For J_ref = I:
      J_j(ν) = V_ref,j(ν) / M_ref,j(ν)
      φ_j(ν) = angle(J_j(ν))
      τ_j = fit delay from φ_j vs ν

    Parameters
    ----------
    vis_obs, vis_model : ndarray (n_bl, n_freq, 2, 2)
        Visibilities
    antenna1, antenna2 : ndarray (n_bl,)
        Antenna indices
    freq : ndarray (n_freq,)
        Frequencies in Hz
    flags : ndarray (n_bl, n_freq, 2, 2)
        Flags
    ref_ant : int
        Reference antenna
    n_ant : int
        Number of antennas

    Returns
    -------
    delays : ndarray (n_ant, 2)
        Initial delays in nanoseconds [τ_x, τ_y]
    """
    delays = np.zeros((n_ant, 2), dtype=np.float64)
    n_bl = len(antenna1)
    n_freq = len(freq)

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

        # For each polarization
        for p in range(2):
            # Extract vis for this baseline and pol
            V = vis_obs[bl, :, p, p]
            M = vis_model[bl, :, p, p]
            F = flags[bl, :, p, p]

            # Build mask for valid data
            valid = np.zeros(n_freq, dtype=np.bool_)
            for f in range(n_freq):
                if not F[f] and np.abs(M[f]) > 1e-10 and np.isfinite(V[f]) and np.isfinite(M[f]):
                    valid[f] = True

            n_valid = np.sum(valid)
            if n_valid < 10:
                continue

            # Extract valid data
            V_valid = np.zeros(n_valid, dtype=np.complex128)
            M_valid = np.zeros(n_valid, dtype=np.complex128)
            freq_valid = np.zeros(n_valid, dtype=np.float64)
            idx = 0
            for f in range(n_freq):
                if valid[f]:
                    V_valid[idx] = V[f]
                    M_valid[idx] = M[f]
                    freq_valid[idx] = freq[f]
                    idx += 1

            # Chain solve: J_j = V / M
            if flip:
                J_j = np.conj(V_valid / M_valid)
            else:
                J_j = V_valid / M_valid

            # Extract phase and unwrap
            phase = np.angle(J_j)
            # Simple unwrapping (numba doesn't have np.unwrap)
            phase_unwrap = np.zeros(n_valid, dtype=np.float64)
            phase_unwrap[0] = phase[0]
            for i in range(1, n_valid):
                diff = phase[i] - phase_unwrap[i-1]
                # Unwrap by adding/subtracting 2π
                if diff > np.pi:
                    phase_unwrap[i] = phase[i] - 2*np.pi
                elif diff < -np.pi:
                    phase_unwrap[i] = phase[i] + 2*np.pi
                else:
                    phase_unwrap[i] = phase[i]

            # Fit delay
            delay_ns = _fit_delay_jit(phase_unwrap, freq_valid)
            delays[ant_j, p] = delay_ns

    return delays


class KDelaySolver(JonesSolverBase):
    """K delay solver."""

    metadata = SolverMetadata(
        jones_type='K',
        ref_constraint='identity',
        can_avg_time=True,
        can_avg_freq=False,  # Need full freq for delay fitting
        description="Antenna delays from phase slope across frequency"
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
        Chain solver for K delays (JIT-accelerated).

        For J_ref = I:
          J_j(ν) = V_ref,j(ν) / M_ref,j(ν)
          φ_j(ν) = angle(J_j(ν))
          τ_j = fit delay from φ_j vs ν

        Parameters
        ----------
        vis_obs, vis_model : ndarray (n_bl, n_freq, 2, 2)
            Observed and model visibilities
        freq : ndarray (n_freq,)
            Frequencies in Hz
        flags : ndarray (n_bl, n_freq, 2, 2)
            Flags

        Returns
        -------
        delays_init : ndarray (n_ant, 2)
            Initial delays in nanoseconds [τ_x, τ_y]
        """
        # Count reference baselines for logging
        if self.verbose:
            n_ref_bl = np.sum((antenna1 == ref_ant) | (antenna2 == ref_ant))
            print(f"[ALAKAZAM] Chain initial guess: {n_ref_bl} ref baselines (JIT-accelerated)")

        # Call JIT-compiled chain solver
        delays_init = _chain_delays_jit(
            vis_obs, vis_model, antenna1, antenna2, freq, flags, ref_ant, n_ant
        )

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
        Compute residuals for K delay optimization.

        Apply K delays to model and compute complex residuals.
        """
        # Unpack delays
        delays = self.unpack_params(params, n_ant, ref_ant)

        # Compute residuals
        residuals = _k_delay_residual(
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
            Delays [τ_x, τ_y] in nanoseconds

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
        """Print K delay solution."""
        if not self.verbose:
            return

        delays = jones  # (n_working, 2)

        print(f"[ALAKAZAM] Cost: {convergence_info['cost_init']:.4e} -> {convergence_info['cost_final']:.4e}")
        print(f"[ALAKAZAM] Final delays (ns):")

        for i in range(len(working_ants)):
            ant_full = working_ants[i]
            status = "[ref]" if ant_full == ref_ant else ""
            print(f"        Ant {ant_full:2d}: X={delays[i, 0]:+8.3f}, Y={delays[i, 1]:+8.3f}  {status}")


@njit(cache=True)
def _k_delay_residual(
    delays: np.ndarray,
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    freq: np.ndarray,
    flags: np.ndarray
) -> np.ndarray:
    """
    Compute residuals for K delay solver.

    Numba-compiled for speed.

    Parameters
    ----------
    delays : ndarray (n_ant, 2)
        Delays [τ_x, τ_y] in nanoseconds
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
        Flattened real residuals for unflagged data
    """
    n_bl = len(antenna1)
    n_freq = len(freq)

    # Count unflagged data points
    n_unflagged = 0
    for bl in range(n_bl):
        for ch in range(n_freq):
            if not flags[bl, ch, 0, 0]:
                n_unflagged += 1
            if not flags[bl, ch, 1, 1]:
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

            # Compute Jones matrices
            # K = exp(-2πi × τ × ν)
            phase_x1 = -2.0 * np.pi * delays[a1, 0] * f * 1e-9
            phase_y1 = -2.0 * np.pi * delays[a1, 1] * f * 1e-9
            phase_x2 = -2.0 * np.pi * delays[a2, 0] * f * 1e-9
            phase_y2 = -2.0 * np.pi * delays[a2, 1] * f * 1e-9

            K1_x = np.exp(1j * phase_x1)
            K1_y = np.exp(1j * phase_y1)
            K2_x = np.exp(1j * phase_x2)
            K2_y = np.exp(1j * phase_y2)

            # Apply K to model: V' = K1 × M × K2^H
            # XX correlation
            if not flags[bl, ch, 0, 0]:
                M_xx = vis_model[bl, ch, 0, 0]
                V_predicted = K1_x * M_xx * np.conj(K2_x)
                V_observed = vis_obs[bl, ch, 0, 0]

                residuals[res_idx] = (V_observed - V_predicted).real
                residuals[res_idx + 1] = (V_observed - V_predicted).imag
                res_idx += 2

            # YY correlation
            if not flags[bl, ch, 1, 1]:
                M_yy = vis_model[bl, ch, 1, 1]
                V_predicted = K1_y * M_yy * np.conj(K2_y)
                V_observed = vis_obs[bl, ch, 1, 1]

                residuals[res_idx] = (V_observed - V_predicted).real
                residuals[res_idx + 1] = (V_observed - V_predicted).imag
                res_idx += 2

    return residuals


__all__ = ['KDelaySolver']
