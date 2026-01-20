"""
Xf Crosshand Phase Solver - Frequency-independent crosshand phase.

Jones Matrix:
  Xf = [[1, exp(i φ_xy)],
        [exp(i φ_yx), 1]]

Reference Constraint:
  Xf_ref = I (identity, φ_ref = 0)

Parameters:
  [φ_xy, φ_yx] per antenna in radians

Averaging:
  - Time: YES (average over solint)
  - Freq: YES (frequency-independent phase)

Chain Initial Guess:
  Xf_j = V_ref,j / M_ref,j  (averaged over freq)
  φ_xy = angle(Xf_j[0,1])
  φ_yx = angle(Xf_j[1,0])
"""

import numpy as np
from scipy.optimize import least_squares
from numba import njit
from typing import Dict, Any

from .base import JonesSolverBase, SolverMetadata
from .utils import find_ref_baselines


class XfCrossphaseSolver(JonesSolverBase):
    """Xf crosshand phase solver."""

    metadata = SolverMetadata(
        jones_type='Xf',
        ref_constraint='phase_zero',
        can_avg_time=True,
        can_avg_freq=True,  # Frequency-independent
        description="Crosshand phase from XY/YX correlations"
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
        Chain solver for Xf crosshand phase.

        For Xf_ref = I:
          Xf_j = V_ref,j / M_ref,j
          φ_xy = angle(Xf_j[0,1])
          φ_yx = angle(Xf_j[1,0])

        Returns
        -------
        phases_init : ndarray (n_ant, 2)
            Initial phases in radians [φ_xy, φ_yx]
        """
        phases_init = np.zeros((n_ant, 2), dtype=np.float64)

        # Find baselines to reference
        ref_baselines = find_ref_baselines(antenna1, antenna2, ref_ant)

        if self.verbose:
            print(f"[ALAKAZAM] Chain initial guess: {len(ref_baselines)} ref baselines")

        # For each baseline to reference
        for bl_idx, ant_j, flip in ref_baselines:
            # For each crosshand correlation (XY, YX)
            for corr_idx, (p1, p2) in enumerate([(0, 1), (1, 0)]):
                V = vis_obs[bl_idx, p1, p2]
                M = vis_model[bl_idx, p1, p2]
                F = flags[bl_idx, p1, p2]

                # Skip if flagged or invalid
                if F or np.abs(M) < 1e-10 or not np.isfinite(V) or not np.isfinite(M):
                    continue

                # Chain solve: Xf_j = V / M (since Xf_ref = I)
                if flip:
                    # V = M × Xf_ref^H × Xf_j^H = M × Xf_j^H
                    # Xf_j = (V / M)^*
                    Xf_j = np.conj(V / M)
                else:
                    # V = Xf_j × M × Xf_ref^H = Xf_j × M
                    # Xf_j = V / M
                    Xf_j = V / M

                # Extract phase
                phase = np.angle(Xf_j)
                phases_init[ant_j, corr_idx] = phase

        return phases_init

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
        Compute residuals for Xf crossphase optimization.

        Apply Xf phases to model and compute complex residuals for XY/YX.
        """
        # Unpack phases
        phases = self.unpack_params(params, n_ant, ref_ant)

        # Compute residuals
        residuals = _xf_crossphase_residual(
            phases, vis_obs, vis_model, antenna1, antenna2, flags
        )

        return residuals

    def pack_params(
        self,
        jones: np.ndarray,
        ref_ant: int,
        **kwargs
    ) -> np.ndarray:
        """
        Pack phases to parameter array.

        jones : ndarray (n_ant, 2)
            Phases [φ_xy, φ_yx] in radians

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
        Unpack parameters to phases.

        Returns phases (n_ant, 2) with ref=0.
        """
        phases = np.zeros((n_ant, 2), dtype=np.float64)
        idx = 0

        for ant in range(n_ant):
            if ant != ref_ant:
                phases[ant, 0] = params[idx]
                phases[ant, 1] = params[idx + 1]
                idx += 2

        return phases

    def print_solution(
        self,
        jones: np.ndarray,
        working_ants: np.ndarray,
        ref_ant: int,
        convergence_info: Dict[str, Any]
    ):
        """Print Xf crossphase solution."""
        if not self.verbose:
            return

        phases = jones  # (n_working, 2)
        phases_deg = np.rad2deg(phases)

        print(f"[ALAKAZAM] Cost: {convergence_info['cost_init']:.4e} -> {convergence_info['cost_final']:.4e}")
        print(f"[ALAKAZAM] Final crosshand phases (deg):")

        for i in range(len(working_ants)):
            ant_full = working_ants[i]
            status = "[ref]" if ant_full == ref_ant else ""
            print(f"        Ant {ant_full:2d}: XY={phases_deg[i, 0]:+8.2f}, YX={phases_deg[i, 1]:+8.2f}  {status}")


@njit(cache=True)
def _xf_crossphase_residual(
    phases: np.ndarray,
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    flags: np.ndarray
) -> np.ndarray:
    """
    Compute residuals for Xf crossphase solver.

    Numba-compiled for speed.

    Parameters
    ----------
    phases : ndarray (n_ant, 2)
        Phases [φ_xy, φ_yx] in radians
    vis_obs, vis_model : ndarray (n_bl, 2, 2)
        Visibilities (time+freq averaged)
    antenna1, antenna2 : ndarray (n_bl,)
        Antenna indices
    flags : ndarray (n_bl, 2, 2)
        Flags (True = flagged)

    Returns
    -------
    residuals : ndarray
        Flattened real residuals for unflagged XY/YX data
    """
    n_bl = len(antenna1)

    # Count unflagged crosshand data points
    n_unflagged = 0
    for bl in range(n_bl):
        if not flags[bl, 0, 1]:  # XY
            n_unflagged += 1
        if not flags[bl, 1, 0]:  # YX
            n_unflagged += 1

    # Allocate residual array (2 values per unflagged correlation: real + imag)
    residuals = np.zeros(n_unflagged * 2, dtype=np.float64)
    res_idx = 0

    # For each baseline
    for bl in range(n_bl):
        a1 = antenna1[bl]
        a2 = antenna2[bl]

        # Compute Jones matrices for crosshand
        # Xf = [[1, exp(i φ_xy)],
        #       [exp(i φ_yx), 1]]

        Xf1_xy = np.exp(1j * phases[a1, 0])
        Xf1_yx = np.exp(1j * phases[a1, 1])
        Xf2_xy = np.exp(1j * phases[a2, 0])
        Xf2_yx = np.exp(1j * phases[a2, 1])

        # Apply Xf to model: V' = Xf1 × M × Xf2^H
        # For XY correlation: V'_xy = Xf1_xy × M_xy × Xf2_xy^*

        # XY correlation
        if not flags[bl, 0, 1]:
            M_xy = vis_model[bl, 0, 1]
            V_predicted = Xf1_xy * M_xy * np.conj(Xf2_xy)
            V_observed = vis_obs[bl, 0, 1]

            residuals[res_idx] = (V_observed - V_predicted).real
            residuals[res_idx + 1] = (V_observed - V_predicted).imag
            res_idx += 2

        # YX correlation
        if not flags[bl, 1, 0]:
            M_yx = vis_model[bl, 1, 0]
            V_predicted = Xf1_yx * M_yx * np.conj(Xf2_yx)
            V_observed = vis_obs[bl, 1, 0]

            residuals[res_idx] = (V_observed - V_predicted).real
            residuals[res_idx + 1] = (V_observed - V_predicted).imag
            res_idx += 2

    return residuals


__all__ = ['XfCrossphaseSolver']
