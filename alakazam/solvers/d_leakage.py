"""
D Leakage Solver - Polarization leakage terms.

Jones Matrix:
  D = [[1,      d_xy],
       [d_yx,   1   ]]

Reference Constraint:
  d_ref = 0 (no leakage for reference)

Parameters:
  - 'XY': [Re(d_xy), Im(d_xy)] per antenna
  - 'YX': [Re(d_yx), Im(d_yx)] per antenna
  - 'both': [Re(d_xy), Im(d_xy), Re(d_yx), Im(d_yx)]

Averaging:
  - Time: YES
  - Freq: YES

Chain Initial Guess:
  d_j = 0 (start from zero)
"""

import numpy as np
from scipy.optimize import least_squares
from numba import njit
from typing import Dict, Any

from .base import JonesSolverBase, SolverMetadata


class DLeakageSolver(JonesSolverBase):
    """D leakage solver."""

    metadata = SolverMetadata(
        jones_type='D',
        ref_constraint='zero',
        can_avg_time=True,
        can_avg_freq=True,
        description="Polarization leakage (off-diagonal terms)"
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
        Initial guess for D leakage.
        
        Start from zero (identity matrix).
        """
        # D = [[1, d_xy], [d_yx, 1]]
        d_init = np.zeros((n_ant, 2, 2), dtype=np.complex128)
        
        # Set diagonal to 1
        d_init[:, 0, 0] = 1.0
        d_init[:, 1, 1] = 1.0

        if self.verbose:
            print(f"[ALAKAZAM] Initial guess: D = I (identity)")

        return d_init

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
        """Compute residuals for D leakage."""
        d_constraint = kwargs.get('d_constraint', 'both')

        d_matrices = self.unpack_params(params, n_ant, ref_ant, d_constraint=d_constraint)

        residuals = _d_leakage_residual(
            d_matrices, vis_obs, vis_model, antenna1, antenna2, flags
        )

        return residuals

    def pack_params(
        self,
        jones: np.ndarray,
        ref_ant: int,
        **kwargs
    ) -> np.ndarray:
        """Pack D leakage to parameters."""
        d_constraint = kwargs.get('d_constraint', 'both')
        n_ant = jones.shape[0]
        params = []

        for ant in range(n_ant):
            if ant == ref_ant:
                continue  # Ref has d = 0

            d_xy = jones[ant, 0, 1]
            d_yx = jones[ant, 1, 0]

            if d_constraint == 'XY':
                params.extend([d_xy.real, d_xy.imag])
            elif d_constraint == 'YX':
                params.extend([d_yx.real, d_yx.imag])
            else:  # both
                params.extend([d_xy.real, d_xy.imag, d_yx.real, d_yx.imag])

        return np.array(params, dtype=np.float64)

    def unpack_params(
        self,
        params: np.ndarray,
        n_ant: int,
        ref_ant: int,
        **kwargs
    ) -> np.ndarray:
        """Unpack parameters to D matrices."""
        d_constraint = kwargs.get('d_constraint', 'both')
        
        d_matrices = np.zeros((n_ant, 2, 2), dtype=np.complex128)
        d_matrices[:, 0, 0] = 1.0
        d_matrices[:, 1, 1] = 1.0
        
        idx = 0
        for ant in range(n_ant):
            if ant == ref_ant:
                continue

            if d_constraint == 'XY':
                d_matrices[ant, 0, 1] = params[idx] + 1j * params[idx + 1]
                idx += 2
            elif d_constraint == 'YX':
                d_matrices[ant, 1, 0] = params[idx] + 1j * params[idx + 1]
                idx += 2
            else:  # both
                d_matrices[ant, 0, 1] = params[idx] + 1j * params[idx + 1]
                d_matrices[ant, 1, 0] = params[idx + 2] + 1j * params[idx + 3]
                idx += 4

        return d_matrices

    def print_solution(
        self,
        jones: np.ndarray,
        working_ants: np.ndarray,
        ref_ant: int,
        convergence_info: Dict[str, Any]
    ):
        """Print D leakage solution."""
        if not self.verbose:
            return

        print(f"[ALAKAZAM] Cost: {convergence_info['cost_init']:.4e} -> {convergence_info['cost_final']:.4e}")
        print(f"[ALAKAZAM] Leakage terms:")

        for i in range(len(working_ants)):
            ant_full = working_ants[i]
            d_xy = jones[i, 0, 1]
            d_yx = jones[i, 1, 0]

            status = "[ref]" if ant_full == ref_ant else ""
            print(f"        Ant {ant_full:2d}: d_XY={abs(d_xy):.4f}, d_YX={abs(d_yx):.4f}  {status}")


@njit(cache=True)
def _d_leakage_residual(
    d_matrices: np.ndarray,
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    flags: np.ndarray
) -> np.ndarray:
    """Compute residuals for D leakage."""
    n_bl = len(antenna1)

    n_unflagged = 0
    for bl in range(n_bl):
        for i in range(2):
            for j in range(2):
                if not flags[bl, i, j]:
                    n_unflagged += 1

    residuals = np.zeros(n_unflagged * 2, dtype=np.float64)
    res_idx = 0

    for bl in range(n_bl):
        a1 = antenna1[bl]
        a2 = antenna2[bl]

        # D matrices
        D1 = d_matrices[a1]
        D2 = d_matrices[a2]

        # Apply: V' = D1 × M × D2^H
        for i in range(2):
            for j in range(2):
                if not flags[bl, i, j]:
                    M = vis_model[bl]
                    
                    # Matrix multiplication: (D1 @ M @ D2.conj().T)[i,j]
                    V_predicted = 0.0 + 0.0j
                    for k in range(2):
                        for l in range(2):
                            V_predicted += D1[i, k] * M[k, l] * np.conj(D2[j, l])

                    V_observed = vis_obs[bl, i, j]

                    residuals[res_idx] = (V_observed - V_predicted).real
                    residuals[res_idx + 1] = (V_observed - V_predicted).imag
                    res_idx += 2

    return residuals


__all__ = ['DLeakageSolver']
