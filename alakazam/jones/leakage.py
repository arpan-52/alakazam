"""
D Leakage Jones Solver.

Solves for off-diagonal leakage terms.

Feed Basis Support:
- LINEAR basis: J = [[1, d_XY], [d_YX, 1]]  (X→Y and Y→X leakage)
- CIRCULAR basis: J = [[1, d_RL], [d_LR, 1]]  (R→L and L→R leakage)

Uses all 4 correlations in the residual.
The solver operates natively in the specified feed basis.
"""

import numpy as np
from numba import njit, prange
from scipy.optimize import least_squares
from typing import Dict, Optional
import logging

from .base import JonesEffect, JonesMetadata, JonesTypeEnum, SolveResult
from ..core.initialization import compute_initial_jones_chain

logger = logging.getLogger('jackal')


# =============================================================================
# Feed basis handling
# =============================================================================

class FeedBasis:
    """Feed basis enumeration."""
    LINEAR = "linear"     # XX, XY, YX, YY
    CIRCULAR = "circular" # RR, RL, LR, LL


@njit(cache=True, inline='always')
def _mm_2x2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Multiply 2x2 matrices."""
    return np.array([
        [A[0,0]*B[0,0] + A[0,1]*B[1,0], A[0,0]*B[0,1] + A[0,1]*B[1,1]],
        [A[1,0]*B[0,0] + A[1,1]*B[1,0], A[1,0]*B[0,1] + A[1,1]*B[1,1]]
    ], dtype=A.dtype)


@njit(cache=True, inline='always')
def _herm_2x2(M: np.ndarray) -> np.ndarray:
    """Hermitian transpose of 2x2 matrix."""
    return np.array([
        [np.conj(M[0,0]), np.conj(M[1,0])],
        [np.conj(M[0,1]), np.conj(M[1,1])]
    ], dtype=M.dtype)


@njit(cache=True)
def _leakage_params_to_jones(params: np.ndarray, n_ant: int, ref_ant: int,
                             d_constraint: str, working_ants: np.ndarray) -> np.ndarray:
    """
    Convert parameters to leakage Jones.

    d_constraint: 'XY', 'YX', or 'both'
      - 'XY': fix d_XY=0 for ref, solve d_YX (2 params per ant)
      - 'YX': fix d_YX=0 for ref, solve d_XY (2 params per ant)
      - 'both': fix both d_XY=0 and d_YX=0 for ref (4 params per ant)

    J = [[1, d_XY], [d_YX, 1]]
    """
    jones = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    idx = 0

    for i in range(len(working_ants)):
        a = working_ants[i]
        jones[a, 0, 0] = 1.0
        jones[a, 1, 1] = 1.0

        if a != ref_ant:
            if d_constraint == 'both':
                # Standard: 4 params [Re(d_XY), Im(d_XY), Re(d_YX), Im(d_YX)]
                jones[a, 0, 1] = params[idx] + 1j * params[idx + 1]
                jones[a, 1, 0] = params[idx + 2] + 1j * params[idx + 3]
                idx += 4
            elif d_constraint == 'XY':
                # Fix d_XY=0, solve d_YX: 2 params
                jones[a, 0, 1] = 0.0
                jones[a, 1, 0] = params[idx] + 1j * params[idx + 1]
                idx += 2
            elif d_constraint == 'YX':
                # Fix d_YX=0, solve d_XY: 2 params
                jones[a, 0, 1] = params[idx] + 1j * params[idx + 1]
                jones[a, 1, 0] = 0.0
                idx += 2

    return jones


@njit(cache=True)
def _leakage_jones_to_params(jones: np.ndarray, ref_ant: int, d_constraint: str,
                             working_ants: np.ndarray) -> np.ndarray:
    """Convert leakage Jones to parameters."""
    n_params = 0
    for a in working_ants:
        if a != ref_ant:
            n_params += 4 if d_constraint == 'both' else 2

    params = np.zeros(n_params, dtype=np.float64)
    idx = 0

    for i in range(len(working_ants)):
        a = working_ants[i]
        if a != ref_ant:
            if d_constraint == 'both':
                params[idx] = jones[a, 0, 1].real
                params[idx + 1] = jones[a, 0, 1].imag
                params[idx + 2] = jones[a, 1, 0].real
                params[idx + 3] = jones[a, 1, 0].imag
                idx += 4
            elif d_constraint == 'XY':
                params[idx] = jones[a, 1, 0].real
                params[idx + 1] = jones[a, 1, 0].imag
                idx += 2
            elif d_constraint == 'YX':
                params[idx] = jones[a, 0, 1].real
                params[idx + 1] = jones[a, 0, 1].imag
                idx += 2

    return params


@njit(parallel=True, cache=True)
def _leakage_residual(params: np.ndarray, vis_obs: np.ndarray, vis_model: np.ndarray,
                     antenna1: np.ndarray, antenna2: np.ndarray, n_ant: int, ref_ant: int,
                     d_constraint: str, working_ants: np.ndarray) -> np.ndarray:
    """
    Leakage residual - uses all 4 correlations.

    Returns: (n_bl * 8,) [Re(XX), Im(XX), Re(XY), Im(XY), Re(YX), Im(YX), Re(YY), Im(YY)]
    """
    n_bl = len(antenna1)
    jones = _leakage_params_to_jones(params, n_ant, ref_ant, d_constraint, working_ants)

    out = np.zeros(n_bl * 8, dtype=np.float64)
    for bl in prange(n_bl):
        Ji = jones[antenna1[bl]]
        Jj_H = _herm_2x2(jones[antenna2[bl]])
        V_pred = _mm_2x2(_mm_2x2(Ji, vis_model[bl]), Jj_H)
        R = vis_obs[bl] - V_pred

        base = bl * 8
        out[base+0] = R[0,0].real
        out[base+1] = R[0,0].imag
        out[base+2] = R[0,1].real
        out[base+3] = R[0,1].imag
        out[base+4] = R[1,0].real
        out[base+5] = R[1,0].imag
        out[base+6] = R[1,1].real
        out[base+7] = R[1,1].imag

    return out


class DLeakage(JonesEffect):
    """
    D Leakage Jones effect.

    Off-diagonal terms: J = [[1, d_XY], [d_YX, 1]]
    """

    def _create_metadata(self) -> JonesMetadata:
        return JonesMetadata(
            jones_type=JonesTypeEnum.D,
            description="Polarization leakage (off-diagonal). Supports LINEAR (d_XY, d_YX) and CIRCULAR (d_RL, d_LR) bases.",
            is_diagonal=False,
            is_frequency_dependent=False,
            is_time_dependent=False,
            needs_freq_axis=False,
            native_param_names=['d_01', 'd_10'],
            constraints={'ref_ant': 'one leakage term fixed to 0', 'd_constraint': 'XY'}  # Default
        )

    def solve(
        self,
        vis_obs: np.ndarray,
        vis_model: np.ndarray,
        antenna1: np.ndarray,
        antenna2: np.ndarray,
        n_ant: int,
        working_ants: np.ndarray,
        freq: Optional[np.ndarray] = None,
        flags: Optional[np.ndarray] = None,
        ref_ant: int = 0,
        max_iter: int = 100,
        tol: float = 1e-10,
        d_constraint: str = 'XY',
        feed_basis: str = FeedBasis.LINEAR,
        **kwargs
    ) -> SolveResult:
        """
        Solve for D leakage.

        Parameters
        ----------
        d_constraint : str
            'XY': fix d_XY=0 (or d_RL=0) for ref (recommended)
            'YX': fix d_YX=0 (or d_LR=0) for ref
            'both': fix both (over-constrained, legacy)
        feed_basis : str
            'linear': solve for d_XY, d_YX
            'circular': solve for d_RL, d_LR
        """
        self.validate_inputs(vis_obs, vis_model, antenna1, antenna2, n_ant, working_ants, freq)

        if vis_obs.ndim != 3 or vis_obs.shape[-2:] != (2, 2):
            raise ValueError("Expected vis_obs shape (n_bl, 2, 2)")

        n_bl = len(antenna1)

        logger.info(f"Solving D (leakage): {len(working_ants)} working antennas, ref_ant={ref_ant}, d_constraint={d_constraint}, feed_basis={feed_basis}")

        # Check ref antenna
        if ref_ant not in working_ants:
            logger.error(f"Reference antenna {ref_ant} is not working")
            return SolveResult(
                jones=np.full((n_ant, 2, 2), np.nan, dtype=np.complex128),
                native_params={'d_xy': np.full(n_ant, np.nan, dtype=np.complex128),
                              'd_yx': np.full(n_ant, np.nan, dtype=np.complex128)},
                cost_init=0.0,
                cost_final=0.0,
                nfev=0,
                success=False,
                convergence_info={'error': 'ref_ant not working'},
                flagging_stats={}
            )

        # Make contiguous
        vis_obs = np.ascontiguousarray(vis_obs, dtype=np.complex128)
        vis_model = np.ascontiguousarray(vis_model, dtype=np.complex128)
        antenna1 = np.ascontiguousarray(antenna1, dtype=np.int32)
        antenna2 = np.ascontiguousarray(antenna2, dtype=np.int32)
        working_ants = np.ascontiguousarray(working_ants, dtype=np.int32)

        # Filter baselines
        working_set = set(working_ants)
        valid_bl_mask = np.array([a1 in working_set and a2 in working_set
                                  for a1, a2 in zip(antenna1, antenna2)])

        vis_obs_filt = vis_obs[valid_bl_mask]
        vis_model_filt = vis_model[valid_bl_mask]
        ant1_filt = antenna1[valid_bl_mask]
        ant2_filt = antenna2[valid_bl_mask]

        logger.info(f"  Using {len(ant1_filt)} baselines between working antennas")

        if len(ant1_filt) == 0:
            logger.error("  No valid baselines")
            return SolveResult(
                jones=np.full((n_ant, 2, 2), np.nan, dtype=np.complex128),
                native_params={'d_xy': np.full(n_ant, np.nan, dtype=np.complex128),
                              'd_yx': np.full(n_ant, np.nan, dtype=np.complex128)},
                cost_init=0.0,
                cost_final=0.0,
                nfev=0,
                success=False,
                convergence_info={'error': 'no valid baselines'},
                flagging_stats={}
            )

        # Compute initial guess using direct chain method
        logger.info("  Computing initial guess from direct chain method...")
        jones_init = compute_initial_jones_chain(
            vis_obs_filt, vis_model_filt, ant1_filt, ant2_filt,
            n_ant, ref_ant=ref_ant, flags=None, verbose=False
        )

        # Set diagonal to 1 (leakage assumes diagonal is unity)
        jones_init[:, 0, 0] = 1.0
        jones_init[:, 1, 1] = 1.0

        # Extract params for optimizer
        p0 = _leakage_jones_to_params(jones_init, ref_ant, d_constraint, working_ants)

        n_params = len(p0)
        logger.info(f"  Initial guess from chain: {n_params} parameters")

        # Optimize
        def residual(p):
            return _leakage_residual(p, vis_obs_filt, vis_model_filt, ant1_filt, ant2_filt,
                                    n_ant, ref_ant, d_constraint, working_ants)

        cost_init = np.sum(residual(p0)**2)
        logger.info(f"  Initial cost: {cost_init:.6e}")
        logger.info("  Optimizing with Levenberg-Marquardt...")

        result = least_squares(residual, p0, method='lm', ftol=tol, xtol=tol, gtol=tol,
                              max_nfev=max_iter * len(p0) if len(p0) > 0 else 1, verbose=0)

        jones = _leakage_params_to_jones(result.x, n_ant, ref_ant, d_constraint, working_ants)

        # Set non-working to NaN
        for a in range(n_ant):
            if a not in working_ants:
                jones[a] = np.nan

        cost_final = result.cost * 2
        logger.info(f"  Final cost: {cost_final:.6e} (reduction: {cost_init/max(cost_final, 1e-20):.2f}x)")
        logger.info(f"  Converged: {result.success}, nfev: {result.nfev}")

        # Extract d-terms for native params
        d_01 = jones[:, 0, 1]
        d_10 = jones[:, 1, 0]

        pol0, pol1 = ('X', 'Y') if feed_basis == FeedBasis.LINEAR else ('R', 'L')
        logger.info(f"  D-term magnitudes: |d_{pol0}{pol1}| max={np.nanmax(np.abs(d_01[working_ants])):.4f}, |d_{pol1}{pol0}| max={np.nanmax(np.abs(d_10[working_ants])):.4f}")

        return SolveResult(
            jones=jones,
            native_params={'d_01': d_01, 'd_10': d_10, 'feed_basis': feed_basis},
            cost_init=cost_init,
            cost_final=cost_final,
            nfev=result.nfev,
            success=result.success,
            convergence_info={'ftol': tol, 'xtol': tol, 'gtol': tol, 'message': result.message, 'feed_basis': feed_basis},
            flagging_stats={}
        )

    def apply(self, jones: np.ndarray, vis: np.ndarray, antenna1: np.ndarray, antenna2: np.ndarray) -> np.ndarray:
        """Apply leakage Jones (full 2x2 multiply)."""
        n_bl = len(antenna1)
        out = np.zeros_like(vis)
        for bl in range(n_bl):
            Ji = jones[antenna1[bl]]
            Jj_H = _herm_2x2(jones[antenna2[bl]])
            out[bl] = _mm_2x2(_mm_2x2(Ji, vis[bl]), Jj_H)
        return out

    def unapply(self, jones: np.ndarray, vis: np.ndarray, antenna1: np.ndarray, antenna2: np.ndarray) -> np.ndarray:
        """Unapply leakage Jones."""
        # For leakage, unapply uses inverse
        jones_inv = self.inverse(jones)
        n_bl = len(antenna1)
        out = np.zeros_like(vis)
        for bl in range(n_bl):
            Ji_inv = jones_inv[antenna1[bl]]
            Jj_inv_H = _herm_2x2(jones_inv[antenna2[bl]])
            out[bl] = _mm_2x2(_mm_2x2(Ji_inv, vis[bl]), Jj_inv_H)
        return out

    def inverse(self, jones: np.ndarray) -> np.ndarray:
        """Invert leakage Jones (2x2 matrix inverse)."""
        n_ant = jones.shape[0]
        jones_inv = np.zeros_like(jones)
        for a in range(n_ant):
            det = jones[a, 0, 0] * jones[a, 1, 1] - jones[a, 0, 1] * jones[a, 1, 0]
            if np.abs(det) < 1e-15:
                jones_inv[a] = np.eye(2, dtype=np.complex128)
            else:
                jones_inv[a, 0, 0] = jones[a, 1, 1] / det
                jones_inv[a, 0, 1] = -jones[a, 0, 1] / det
                jones_inv[a, 1, 0] = -jones[a, 1, 0] / det
                jones_inv[a, 1, 1] = jones[a, 0, 0] / det
        return jones_inv

    def to_native_params(self, jones: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract d-terms."""
        return {
            'd_xy': jones[:, 0, 1],
            'd_yx': jones[:, 1, 0]
        }

    def from_native_params(self, params: Dict[str, np.ndarray], freq: Optional[np.ndarray] = None) -> np.ndarray:
        """Construct Jones from d-terms."""
        n_ant = len(params['d_xy'])
        jones = np.zeros((n_ant, 2, 2), dtype=np.complex128)
        jones[:, 0, 0] = 1.0
        jones[:, 1, 1] = 1.0
        jones[:, 0, 1] = params['d_xy']
        jones[:, 1, 0] = params['d_yx']
        return jones


__all__ = ['DLeakage']
