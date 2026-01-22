"""
Gain and Bandpass Jones Solvers (G and B).

Both solve for diagonal Jones matrices with amplitude and phase.

Feed Basis Support:
- LINEAR basis: J = diag(g_X, g_Y) where g = amp * exp(iφ)
- CIRCULAR basis: J = diag(g_R, g_L) where g = amp * exp(iφ)

G: Time-dependent, frequency-averaged
B: Frequency-dependent, can have time dependence

The solver operates natively in the specified feed basis.
"""

import numpy as np
from numba import njit, prange
from scipy.optimize import least_squares
from typing import Dict, Optional
import logging

from .base import JonesEffect, JonesMetadata, JonesTypeEnum, SolveResult
from ..core.initialization import compute_initial_jones_chain, normalize_jones_to_reference

logger = logging.getLogger('jackal')


# =============================================================================
# Feed basis handling
# =============================================================================

class FeedBasis:
    """Feed basis enumeration."""
    LINEAR = "linear"     # XX, XY, YX, YY
    CIRCULAR = "circular" # RR, RL, LR, LL


@njit(cache=True)
def _diag_params_to_jones(params: np.ndarray, n_ant: int, ref_ant: int, working_ants: np.ndarray, phase_only: bool) -> np.ndarray:
    """
    Convert parameters to diagonal Jones.

    phase_only=False: ref gets [amp_X, amp_Y], others get [amp_X, phase_X, amp_Y, phase_Y]
    phase_only=True: ref gets identity, others get [phase_X, phase_Y]
    """
    jones = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    idx = 0

    for i in range(len(working_ants)):
        a = working_ants[i]
        if a == ref_ant:
            if phase_only:
                jones[a, 0, 0] = 1.0
                jones[a, 1, 1] = 1.0
            else:
                jones[a, 0, 0] = params[idx]
                jones[a, 1, 1] = params[idx + 1]
                idx += 2
        else:
            if phase_only:
                jones[a, 0, 0] = np.exp(1j * params[idx])
                jones[a, 1, 1] = np.exp(1j * params[idx + 1])
                idx += 2
            else:
                jones[a, 0, 0] = params[idx] * np.exp(1j * params[idx + 1])
                jones[a, 1, 1] = params[idx + 2] * np.exp(1j * params[idx + 3])
                idx += 4

    return jones


@njit(cache=True)
def _diag_jones_to_params(jones: np.ndarray, ref_ant: int, working_ants: np.ndarray, phase_only: bool) -> np.ndarray:
    """Convert diagonal Jones to parameters."""
    n_working = len(working_ants)
    n_params = 0

    if phase_only:
        n_params = (n_working - 1) * 2  # ref has no params
    else:
        n_params = 2  # ref gets 2 amps
        for a in working_ants:
            if a != ref_ant:
                n_params += 4  # amp_X, phase_X, amp_Y, phase_Y

    params = np.zeros(n_params, dtype=np.float64)
    idx = 0

    for i in range(n_working):
        a = working_ants[i]
        if a == ref_ant:
            if not phase_only:
                params[idx] = np.abs(jones[a, 0, 0])
                params[idx + 1] = np.abs(jones[a, 1, 1])
                idx += 2
        else:
            if phase_only:
                params[idx] = np.angle(jones[a, 0, 0])
                params[idx + 1] = np.angle(jones[a, 1, 1])
                idx += 2
            else:
                params[idx] = np.abs(jones[a, 0, 0])
                params[idx + 1] = np.angle(jones[a, 0, 0])
                params[idx + 2] = np.abs(jones[a, 1, 1])
                params[idx + 3] = np.angle(jones[a, 1, 1])
                idx += 4

    return params


@njit(parallel=True, cache=True)
def _diag_residual(params: np.ndarray, vis_obs: np.ndarray, vis_model: np.ndarray,
                   antenna1: np.ndarray, antenna2: np.ndarray, n_ant: int, ref_ant: int,
                   working_ants: np.ndarray, phase_only: bool) -> np.ndarray:
    """Compute residuals for diagonal Jones (XX and YY only)."""
    n_bl = len(antenna1)
    jones = _diag_params_to_jones(params, n_ant, ref_ant, working_ants, phase_only)

    out = np.zeros(n_bl * 4, dtype=np.float64)
    for bl in prange(n_bl):
        a1, a2 = antenna1[bl], antenna2[bl]

        V_xx = jones[a1, 0, 0] * vis_model[bl, 0, 0] * np.conj(jones[a2, 0, 0])
        V_yy = jones[a1, 1, 1] * vis_model[bl, 1, 1] * np.conj(jones[a2, 1, 1])

        R_xx = vis_obs[bl, 0, 0] - V_xx
        R_yy = vis_obs[bl, 1, 1] - V_yy

        base = bl * 4
        out[base+0] = R_xx.real
        out[base+1] = R_xx.imag
        out[base+2] = R_yy.real
        out[base+3] = R_yy.imag

    return out


@njit(cache=True)
def _chain_solve_diagonal(vis_obs: np.ndarray, vis_model: np.ndarray,
                         antenna1: np.ndarray, antenna2: np.ndarray,
                         n_ant: int, ref_ant: int) -> np.ndarray:
    """Chain solver for diagonal gains."""
    n_bl = len(antenna1)
    gains = np.ones((n_ant, 2), dtype=np.complex128)
    solved = np.zeros(n_ant, dtype=np.bool_)

    # Ref amplitude from median
    ref_amps_x = []
    ref_amps_y = []
    for bl in range(n_bl):
        a1, a2 = antenna1[bl], antenna2[bl]
        if a1 == ref_ant or a2 == ref_ant:
            M_xx, M_yy = vis_model[bl, 0, 0], vis_model[bl, 1, 1]
            if np.abs(M_xx) > 1e-10:
                ref_amps_x.append(np.sqrt(np.abs(vis_obs[bl, 0, 0] / M_xx)))
            if np.abs(M_yy) > 1e-10:
                ref_amps_y.append(np.sqrt(np.abs(vis_obs[bl, 1, 1] / M_yy)))

    if len(ref_amps_x) > 0:
        gains[ref_ant, 0] = np.median(np.array(ref_amps_x))
        gains[ref_ant, 1] = np.median(np.array(ref_amps_y))
    solved[ref_ant] = True

    # Propagate
    for _ in range(n_ant):
        for bl in range(n_bl):
            a1, a2 = antenna1[bl], antenna2[bl]

            if solved[a1] and not solved[a2]:
                for pol in range(2):
                    M, V = vis_model[bl, pol, pol], vis_obs[bl, pol, pol]
                    g_known = gains[a1, pol]
                    if np.abs(M) > 1e-10 and np.abs(g_known) > 1e-10:
                        gains[a2, pol] = np.conj(V / (g_known * M))
                solved[a2] = True

            elif solved[a2] and not solved[a1]:
                for pol in range(2):
                    M, V = vis_model[bl, pol, pol], vis_obs[bl, pol, pol]
                    g_known = gains[a2, pol]
                    if np.abs(M) > 1e-10 and np.abs(g_known) > 1e-10:
                        gains[a1, pol] = V / (M * np.conj(g_known))
                solved[a1] = True

    return gains


class GainDiagonal(JonesEffect):
    """Base class for diagonal gain solvers (G and B)."""

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
        phase_only: bool = False,
        feed_basis: str = FeedBasis.LINEAR,
        **kwargs
    ) -> SolveResult:
        """
        Solve for diagonal Jones (G or B).

        feed_basis: 'linear' (g_X, g_Y) or 'circular' (g_R, g_L)
        """
        self.validate_inputs(vis_obs, vis_model, antenna1, antenna2, n_ant, working_ants, freq)

        if vis_obs.ndim != 3 or vis_obs.shape[-2:] != (2, 2):
            raise ValueError("Expected vis_obs shape (n_bl, 2, 2)")

        n_bl = len(antenna1)
        mode = "phase-only" if phase_only else "amp+phase"

        logger.info(f"Solving {self.jones_type} ({mode}): {len(working_ants)} working antennas, ref_ant={ref_ant}, feed_basis={feed_basis}")

        # Check ref antenna
        if ref_ant not in working_ants:
            logger.error(f"Reference antenna {ref_ant} is not working")
            return SolveResult(
                jones=np.full((n_ant, 2, 2), np.nan, dtype=np.complex128),
                native_params={},
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
                native_params={},
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

        # Normalize to reference antenna constraint
        jones_init = normalize_jones_to_reference(jones_init, ref_ant, phase_only=phase_only)

        # Extract params for optimizer
        p0 = _diag_jones_to_params(jones_init, ref_ant, working_ants, phase_only)

        # Optimize
        def residual(p):
            return _diag_residual(p, vis_obs_filt, vis_model_filt, ant1_filt, ant2_filt,
                                 n_ant, ref_ant, working_ants, phase_only)

        cost_init = np.sum(residual(p0)**2)
        logger.info(f"  Initial cost: {cost_init:.6e}")
        logger.info("  Optimizing with Levenberg-Marquardt...")

        result = least_squares(residual, p0, method='lm', ftol=tol, xtol=tol, gtol=tol,
                              max_nfev=max_iter * len(p0) if len(p0) > 0 else 1, verbose=0)

        jones = _diag_params_to_jones(result.x, n_ant, ref_ant, working_ants, phase_only)

        # Set non-working to NaN
        for a in range(n_ant):
            if a not in working_ants:
                jones[a] = np.nan

        cost_final = result.cost * 2
        logger.info(f"  Final cost: {cost_final:.6e} (reduction: {cost_init/max(cost_final, 1e-20):.2f}x)")
        logger.info(f"  Converged: {result.success}, nfev: {result.nfev}")

        return SolveResult(
            jones=jones,
            native_params={'gains': jones, 'feed_basis': feed_basis},
            cost_init=cost_init,
            cost_final=cost_final,
            nfev=result.nfev,
            success=result.success,
            convergence_info={'ftol': tol, 'xtol': tol, 'gtol': tol, 'message': result.message, 'feed_basis': feed_basis},
            flagging_stats={}
        )

    def apply(self, jones: np.ndarray, vis: np.ndarray, antenna1: np.ndarray, antenna2: np.ndarray) -> np.ndarray:
        """Apply diagonal Jones (simple element-wise for diagonal)."""
        n_bl = len(antenna1)
        out = vis.copy()
        for bl in range(n_bl):
            a1, a2 = antenna1[bl], antenna2[bl]
            out[bl, 0, 0] = jones[a1, 0, 0] * vis[bl, 0, 0] * np.conj(jones[a2, 0, 0])
            out[bl, 1, 1] = jones[a1, 1, 1] * vis[bl, 1, 1] * np.conj(jones[a2, 1, 1])
        return out

    def unapply(self, jones: np.ndarray, vis: np.ndarray, antenna1: np.ndarray, antenna2: np.ndarray) -> np.ndarray:
        """Unapply diagonal Jones."""
        n_bl = len(antenna1)
        out = vis.copy()
        for bl in range(n_bl):
            a1, a2 = antenna1[bl], antenna2[bl]
            g1_x, g2_x = jones[a1, 0, 0], jones[a2, 0, 0]
            g1_y, g2_y = jones[a1, 1, 1], jones[a2, 1, 1]
            if np.abs(g1_x) > 1e-15 and np.abs(g2_x) > 1e-15:
                out[bl, 0, 0] = vis[bl, 0, 0] / (g1_x * np.conj(g2_x))
            if np.abs(g1_y) > 1e-15 and np.abs(g2_y) > 1e-15:
                out[bl, 1, 1] = vis[bl, 1, 1] / (g1_y * np.conj(g2_y))
        return out

    def inverse(self, jones: np.ndarray) -> np.ndarray:
        """Invert diagonal Jones."""
        jones_inv = np.zeros_like(jones)
        jones_inv[:, 0, 0] = 1.0 / jones[:, 0, 0]
        jones_inv[:, 1, 1] = 1.0 / jones[:, 1, 1]
        return jones_inv

    def to_native_params(self, jones: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract amplitude and phase."""
        amp_x = np.abs(jones[:, 0, 0])
        amp_y = np.abs(jones[:, 1, 1])
        phase_x = np.angle(jones[:, 0, 0])
        phase_y = np.angle(jones[:, 1, 1])
        return {'amp_X': amp_x, 'amp_Y': amp_y, 'phase_X': phase_x, 'phase_Y': phase_y}

    def from_native_params(self, params: Dict[str, np.ndarray], freq: Optional[np.ndarray] = None) -> np.ndarray:
        """Construct Jones from amp/phase."""
        n_ant = len(params['amp_X'])
        jones = np.zeros((n_ant, 2, 2), dtype=np.complex128)
        jones[:, 0, 0] = params['amp_X'] * np.exp(1j * params['phase_X'])
        jones[:, 1, 1] = params['amp_Y'] * np.exp(1j * params['phase_Y'])
        return jones


class GGain(GainDiagonal):
    """G Gain solver (time-dependent, frequency-averaged)."""

    def _create_metadata(self) -> JonesMetadata:
        return JonesMetadata(
            jones_type=JonesTypeEnum.G,
            description="Complex gain (diagonal, time-dependent). Supports LINEAR (g_X, g_Y) and CIRCULAR (g_R, g_L) bases.",
            is_diagonal=True,
            is_frequency_dependent=False,
            is_time_dependent=True,
            needs_freq_axis=False,
            native_param_names=['amp_pol0', 'amp_pol1', 'phase_pol0', 'phase_pol1'],
            constraints={'ref_ant': 'phase = 0 for both pols, amplitude optimized'}
        )


class BBandpass(GainDiagonal):
    """B Bandpass solver (frequency-dependent)."""

    def _create_metadata(self) -> JonesMetadata:
        return JonesMetadata(
            jones_type=JonesTypeEnum.B,
            description="Bandpass (diagonal, frequency-dependent). Supports LINEAR (g_X, g_Y) and CIRCULAR (g_R, g_L) bases.",
            is_diagonal=True,
            is_frequency_dependent=True,
            is_time_dependent=False,
            needs_freq_axis=False,
            native_param_names=['amp_pol0', 'amp_pol1', 'phase_pol0', 'phase_pol1'],
            constraints={'ref_ant': 'phase = 0 for both pols, amplitude optimized'}
        )


__all__ = ['GGain', 'BBandpass']
