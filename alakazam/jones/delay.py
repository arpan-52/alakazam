"""
K Delay Jones Solver.

Solves for antenna-based delays by fitting phase slopes across frequency.

Feed Basis Support:
- LINEAR basis: J = diag(exp(-2πi τ_X ν), exp(-2πi τ_Y ν))
- CIRCULAR basis: J = diag(exp(-2πi τ_R ν), exp(-2πi τ_L ν))

The solver operates natively in the specified feed basis.
"""

import numpy as np
from numba import njit, prange
from scipy.optimize import least_squares
from typing import Dict, Tuple, Optional
from collections import deque
import logging

from .base import JonesEffect, JonesMetadata, JonesTypeEnum, SolveResult

logger = logging.getLogger('jackal')


# =============================================================================
# Feed basis handling
# =============================================================================

class FeedBasis:
    """Feed basis enumeration."""
    LINEAR = "linear"     # XX, XY, YX, YY
    CIRCULAR = "circular" # RR, RL, LR, LL


def convert_linear_to_circular(jones_linear: np.ndarray) -> np.ndarray:
    """
    Convert Jones from linear (XX, YY) to circular (RR, LL) basis.

    Transformation: J_circ = P^{-1} @ J_lin @ P
    where P = (1/sqrt(2)) [[1, 1], [i, -i]]
    """
    sqrt2 = np.sqrt(2.0)
    P = np.array([[1.0, 1.0], [1j, -1j]], dtype=np.complex128) / sqrt2
    P_inv = np.array([[1.0, -1j], [1.0, 1j]], dtype=np.complex128) / sqrt2

    # Handle both (n_ant, 2, 2) and (n_freq, n_ant, 2, 2) shapes
    if jones_linear.ndim == 3:
        n_ant = jones_linear.shape[0]
        jones_circ = np.zeros_like(jones_linear)
        for a in range(n_ant):
            jones_circ[a] = P_inv @ jones_linear[a] @ P
    elif jones_linear.ndim == 4:
        n_freq, n_ant = jones_linear.shape[:2]
        jones_circ = np.zeros_like(jones_linear)
        for f in range(n_freq):
            for a in range(n_ant):
                jones_circ[f, a] = P_inv @ jones_linear[f, a] @ P
    else:
        raise ValueError(f"Unexpected jones shape: {jones_linear.shape}")

    return jones_circ


def convert_circular_to_linear(jones_circ: np.ndarray) -> np.ndarray:
    """Convert Jones from circular (RR, LL) to linear (XX, YY) basis."""
    sqrt2 = np.sqrt(2.0)
    P = np.array([[1.0, 1.0], [1j, -1j]], dtype=np.complex128) / sqrt2
    P_inv = np.array([[1.0, -1j], [1.0, 1j]], dtype=np.complex128) / sqrt2

    # Handle both (n_ant, 2, 2) and (n_freq, n_ant, 2, 2) shapes
    if jones_circ.ndim == 3:
        n_ant = jones_circ.shape[0]
        jones_lin = np.zeros_like(jones_circ)
        for a in range(n_ant):
            jones_lin[a] = P @ jones_circ[a] @ P_inv
    elif jones_circ.ndim == 4:
        n_freq, n_ant = jones_circ.shape[:2]
        jones_lin = np.zeros_like(jones_circ)
        for f in range(n_freq):
            for a in range(n_ant):
                jones_lin[f, a] = P @ jones_circ[f, a] @ P_inv
    else:
        raise ValueError(f"Unexpected jones shape: {jones_circ.shape}")

    return jones_lin


# =============================================================================
# Numba JIT Functions
# =============================================================================

@njit(cache=True)
def _fit_delay_baseline(vis_ratio: np.ndarray, freq: np.ndarray) -> float:
    """
    Fit delay from phase vs frequency for single baseline.

    vis_ratio: V_obs / M, shape (n_freq,) complex
    freq: (n_freq,) in Hz

    Returns delay in nanoseconds.
    """
    n = len(freq)
    phase = np.angle(vis_ratio)

    # Unwrap phase
    unwrapped = np.zeros(n, dtype=np.float64)
    unwrapped[0] = phase[0]
    for i in range(1, n):
        d = phase[i] - phase[i-1]
        if d > np.pi:
            d -= 2*np.pi
        elif d < -np.pi:
            d += 2*np.pi
        unwrapped[i] = unwrapped[i-1] + d

    # Linear fit: phase = -2π * τ * ν
    f_mean = np.mean(freq)
    p_mean = np.mean(unwrapped)
    num = 0.0
    den = 0.0
    for i in range(n):
        df = freq[i] - f_mean
        dp = unwrapped[i] - p_mean
        num += df * dp
        den += df * df

    if den < 1e-20:
        return 0.0

    slope = num / den
    return -slope / (2.0 * np.pi) * 1e9  # Convert to nanoseconds


@njit(parallel=True, cache=True)
def _delay_to_jones_single(delay: np.ndarray, freq: np.ndarray, n_ant: int) -> np.ndarray:
    """
    Convert delays to Jones matrices for a single frequency set.

    delay: (n_ant, 2) in nanoseconds [τ_X, τ_Y]
    freq: (n_freq,) in Hz

    Returns: (n_freq, n_ant, 2, 2) complex128
    """
    n_freq = len(freq)
    jones = np.zeros((n_freq, n_ant, 2, 2), dtype=np.complex128)

    for f in prange(n_freq):
        for a in range(n_ant):
            phase_x = -2.0 * np.pi * delay[a, 0] * 1e-9 * freq[f]
            phase_y = -2.0 * np.pi * delay[a, 1] * 1e-9 * freq[f]
            jones[f, a, 0, 0] = np.exp(1j * phase_x)
            jones[f, a, 1, 1] = np.exp(1j * phase_y)

    return jones


@njit(parallel=True, cache=True)
def _delay_residual(
    params: np.ndarray,
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    freq: np.ndarray,
    n_ant: int,
    ref_ant: int,
    working_ants: np.ndarray
) -> np.ndarray:
    """
    Compute residuals for delay optimization (diagonal only: XX and YY).

    params: Delays for working antennas except ref, [(τ_X, τ_Y), ...] flattened
    vis_obs, vis_model: (n_bl, n_freq, 2, 2)
    working_ants: Array of working antenna indices

    Returns: Residuals (n_bl * n_freq * 4,) [Re(XX), Im(XX), Re(YY), Im(YY)]
    """
    n_bl = len(antenna1)
    n_freq = len(freq)

    # Unpack delays - only for working antennas
    delay = np.zeros((n_ant, 2), dtype=np.float64)
    idx = 0
    for i in range(len(working_ants)):
        a = working_ants[i]
        if a != ref_ant:
            delay[a, 0] = params[idx]
            delay[a, 1] = params[idx + 1]
            idx += 2

    # Compute residuals
    out = np.zeros(n_bl * n_freq * 4, dtype=np.float64)

    for bl in prange(n_bl):
        a1, a2 = antenna1[bl], antenna2[bl]
        for f in range(n_freq):
            # XX correlation
            dτ_x = delay[a1, 0] - delay[a2, 0]
            φ_x = -2.0 * np.pi * dτ_x * 1e-9 * freq[f]
            V_pred_xx = np.exp(1j * φ_x) * vis_model[bl, f, 0, 0]
            R_xx = vis_obs[bl, f, 0, 0] - V_pred_xx

            # YY correlation
            dτ_y = delay[a1, 1] - delay[a2, 1]
            φ_y = -2.0 * np.pi * dτ_y * 1e-9 * freq[f]
            V_pred_yy = np.exp(1j * φ_y) * vis_model[bl, f, 1, 1]
            R_yy = vis_obs[bl, f, 1, 1] - V_pred_yy

            base = (bl * n_freq + f) * 4
            out[base+0] = R_xx.real
            out[base+1] = R_xx.imag
            out[base+2] = R_yy.real
            out[base+3] = R_yy.imag

    return out


@njit(cache=True, inline='always')
def _inv_2x2(M: np.ndarray) -> np.ndarray:
    """Invert 2x2 matrix."""
    det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    if np.abs(det) < 1e-15:
        return np.eye(2, dtype=M.dtype)
    return np.array([
        [M[1,1]/det, -M[0,1]/det],
        [-M[1,0]/det, M[0,0]/det]
    ], dtype=M.dtype)


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


@njit(parallel=True, cache=True)
def _apply_jones_freq_dependent(
    jones: np.ndarray,
    vis: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray
) -> np.ndarray:
    """
    Apply frequency-dependent Jones: V_out = J_i(f) @ V(f) @ J_j(f)^H

    jones: (n_freq, n_ant, 2, 2)
    vis: (n_bl, n_freq, 2, 2)

    Returns: (n_bl, n_freq, 2, 2)
    """
    n_bl, n_freq = vis.shape[:2]
    out = np.zeros((n_bl, n_freq, 2, 2), dtype=np.complex128)

    for bl in prange(n_bl):
        a1, a2 = antenna1[bl], antenna2[bl]
        for f in range(n_freq):
            Ji = jones[f, a1]
            Jj_H = _herm_2x2(jones[f, a2])
            out[bl, f] = _mm_2x2(_mm_2x2(Ji, vis[bl, f]), Jj_H)

    return out


@njit(parallel=True, cache=True)
def _unapply_jones_freq_dependent(
    jones: np.ndarray,
    vis: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray
) -> np.ndarray:
    """
    Unapply frequency-dependent Jones: V_out = J_i(f)^{-1} @ V(f) @ J_j(f)^{-H}

    jones: (n_freq, n_ant, 2, 2)
    vis: (n_bl, n_freq, 2, 2)

    Returns: (n_bl, n_freq, 2, 2)
    """
    n_bl, n_freq = vis.shape[:2]
    out = np.zeros((n_bl, n_freq, 2, 2), dtype=np.complex128)

    for bl in prange(n_bl):
        a1, a2 = antenna1[bl], antenna2[bl]
        for f in range(n_freq):
            Ji_inv = _inv_2x2(jones[f, a1])
            Jj_inv_H = _herm_2x2(_inv_2x2(jones[f, a2]))
            out[bl, f] = _mm_2x2(_mm_2x2(Ji_inv, vis[bl, f]), Jj_inv_H)

    return out


@njit(parallel=True, cache=True)
def _inverse_jones_freq_dependent(jones: np.ndarray) -> np.ndarray:
    """
    Invert frequency-dependent Jones matrices.

    jones: (n_freq, n_ant, 2, 2)

    Returns: (n_freq, n_ant, 2, 2)
    """
    n_freq, n_ant = jones.shape[:2]
    out = np.zeros((n_freq, n_ant, 2, 2), dtype=np.complex128)

    for f in prange(n_freq):
        for a in range(n_ant):
            out[f, a] = _inv_2x2(jones[f, a])

    return out


# =============================================================================
# K Delay Solver Class
# =============================================================================

class KDelay(JonesEffect):
    """
    K Delay Jones effect.

    Solves for antenna-based delays by fitting phase slopes across frequency.
    """

    def _create_metadata(self) -> JonesMetadata:
        return JonesMetadata(
            jones_type=JonesTypeEnum.K,
            description="Antenna delay (frequency-dependent phase slope). Supports LINEAR (τ_X, τ_Y) and CIRCULAR (τ_R, τ_L) bases.",
            is_diagonal=True,
            is_frequency_dependent=True,
            is_time_dependent=False,
            needs_freq_axis=True,
            native_param_names=['delay_pol0_ns', 'delay_pol1_ns'],
            constraints={'ref_ant': 'delay = 0 for both polarizations'}
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
        feed_basis: str = FeedBasis.LINEAR,
        **kwargs
    ) -> SolveResult:
        """
        Solve for K delays.

        vis_obs, vis_model: (n_bl, n_freq, 2, 2)
        freq: (n_freq,) in Hz - REQUIRED
        feed_basis: 'linear' (XX, YY) or 'circular' (RR, LL)

        For LINEAR basis: solves for delays τ_X, τ_Y (nanoseconds)
        For CIRCULAR basis: solves for delays τ_R, τ_L (nanoseconds)

        Returns: SolveResult with jones (n_freq, n_ant, 2, 2) and delays (n_ant, 2)
        """
        # Validate inputs
        self.validate_inputs(vis_obs, vis_model, antenna1, antenna2, n_ant, working_ants, freq)

        if vis_obs.ndim != 4:
            raise ValueError("K solver requires 4D visibilities (n_bl, n_freq, 2, 2)")

        n_bl, n_freq = vis_obs.shape[:2]

        logger.info(f"Solving K (delay): {len(working_ants)} working antennas, ref_ant={ref_ant}, feed_basis={feed_basis}")

        # Check if ref is working
        if ref_ant not in working_ants:
            logger.error(f"Reference antenna {ref_ant} is not working. Cannot solve.")
            return SolveResult(
                jones=np.full((n_freq, n_ant, 2, 2), np.nan, dtype=np.complex128),
                native_params={'delay': np.full((n_ant, 2), np.nan, dtype=np.float64)},
                cost_init=0.0,
                cost_final=0.0,
                nfev=0,
                success=False,
                convergence_info={'error': 'ref_ant not working'},
                flagging_stats={}
            )

        # Initialize flags if needed
        if flags is None:
            flags = np.zeros(vis_obs.shape, dtype=bool)

        # Make contiguous
        vis_obs = np.ascontiguousarray(vis_obs, dtype=np.complex128)
        vis_model = np.ascontiguousarray(vis_model, dtype=np.complex128)
        antenna1 = np.ascontiguousarray(antenna1, dtype=np.int32)
        antenna2 = np.ascontiguousarray(antenna2, dtype=np.int32)
        freq = np.ascontiguousarray(freq, dtype=np.float64)
        working_ants = np.ascontiguousarray(working_ants, dtype=np.int32)

        # Filter baselines to working antennas only
        working_set = set(working_ants)
        valid_bl_mask = np.array([a1 in working_set and a2 in working_set
                                  for a1, a2 in zip(antenna1, antenna2)])

        vis_obs_filt = vis_obs[valid_bl_mask]
        vis_model_filt = vis_model[valid_bl_mask]
        ant1_filt = antenna1[valid_bl_mask]
        ant2_filt = antenna2[valid_bl_mask]

        n_bl_valid = len(ant1_filt)
        logger.info(f"  Using {n_bl_valid} baselines between working antennas")

        if n_bl_valid == 0:
            logger.error("  No valid baselines between working antennas")
            return SolveResult(
                jones=np.full((n_freq, n_ant, 2, 2), np.nan, dtype=np.complex128),
                native_params={'delay': np.full((n_ant, 2), np.nan, dtype=np.float64)},
                cost_init=0.0,
                cost_final=0.0,
                nfev=0,
                success=False,
                convergence_info={'error': 'no valid baselines'},
                flagging_stats={}
            )

        # Compute initial guess from chain solver
        logger.info("  Computing initial guess from chain solver...")
        delay_init = self._chain_solve(vis_obs_filt, vis_model_filt, ant1_filt, ant2_filt,
                                      freq, n_ant, ref_ant, working_ants)

        # Pack parameters: only working antennas except ref
        p0 = []
        for a in working_ants:
            if a != ref_ant:
                p0.extend([delay_init[a, 0], delay_init[a, 1]])
        p0 = np.array(p0, dtype=np.float64)

        pol0, pol1 = ('X', 'Y') if feed_basis == FeedBasis.LINEAR else ('R', 'L')
        logger.info(f"  Initial delays (nanoseconds): {pol0}=[{delay_init[working_ants, 0].min():.3f}, {delay_init[working_ants, 0].max():.3f}], "
                   f"{pol1}=[{delay_init[working_ants, 1].min():.3f}, {delay_init[working_ants, 1].max():.3f}]")

        # Define residual function
        def residual(p):
            return _delay_residual(p, vis_obs_filt, vis_model_filt, ant1_filt, ant2_filt,
                                  freq, n_ant, ref_ant, working_ants)

        cost_init = np.sum(residual(p0)**2)
        logger.info(f"  Initial cost: {cost_init:.6e}")

        # Optimize
        logger.info("  Optimizing with Levenberg-Marquardt...")
        result = least_squares(
            residual, p0,
            method='lm',
            ftol=tol,
            xtol=tol,
            gtol=tol,
            max_nfev=max_iter * len(p0) if len(p0) > 0 else 1,
            verbose=0
        )

        # Unpack solution
        delay = np.full((n_ant, 2), np.nan, dtype=np.float64)
        delay[ref_ant, :] = 0.0  # Reference antenna

        idx = 0
        for a in working_ants:
            if a != ref_ant:
                delay[a, 0] = result.x[idx]
                delay[a, 1] = result.x[idx + 1]
                idx += 2

        cost_final = result.cost * 2  # least_squares returns 0.5 * sum(residuals^2)

        logger.info(f"  Final cost: {cost_final:.6e} (reduction: {cost_init/max(cost_final, 1e-20):.2f}x)")
        logger.info(f"  Converged: {result.success}, nfev: {result.nfev}")
        pol0, pol1 = ('X', 'Y') if feed_basis == FeedBasis.LINEAR else ('R', 'L')
        logger.info(f"  Final delays (nanoseconds): {pol0}=[{delay[working_ants, 0].min():.3f}, {delay[working_ants, 0].max():.3f}], "
                   f"{pol1}=[{delay[working_ants, 1].min():.3f}, {delay[working_ants, 1].max():.3f}]")

        # Convert to Jones matrices
        jones = _delay_to_jones_single(delay, freq, n_ant)

        return SolveResult(
            jones=jones,
            native_params={'delay': delay, 'feed_basis': feed_basis},
            cost_init=cost_init,
            cost_final=cost_final,
            nfev=result.nfev,
            success=result.success,
            convergence_info={
                'ftol': tol,
                'xtol': tol,
                'gtol': tol,
                'message': result.message,
                'feed_basis': feed_basis
            },
            flagging_stats={}
        )

    def _chain_solve(
        self,
        vis_obs: np.ndarray,
        vis_model: np.ndarray,
        antenna1: np.ndarray,
        antenna2: np.ndarray,
        freq: np.ndarray,
        n_ant: int,
        ref_ant: int,
        working_ants: np.ndarray
    ) -> np.ndarray:
        """
        Chain solver for initial delay guess.

        Fit delay differences per baseline, then propagate from reference.
        """
        n_bl = len(antenna1)
        delay_init = np.zeros((n_ant, 2), dtype=np.float64)
        solved = np.zeros(n_ant, dtype=bool)
        solved[ref_ant] = True

        # Fit delay differences per baseline
        delay_diff = np.zeros((n_bl, 2), dtype=np.float64)
        for bl in range(n_bl):
            # XX
            M_xx = vis_model[bl, :, 0, 0]
            V_xx = vis_obs[bl, :, 0, 0]
            mask = np.abs(M_xx) > 1e-10
            if np.sum(mask) > 2:
                ratio = np.ones(len(freq), dtype=np.complex128)
                ratio[mask] = V_xx[mask] / M_xx[mask]
                delay_diff[bl, 0] = _fit_delay_baseline(ratio, freq)

            # YY
            M_yy = vis_model[bl, :, 1, 1]
            V_yy = vis_obs[bl, :, 1, 1]
            mask = np.abs(M_yy) > 1e-10
            if np.sum(mask) > 2:
                ratio = np.ones(len(freq), dtype=np.complex128)
                ratio[mask] = V_yy[mask] / M_yy[mask]
                delay_diff[bl, 1] = _fit_delay_baseline(ratio, freq)

        # Build adjacency list
        adj = [[] for _ in range(n_ant)]
        for bl in range(n_bl):
            a1, a2 = antenna1[bl], antenna2[bl]
            adj[a1].append((a2, bl, 1))
            adj[a2].append((a1, bl, -1))

        # BFS from reference antenna
        queue = deque([ref_ant])
        while queue:
            curr = queue.popleft()
            for neighbor, bl, direction in adj[curr]:
                if not solved[neighbor]:
                    if direction == 1:
                        delay_init[neighbor] = delay_init[curr] - delay_diff[bl]
                    else:
                        delay_init[neighbor] = delay_init[curr] + delay_diff[bl]
                    solved[neighbor] = True
                    queue.append(neighbor)

        return delay_init

    def apply(
        self,
        jones: np.ndarray,
        vis: np.ndarray,
        antenna1: np.ndarray,
        antenna2: np.ndarray
    ) -> np.ndarray:
        """Apply delay Jones (frequency-dependent)."""
        return _apply_jones_freq_dependent(jones, vis, antenna1, antenna2)

    def unapply(
        self,
        jones: np.ndarray,
        vis: np.ndarray,
        antenna1: np.ndarray,
        antenna2: np.ndarray
    ) -> np.ndarray:
        """Unapply delay Jones (frequency-dependent)."""
        return _unapply_jones_freq_dependent(jones, vis, antenna1, antenna2)

    def inverse(self, jones: np.ndarray) -> np.ndarray:
        """Invert delay Jones (frequency-dependent)."""
        return _inverse_jones_freq_dependent(jones)

    def to_native_params(self, jones: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract delays from Jones matrices.

        This is an approximation - extracts phase at reference frequency.
        Use stored native_params from solve() when available.
        """
        # Not implemented - use stored delays from solve()
        raise NotImplementedError("Use native_params from SolveResult")

    def from_native_params(
        self,
        params: Dict[str, np.ndarray],
        freq: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Convert delays to Jones matrices."""
        if freq is None:
            raise ValueError("freq required to convert delays to Jones")

        delay = params['delay']  # (n_ant, 2)
        n_ant = delay.shape[0]

        return _delay_to_jones_single(delay, freq, n_ant)


__all__ = ['KDelay']
