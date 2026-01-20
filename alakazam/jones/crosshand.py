"""
Cross-hand Jones Solvers (Xf and Kcross) - CORRECTED VERSION.

Xf: Cross-hand phase - NO REFERENCE ANTENNA CONSTRAINT
    All antennas optimized: J = [[1, 0], [0, exp(iφ)]]

Kcross: Cross-hand delay - measures delay BETWEEN polarizations on SAME antenna
    Reference antenna: one pol has 0 delay
    J = [[1, 0], [0, exp(-2πiτν)]] where τ is Y-X delay on same antenna
"""

import numpy as np
from numba import njit, prange
from scipy.optimize import least_squares
from typing import Dict, Optional
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
    # Conversion matrix
    sqrt2 = np.sqrt(2.0)
    P = np.array([[1.0, 1.0], [1j, -1j]], dtype=np.complex128) / sqrt2
    P_inv = np.array([[1.0, -1j], [1.0, 1j]], dtype=np.complex128) / sqrt2

    n_ant = jones_linear.shape[0]
    jones_circ = np.zeros_like(jones_linear)

    for a in range(n_ant):
        jones_circ[a] = P_inv @ jones_linear[a] @ P

    return jones_circ


def convert_circular_to_linear(jones_circ: np.ndarray) -> np.ndarray:
    """Convert Jones from circular (RR, LL) to linear (XX, YY) basis."""
    sqrt2 = np.sqrt(2.0)
    P = np.array([[1.0, 1.0], [1j, -1j]], dtype=np.complex128) / sqrt2
    P_inv = np.array([[1.0, -1j], [1.0, 1j]], dtype=np.complex128) / sqrt2

    n_ant = jones_circ.shape[0]
    jones_lin = np.zeros_like(jones_circ)

    for a in range(n_ant):
        jones_lin[a] = P @ jones_circ[a] @ P_inv

    return jones_lin


# =============================================================================
# Shared utilities
# =============================================================================

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


# =============================================================================
# Xf (Cross-hand phase) - NO REFERENCE ANTENNA CONSTRAINT
# =============================================================================

@njit(cache=True)
def _xf_params_to_jones(params: np.ndarray, n_ant: int) -> np.ndarray:
    """
    Convert parameters to Xf Jones.

    J = [[1, 0], [0, exp(iφ)]]

    NO REFERENCE ANTENNA CONSTRAINT - all antennas have free phase parameter.

    params: [φ_0, φ_1, φ_2, ...] for ALL antennas
    """
    jones = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    for a in range(n_ant):
        jones[a, 0, 0] = 1.0
        jones[a, 1, 1] = np.exp(1j * params[a])
    return jones


@njit(parallel=True, cache=True)
def _xf_residual(params: np.ndarray, vis_obs: np.ndarray, vis_model: np.ndarray,
                antenna1: np.ndarray, antenna2: np.ndarray, n_ant: int) -> np.ndarray:
    """
    Xf residual - uses cross-hand correlations (XY and YX).

    Returns: (n_bl * 4,) [Re(XY), Im(XY), Re(YX), Im(YX)]
    """
    n_bl = len(antenna1)
    jones = _xf_params_to_jones(params, n_ant)

    out = np.zeros(n_bl * 4, dtype=np.float64)
    for bl in prange(n_bl):
        Ji = jones[antenna1[bl]]
        Jj_H = _herm_2x2(jones[antenna2[bl]])
        V_pred = _mm_2x2(_mm_2x2(Ji, vis_model[bl]), Jj_H)

        R_xy = vis_obs[bl, 0, 1] - V_pred[0, 1]
        R_yx = vis_obs[bl, 1, 0] - V_pred[1, 0]

        base = bl * 4
        out[base+0] = R_xy.real
        out[base+1] = R_xy.imag
        out[base+2] = R_yx.real
        out[base+3] = R_yx.imag

    return out


class Xf(JonesEffect):
    """
    Xf Cross-hand phase Jones effect.

    J = [[1, 0], [0, exp(iφ)]]

    NO REFERENCE ANTENNA CONSTRAINT - all antennas have free phase.
    """

    def _create_metadata(self) -> JonesMetadata:
        return JonesMetadata(
            jones_type=JonesTypeEnum.Xf,
            description="Cross-hand phase (no reference antenna constraint)",
            is_diagonal=False,
            is_frequency_dependent=False,
            is_time_dependent=False,
            needs_freq_axis=False,
            native_param_names=['phase_cross'],
            constraints={}
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
        Solve for Xf cross-hand phase.

        NOTE: ref_ant parameter is ignored for Xf (no constraint).
        """
        self.validate_inputs(vis_obs, vis_model, antenna1, antenna2, n_ant, working_ants, freq)

        if vis_obs.ndim != 3 or vis_obs.shape[-2:] != (2, 2):
            raise ValueError("Expected vis_obs shape (n_bl, 2, 2)")

        logger.info(f"Solving Xf (cross-hand phase): {len(working_ants)} working antennas")
        logger.info(f"  Feed basis: {feed_basis}")
        logger.info(f"  NOTE: No reference antenna constraint for Xf")

        # Make contiguous
        vis_obs = np.ascontiguousarray(vis_obs, dtype=np.complex128)
        vis_model = np.ascontiguousarray(vis_model, dtype=np.complex128)
        antenna1 = np.ascontiguousarray(antenna1, dtype=np.int32)
        antenna2 = np.ascontiguousarray(antenna2, dtype=np.int32)

        # Initial guess: phase = 0 for ALL antennas (no reference antenna special case)
        p0 = np.zeros(n_ant, dtype=np.float64)

        def residual(p):
            return _xf_residual(p, vis_obs, vis_model, antenna1, antenna2, n_ant)

        cost_init = np.sum(residual(p0)**2)
        logger.info(f"  Initial cost: {cost_init:.6e}")
        logger.info("  Optimizing...")

        result = least_squares(residual, p0, method='lm', ftol=tol, xtol=tol, gtol=tol,
                              max_nfev=max_iter * len(p0), verbose=0)

        jones = _xf_params_to_jones(result.x, n_ant)

        # Set non-working to NaN
        for a in range(n_ant):
            if a not in working_ants:
                jones[a] = np.nan

        # Convert to appropriate feed basis if needed
        if feed_basis == FeedBasis.CIRCULAR:
            jones = convert_linear_to_circular(jones)

        cost_final = result.cost * 2
        logger.info(f"  Final cost: {cost_final:.6e} (reduction: {cost_init/max(cost_final, 1e-20):.2f}x)")
        logger.info(f"  Converged: {result.success}, nfev: {result.nfev}")

        # Extract phases
        phase_cross = np.angle(jones[:, 1, 1])
        logger.info(f"  Cross-hand phase range: [{np.nanmin(phase_cross[working_ants])*180/np.pi:.1f}, "
                   f"{np.nanmax(phase_cross[working_ants])*180/np.pi:.1f}] deg")

        return SolveResult(
            jones=jones,
            native_params={'phase_cross': phase_cross},
            cost_init=cost_init,
            cost_final=cost_final,
            nfev=result.nfev,
            success=result.success,
            convergence_info={'ftol': tol, 'xtol': tol, 'gtol': tol, 'message': result.message},
            flagging_stats={}
        )

    def apply(self, jones: np.ndarray, vis: np.ndarray, antenna1: np.ndarray, antenna2: np.ndarray) -> np.ndarray:
        """Apply Xf Jones."""
        n_bl = len(antenna1)
        out = np.zeros_like(vis)
        for bl in range(n_bl):
            Ji = jones[antenna1[bl]]
            Jj_H = _herm_2x2(jones[antenna2[bl]])
            out[bl] = _mm_2x2(_mm_2x2(Ji, vis[bl]), Jj_H)
        return out

    def unapply(self, jones: np.ndarray, vis: np.ndarray, antenna1: np.ndarray, antenna2: np.ndarray) -> np.ndarray:
        """Unapply Xf Jones."""
        jones_inv = self.inverse(jones)
        n_bl = len(antenna1)
        out = np.zeros_like(vis)
        for bl in range(n_bl):
            Ji_inv = jones_inv[antenna1[bl]]
            Jj_inv_H = _herm_2x2(jones_inv[antenna1[bl]])
            out[bl] = _mm_2x2(_mm_2x2(Ji_inv, vis[bl]), Jj_inv_H)
        return out

    def inverse(self, jones: np.ndarray) -> np.ndarray:
        """Invert Xf Jones (simple since diagonal-like structure)."""
        jones_inv = jones.copy()
        jones_inv[:, 1, 1] = 1.0 / jones[:, 1, 1]
        return jones_inv

    def to_native_params(self, jones: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract cross-hand phase."""
        return {'phase_cross': np.angle(jones[:, 1, 1])}

    def from_native_params(self, params: Dict[str, np.ndarray], freq: Optional[np.ndarray] = None) -> np.ndarray:
        """Construct Jones from cross-hand phase."""
        n_ant = len(params['phase_cross'])
        jones = np.zeros((n_ant, 2, 2), dtype=np.complex128)
        jones[:, 0, 0] = 1.0
        jones[:, 1, 1] = np.exp(1j * params['phase_cross'])
        return jones


# =============================================================================
# Kcross (Cross-hand delay) - Delay BETWEEN pols on SAME antenna
# =============================================================================

@njit(cache=True)
def _fit_delay_baseline(vis_ratio: np.ndarray, freq: np.ndarray) -> float:
    """Fit delay from phase vs frequency for single baseline."""
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

    # Linear fit
    f_mean = np.mean(freq)
    p_mean = np.mean(unwrapped)
    num, den = 0.0, 0.0
    for i in range(n):
        df = freq[i] - f_mean
        dp = unwrapped[i] - p_mean
        num += df * dp
        den += df * df

    if den < 1e-20:
        return 0.0

    slope = num / den
    return -slope / (2.0 * np.pi) * 1e9  # nanoseconds


@njit(parallel=True, cache=True)
def _kcross_residual(params: np.ndarray, vis_obs: np.ndarray, vis_model: np.ndarray,
                    antenna1: np.ndarray, antenna2: np.ndarray, freq: np.ndarray,
                    n_ant: int, ref_ant: int) -> np.ndarray:
    """
    Kcross residual.

    Kcross measures the delay BETWEEN the two polarizations on the SAME antenna.

    J = [[1, 0], [0, exp(-2πi τ ν)]]

    where τ is the Y-X delay on each antenna.

    For cross-correlations: φ_cross = φ(Y_i - X_i) - φ(Y_j - X_j) = (τ_i - τ_j) * 2πν

    params: [τ for each working ant except ref] in nanoseconds
    """
    n_bl, n_freq = vis_obs.shape[:2]

    # Unpack delays (delay between pols on same antenna)
    delay_cross = np.zeros(n_ant, dtype=np.float64)
    idx = 0
    for a in range(n_ant):
        if a != ref_ant:
            delay_cross[a] = params[idx]
            idx += 1

    # Compute residuals for cross-hand correlations
    # The model is: V_XY and V_YX affected by the pol delay difference on each antenna
    out = np.zeros(n_bl * n_freq * 4, dtype=np.float64)

    for bl in prange(n_bl):
        a1, a2 = antenna1[bl], antenna2[bl]
        for f in range(n_freq):
            # Delay difference: (τ_Y - τ_X) for antenna i minus (τ_Y - τ_X) for antenna j
            dτ = delay_cross[a1] - delay_cross[a2]
            φ = -2.0 * np.pi * dτ * 1e-9 * freq[f]
            phase_factor = np.exp(1j * φ)

            # XY correlation: affected by phase_factor
            V_pred_xy = phase_factor * vis_model[bl, f, 0, 1]
            R_xy = vis_obs[bl, f, 0, 1] - V_pred_xy

            # YX correlation: affected by conjugate
            V_pred_yx = np.conj(phase_factor) * vis_model[bl, f, 1, 0]
            R_yx = vis_obs[bl, f, 1, 0] - V_pred_yx

            base = (bl * n_freq + f) * 4
            out[base+0] = R_xy.real
            out[base+1] = R_xy.imag
            out[base+2] = R_yx.real
            out[base+3] = R_yx.imag

    return out


@njit(parallel=True, cache=True)
def _kcross_delay_to_jones(delay_cross: np.ndarray, freq: np.ndarray, n_ant: int) -> np.ndarray:
    """
    Convert cross-pol delays to Jones matrices.

    delay_cross[a] = delay between Y and X pols on antenna a
    J[a] = [[1, 0], [0, exp(-2πi τ_a ν)]]
    """
    n_freq = len(freq)
    jones = np.zeros((n_freq, n_ant, 2, 2), dtype=np.complex128)

    for f in prange(n_freq):
        for a in range(n_ant):
            jones[f, a, 0, 0] = 1.0
            phase = -2.0 * np.pi * delay_cross[a] * 1e-9 * freq[f]
            jones[f, a, 1, 1] = np.exp(1j * phase)

    return jones


class Kcross(JonesEffect):
    """
    Kcross Cross-polarization delay Jones effect.

    Measures the delay BETWEEN the two polarizations on the SAME antenna.

    J = [[1, 0], [0, exp(-2πi τ ν)]]

    where τ is the Y-X delay (or LL-RR for circular feeds).

    Reference antenna constraint: τ_ref = 0 (one pol is reference)
    """

    def _create_metadata(self) -> JonesMetadata:
        return JonesMetadata(
            jones_type=JonesTypeEnum.Kcross,
            description="Cross-polarization delay (measures Y-X delay on same antenna)",
            is_diagonal=False,
            is_frequency_dependent=True,
            is_time_dependent=False,
            needs_freq_axis=True,
            native_param_names=['delay_cross_ns'],
            constraints={}
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
        Solve for Kcross cross-polarization delay.

        Measures delay between Y and X (or LL and RR for circular) on same antenna.
        """
        self.validate_inputs(vis_obs, vis_model, antenna1, antenna2, n_ant, working_ants, freq)

        if vis_obs.ndim != 4:
            raise ValueError("Kcross requires 4D visibilities (n_bl, n_freq, 2, 2)")

        n_bl, n_freq = vis_obs.shape[:2]

        logger.info(f"Solving Kcross (cross-pol delay): {len(working_ants)} working antennas, ref_ant={ref_ant}")
        logger.info(f"  Feed basis: {feed_basis}")
        logger.info(f"  Measuring delay between Y-X (or LL-RR) on same antenna")

        if ref_ant not in working_ants:
            logger.error(f"Reference antenna {ref_ant} is not working")
            return SolveResult(
                jones=np.full((n_freq, n_ant, 2, 2), np.nan, dtype=np.complex128),
                native_params={'delay_cross': np.full(n_ant, np.nan, dtype=np.float64)},
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
        freq = np.ascontiguousarray(freq, dtype=np.float64)

        # Chain solve initial guess
        logger.info("  Computing initial guess...")
        delay_init = self._chain_solve(vis_obs, vis_model, antenna1, antenna2, freq, n_ant)

        # Pack params (exclude ref antenna)
        p0 = []
        for a in range(n_ant):
            if a != ref_ant:
                p0.append(delay_init[a])
        p0 = np.array(p0, dtype=np.float64)

        logger.info(f"  Initial cross-pol delays (ns): [{np.min(delay_init):.3f}, {np.max(delay_init):.3f}]")

        def residual(p):
            return _kcross_residual(p, vis_obs, vis_model, antenna1, antenna2, freq, n_ant, ref_ant)

        cost_init = np.sum(residual(p0)**2)
        logger.info(f"  Initial cost: {cost_init:.6e}")
        logger.info("  Optimizing...")

        result = least_squares(residual, p0, method='lm', ftol=tol, xtol=tol, gtol=tol,
                              max_nfev=max_iter * len(p0), verbose=0)

        # Unpack
        delay_cross = np.full(n_ant, np.nan, dtype=np.float64)
        delay_cross[ref_ant] = 0.0  # Reference: Y-X delay = 0
        idx = 0
        for a in range(n_ant):
            if a != ref_ant:
                if a in working_ants:
                    delay_cross[a] = result.x[idx]
                idx += 1

        jones = _kcross_delay_to_jones(delay_cross, freq, n_ant)

        # Set non-working to NaN
        for a in range(n_ant):
            if a not in working_ants:
                jones[:, a] = np.nan

        # Convert to appropriate feed basis if needed
        if feed_basis == FeedBasis.CIRCULAR:
            for f in range(n_freq):
                jones[f] = convert_linear_to_circular(jones[f])

        cost_final = result.cost * 2
        logger.info(f"  Final cost: {cost_final:.6e} (reduction: {cost_init/max(cost_final, 1e-20):.2f}x)")
        logger.info(f"  Converged: {result.success}, nfev: {result.nfev}")
        logger.info(f"  Final cross-pol delays (ns): [{np.nanmin(delay_cross[working_ants]):.3f}, "
                   f"{np.nanmax(delay_cross[working_ants]):.3f}]")

        return SolveResult(
            jones=jones,
            native_params={'delay_cross': delay_cross},
            cost_init=cost_init,
            cost_final=cost_final,
            nfev=result.nfev,
            success=result.success,
            convergence_info={'ftol': tol, 'xtol': tol, 'gtol': tol, 'message': result.message},
            flagging_stats={}
        )

    def _chain_solve(self, vis_obs, vis_model, antenna1, antenna2, freq, n_ant):
        """
        Simple initial guess for Kcross.

        Estimate cross-pol delay from cross-correlation phase slopes.
        """
        n_bl = len(antenna1)
        delay_est = []

        for bl in range(n_bl):
            # Use XY correlation to estimate delay difference
            M_xy = vis_model[bl, :, 0, 1]
            V_xy = vis_obs[bl, :, 0, 1]
            mask = np.abs(M_xy) > 1e-10
            if np.sum(mask) > 2:
                ratio = np.where(mask, V_xy / M_xy, 1.0+0j)
                delay_diff = _fit_delay_baseline(ratio, freq)
                delay_est.append(delay_diff)

        if len(delay_est) > 0:
            # Use median as typical value
            typical_delay = np.median(np.array(delay_est))
        else:
            typical_delay = 0.0

        # Start all antennas near typical value (will be refined by optimization)
        delay_init = np.full(n_ant, typical_delay, dtype=np.float64)

        return delay_init

    def apply(self, jones, vis, antenna1, antenna2):
        """Apply Kcross (freq-dependent)."""
        from .delay import _apply_jones_freq_dependent
        return _apply_jones_freq_dependent(jones, vis, antenna1, antenna2)

    def unapply(self, jones, vis, antenna1, antenna2):
        """Unapply Kcross (freq-dependent)."""
        from .delay import _unapply_jones_freq_dependent
        return _unapply_jones_freq_dependent(jones, vis, antenna1, antenna2)

    def inverse(self, jones):
        """Invert Kcross (freq-dependent)."""
        from .delay import _inverse_jones_freq_dependent
        return _inverse_jones_freq_dependent(jones)

    def to_native_params(self, jones):
        """Extract cross-pol delays."""
        raise NotImplementedError("Use native_params from SolveResult")

    def from_native_params(self, params, freq=None):
        """Convert cross-pol delays to Jones."""
        if freq is None:
            raise ValueError("freq required")
        return _kcross_delay_to_jones(params['delay_cross'], freq, len(params['delay_cross']))


__all__ = ['Xf', 'Kcross', 'FeedBasis', 'convert_linear_to_circular', 'convert_circular_to_linear']
