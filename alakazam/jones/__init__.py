"""
JACKAL Jones Module.

All Jones types, operations, and solvers.
Fully vectorized with Numba JIT.

Jones Types:
- K: Delay (diagonal, fit across freq)
- B: Bandpass (diagonal, per freq chunk)
- G: Gain (diagonal, per time chunk)  
- D: Leakage (off-diagonal)
- Kcross: Cross-hand delay
- Xf: Cross-hand phase

All solutions stored as: (n_time, n_freq, n_ant, 2, 2) complex128
"""

import numpy as np
import numba
from numba import njit, prange
from scipy.optimize import least_squares
from typing import Tuple, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass


# =============================================================================
# Enums and Config
# =============================================================================

class FeedBasis(Enum):
    LINEAR = "linear"
    CIRCULAR = "circular"


class JonesType(Enum):
    K = "K"
    B = "B"
    G = "G"
    D = "D"
    Kcross = "Kcross"
    Xf = "Xf"


# =============================================================================
# Core 2x2 Operations (Numba JIT)
# =============================================================================

@njit(cache=True, inline='always')
def _mm(A, B):
    """2x2 matrix multiply."""
    return np.array([
        [A[0,0]*B[0,0] + A[0,1]*B[1,0], A[0,0]*B[0,1] + A[0,1]*B[1,1]],
        [A[1,0]*B[0,0] + A[1,1]*B[1,0], A[1,0]*B[0,1] + A[1,1]*B[1,1]]
    ], dtype=np.complex128)


@njit(cache=True, inline='always')
def _inv(M):
    """2x2 matrix inverse."""
    det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    if np.abs(det) < 1e-15:
        return np.eye(2, dtype=np.complex128)
    return np.array([
        [M[1,1]/det, -M[0,1]/det],
        [-M[1,0]/det, M[0,0]/det]
    ], dtype=np.complex128)


@njit(cache=True, inline='always')
def _herm(M):
    """2x2 Hermitian (conjugate transpose)."""
    return np.array([
        [np.conj(M[0,0]), np.conj(M[1,0])],
        [np.conj(M[0,1]), np.conj(M[1,1])]
    ], dtype=np.complex128)


# =============================================================================
# Vectorized Batch Operations
# =============================================================================

@njit(parallel=True, cache=True)
def jones_multiply(J1, J2):
    """Multiply Jones: J_out = J1 @ J2 per antenna. Shape (n_ant, 2, 2)."""
    n = J1.shape[0]
    out = np.zeros((n, 2, 2), dtype=np.complex128)
    for i in prange(n):
        out[i] = _mm(J1[i], J2[i])
    return out


@njit(parallel=True, cache=True)
def jones_inverse(J):
    """Invert Jones per antenna. Shape (n_ant, 2, 2)."""
    n = J.shape[0]
    out = np.zeros((n, 2, 2), dtype=np.complex128)
    for i in prange(n):
        out[i] = _inv(J[i])
    return out


@njit(parallel=True, cache=True)
def jones_apply(jones, vis, antenna1, antenna2):
    """
    Apply Jones: V_out = J_i @ V @ J_j^H
    
    jones: (n_ant, 2, 2)
    vis: (n_bl, 2, 2)
    Returns: (n_bl, 2, 2)
    """
    n_bl = vis.shape[0]
    out = np.zeros((n_bl, 2, 2), dtype=np.complex128)
    
    for bl in prange(n_bl):
        Ji = jones[antenna1[bl]]
        Jj_H = _herm(jones[antenna2[bl]])
        out[bl] = _mm(_mm(Ji, vis[bl]), Jj_H)
    
    return out


@njit(parallel=True, cache=True)
def jones_unapply(jones, vis, antenna1, antenna2):
    """
    Unapply Jones: V_out = J_i^{-1} @ V @ J_j^{-H}
    
    jones: (n_ant, 2, 2)
    vis: (n_bl, 2, 2)
    Returns: (n_bl, 2, 2)
    """
    n_bl = vis.shape[0]
    out = np.zeros((n_bl, 2, 2), dtype=np.complex128)
    
    for bl in prange(n_bl):
        Ji_inv = _inv(jones[antenna1[bl]])
        Jj_inv_H = _herm(_inv(jones[antenna2[bl]]))
        out[bl] = _mm(_mm(Ji_inv, vis[bl]), Jj_inv_H)
    
    return out


@njit(parallel=True, cache=True)
def compute_residual(jones, vis_obs, vis_model, antenna1, antenna2):
    """
    Compute residual: R = V_obs - J_i @ M @ J_j^H
    Returns flattened [Re, Im] for all baselines and correlations.
    """
    n_bl = vis_obs.shape[0]
    out = np.zeros(n_bl * 8, dtype=np.float64)  # 4 corr * 2 (re/im)
    
    for bl in prange(n_bl):
        Ji = jones[antenna1[bl]]
        Jj_H = _herm(jones[antenna2[bl]])
        V_pred = _mm(_mm(Ji, vis_model[bl]), Jj_H)
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


# =============================================================================
# Delay (K) - Fit phase slope across frequency
# =============================================================================

@njit(parallel=True, cache=True)
def delay_to_jones(delay, freq):
    """
    Convert delays to Jones matrices.
    
    delay: (n_ant, 2) float64, nanoseconds [τ_X, τ_Y]
    freq: (n_freq,) float64, Hz
    
    Returns: (n_freq, n_ant, 2, 2) complex128
    J = diag(exp(-2πi * τ * ν), ...)
    """
    n_ant = delay.shape[0]
    n_freq = freq.shape[0]
    jones = np.zeros((n_freq, n_ant, 2, 2), dtype=np.complex128)
    
    for f in prange(n_freq):
        for a in range(n_ant):
            phase_x = -2.0 * np.pi * delay[a, 0] * 1e-9 * freq[f]
            phase_y = -2.0 * np.pi * delay[a, 1] * 1e-9 * freq[f]
            jones[f, a, 0, 0] = np.exp(1j * phase_x)
            jones[f, a, 1, 1] = np.exp(1j * phase_y)
    
    return jones


@njit(cache=True)
def _fit_delay_baseline(vis_ratio, freq):
    """
    Fit delay from phase vs freq for single baseline.
    vis_ratio: V_obs / M, shape (n_freq,)
    Returns delay in ns.
    """
    n = len(freq)
    phase = np.angle(vis_ratio)
    
    # Unwrap phase
    unwrapped = np.zeros(n, dtype=np.float64)
    unwrapped[0] = phase[0]
    for i in range(1, n):
        d = phase[i] - phase[i-1]
        if d > np.pi: d -= 2*np.pi
        elif d < -np.pi: d += 2*np.pi
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
    return -slope / (2.0 * np.pi) * 1e9  # ns


@njit(parallel=True, cache=True)
def _delay_residual(params, vis_obs, vis_model, antenna1, antenna2, freq, n_ant, ref_ant):
    """
    Delay residual for least squares.
    
    params: [(τ_X, τ_Y) for each ant except ref]
    vis_obs, vis_model: (n_bl, n_freq, 2, 2)
    """
    n_bl = len(antenna1)
    n_freq = len(freq)
    
    # Unpack delays
    delay = np.zeros((n_ant, 2), dtype=np.float64)
    idx = 0
    for a in range(n_ant):
        if a != ref_ant:
            delay[a, 0] = params[idx]
            delay[a, 1] = params[idx + 1]
            idx += 2
    
    # Compute residuals (XX and YY only for diagonal)
    out = np.zeros(n_bl * n_freq * 4, dtype=np.float64)
    
    for bl in prange(n_bl):
        a1, a2 = antenna1[bl], antenna2[bl]
        for f in range(n_freq):
            # XX
            dτ_x = delay[a1, 0] - delay[a2, 0]
            φ_x = -2.0 * np.pi * dτ_x * 1e-9 * freq[f]
            V_pred_xx = np.exp(1j * φ_x) * vis_model[bl, f, 0, 0]
            R_xx = vis_obs[bl, f, 0, 0] - V_pred_xx
            
            # YY
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


"""
New K delay solver - only solves for working antennas, outputs NaNs for non-working ones.
"""

import numpy as np
from scipy.optimize import least_squares
from numba import njit


@njit(cache=True)
def _fit_delay_baseline(ratio, freq):
    """Fit delay from phase slope vs frequency."""
    phase = np.angle(ratio)
    phase_unwrap = np.unwrap(phase)
    weights = np.abs(ratio)

    # Weighted linear fit
    w_sum = np.sum(weights)
    if w_sum < 1e-10:
        return 0.0

    f_mean = np.sum(freq * weights) / w_sum
    p_mean = np.sum(phase_unwrap * weights) / w_sum

    num = np.sum(weights * (freq - f_mean) * (phase_unwrap - p_mean))
    den = np.sum(weights * (freq - f_mean)**2)

    if np.abs(den) < 1e-10:
        return 0.0

    slope = num / den
    delay_s = -slope / (2 * np.pi)
    delay_ns = delay_s * 1e9

    return delay_ns


@njit(cache=True)
def _delay_residual(params, vis_obs, vis_model, antenna1, antenna2, freq, n_ant, ref_ant):
    """Residual function for delay solver."""
    n_bl = len(antenna1)
    n_freq = len(freq)

    # Unpack delays (ref is fixed at 0)
    delay = np.zeros((n_ant, 2), dtype=np.float64)
    idx = 0
    for a in range(n_ant):
        if a != ref_ant:
            delay[a, 0] = params[idx]
            delay[a, 1] = params[idx + 1]
            idx += 2

    # Compute residuals
    residuals = np.zeros(n_bl * n_freq * 4, dtype=np.float64)
    res_idx = 0

    for bl in range(n_bl):
        a1 = antenna1[bl]
        a2 = antenna2[bl]

        for f_idx in range(n_freq):
            f = freq[f_idx]

            # Jones matrices for a1, a2
            # K = exp(-2πi * delay * freq)
            phase_x1 = -2 * np.pi * delay[a1, 0] * f * 1e-9
            phase_y1 = -2 * np.pi * delay[a1, 1] * f * 1e-9
            phase_x2 = -2 * np.pi * delay[a2, 0] * f * 1e-9
            phase_y2 = -2 * np.pi * delay[a2, 1] * f * 1e-9

            K1_x = np.exp(1j * phase_x1)
            K1_y = np.exp(1j * phase_y1)
            K2_x = np.exp(1j * phase_x2)
            K2_y = np.exp(1j * phase_y2)

            # Model visibility with K applied
            M_xx = vis_model[bl, f_idx, 0, 0]
            M_xy = vis_model[bl, f_idx, 0, 1]
            M_yx = vis_model[bl, f_idx, 1, 0]
            M_yy = vis_model[bl, f_idx, 1, 1]

            # Apply K: V_model' = K1 @ V_model @ K2^H
            V_xx = K1_x * M_xx * np.conj(K2_x)
            V_xy = K1_x * M_xy * np.conj(K2_y)
            V_yx = K1_y * M_yx * np.conj(K2_x)
            V_yy = K1_y * M_yy * np.conj(K2_y)

            # Observed
            O_xx = vis_obs[bl, f_idx, 0, 0]
            O_xy = vis_obs[bl, f_idx, 0, 1]
            O_yx = vis_obs[bl, f_idx, 1, 0]
            O_yy = vis_obs[bl, f_idx, 1, 1]

            # Residuals (real and imag)
            residuals[res_idx] = (O_xx - V_xx).real
            residuals[res_idx + 1] = (O_xx - V_xx).imag
            residuals[res_idx + 2] = (O_yy - V_yy).imag
            residuals[res_idx + 3] = (O_yy - V_yy).imag
            res_idx += 4

    return residuals


def solve_delay(vis_obs, vis_model, antenna1, antenna2, freq, n_ant, ref_ant=0,
                             max_iter=100, tol=1e-10, verbose=True):
    """
    Solve for antenna delays K - only for working antennas.

    Parameters
    ----------
    vis_obs : (n_bl, n_freq, 2, 2) complex128
        Observed visibilities
    vis_model : (n_bl, n_freq, 2, 2) complex128
        Model visibilities
    antenna1, antenna2 : (n_bl,) int32
        Antenna indices for each baseline
    freq : (n_freq,) float64
        Frequencies in Hz
    n_ant : int
        Total number of antennas in array (including non-working)
    ref_ant : int
        Reference antenna index (must be working)

    Returns
    -------
    delay_full : (n_ant, 2) float64
        Delays in nanoseconds. NaN for non-working antennas.
    info : dict
        Solver info including n_working_ants
    """
    n_bl = len(antenna1)
    n_freq = len(freq)

    # Identify working antennas
    working_ants = np.unique(np.concatenate([antenna1, antenna2]))
    n_working = len(working_ants)

    if ref_ant not in working_ants:
        raise ValueError(f"Reference antenna {ref_ant} has no data")

    if verbose:
        print(f"[ALAKAZAM] K solver: {n_working}/{n_ant} working ants, {n_bl} bl, {n_freq} chan, ref={ref_ant}")

    # Mapping: full index <-> working index
    full_to_working = {ant: i for i, ant in enumerate(working_ants)}
    working_to_full = {i: ant for i, ant in enumerate(working_ants)}

    # Remap antenna indices
    ant1_work = np.array([full_to_working[a] for a in antenna1], dtype=np.int32)
    ant2_work = np.array([full_to_working[a] for a in antenna2], dtype=np.int32)
    ref_work = full_to_working[ref_ant]

    # Initial guess from phase slopes
    delay_init = np.zeros((n_working, 2), dtype=np.float64)
    solved = np.zeros(n_working, dtype=bool)
    solved[ref_work] = True

    delay_diff = np.zeros((n_bl, 2), dtype=np.float64)
    for bl in range(n_bl):
        # XX
        M_xx = vis_model[bl, :, 0, 0]
        V_xx = vis_obs[bl, :, 0, 0]
        mask = (np.abs(M_xx) > 1e-10) & (np.abs(V_xx) > 1e-10) & np.isfinite(V_xx) & np.isfinite(M_xx)
        if np.sum(mask) > 2:
            ratio = V_xx[mask] / M_xx[mask]
            delay_diff[bl, 0] = _fit_delay_baseline(ratio, freq[mask])

        # YY
        M_yy = vis_model[bl, :, 1, 1]
        V_yy = vis_obs[bl, :, 1, 1]
        mask = (np.abs(M_yy) > 1e-10) & (np.abs(V_yy) > 1e-10) & np.isfinite(V_yy) & np.isfinite(M_yy)
        if np.sum(mask) > 2:
            ratio = V_yy[mask] / M_yy[mask]
            delay_diff[bl, 1] = _fit_delay_baseline(ratio, freq[mask])

    # BFS to propagate initial guess
    adj = [[] for _ in range(n_working)]
    for bl in range(n_bl):
        a1, a2 = ant1_work[bl], ant2_work[bl]
        adj[a1].append((a2, bl, 1))
        adj[a2].append((a1, bl, -1))

    from collections import deque
    queue = deque([ref_work])
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

    if verbose:
        print(f"[ALAKAZAM] Initial delays (ns):")
        for i in range(n_working):
            ant_full = working_to_full[i]
            status = "[ref]" if i == ref_work else ""
            print(f"        Ant {ant_full:2d}: X={delay_init[i,0]:+8.3f}, Y={delay_init[i,1]:+8.3f}  {status}")

    # Pack parameters
    p0 = []
    for i in range(n_working):
        if i != ref_work:
            p0.extend([delay_init[i, 0], delay_init[i, 1]])
    p0 = np.array(p0, dtype=np.float64)

    # Make contiguous
    vis_obs = np.ascontiguousarray(vis_obs, dtype=np.complex128)
    vis_model = np.ascontiguousarray(vis_model, dtype=np.complex128)
    ant1_work = np.ascontiguousarray(ant1_work, dtype=np.int32)
    ant2_work = np.ascontiguousarray(ant2_work, dtype=np.int32)
    freq = np.ascontiguousarray(freq, dtype=np.float64)

    def residual(p):
        return _delay_residual(p, vis_obs, vis_model, ant1_work, ant2_work, freq, n_working, ref_work)

    cost_init = np.sum(residual(p0)**2)

    result = least_squares(residual, p0, method='lm', ftol=tol, xtol=tol, gtol=tol,
                          max_nfev=max_iter * max(1, len(p0)))

    # Unpack
    delay_work = np.zeros((n_working, 2), dtype=np.float64)
    idx = 0
    for i in range(n_working):
        if i != ref_work:
            delay_work[i, 0] = result.x[idx]
            delay_work[i, 1] = result.x[idx + 1]
            idx += 2

    if verbose:
        print(f"[ALAKAZAM] Cost: {cost_init:.4e} -> {result.cost*2:.4e}")
        print(f"[ALAKAZAM] Final delays (ns):")
        for i in range(n_working):
            ant_full = working_to_full[i]
            status = "[ref]" if i == ref_work else ""
            print(f"        Ant {ant_full:2d}: X={delay_work[i,0]:+8.3f}, Y={delay_work[i,1]:+8.3f}  {status}")

    # Expand to full array with NaNs
    delay_full = np.full((n_ant, 2), np.nan, dtype=np.float64)
    for i in range(n_working):
        ant_full = working_to_full[i]
        delay_full[ant_full] = delay_work[i]

    return delay_full, {
        'cost_init': cost_init,
        'cost_final': result.cost * 2,
        'nfev': result.nfev,
        'n_working_ants': n_working,
        'working_ants': working_ants
    }



# =============================================================================
# Diagonal Gain (G, B) - Amplitude + Phase or Phase-only
# =============================================================================

@njit(cache=True)
def _diag_params_to_jones_ap(params, n_ant, ref_ant):
    """
    Amplitude+Phase params to diagonal Jones.
    Ref: [amp_X, amp_Y] (phase=0)
    Others: [amp_X, phase_X, amp_Y, phase_Y]
    """
    jones = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    idx = 0
    for a in range(n_ant):
        if a == ref_ant:
            jones[a, 0, 0] = params[idx]
            jones[a, 1, 1] = params[idx + 1]
            idx += 2
        else:
            jones[a, 0, 0] = params[idx] * np.exp(1j * params[idx + 1])
            jones[a, 1, 1] = params[idx + 2] * np.exp(1j * params[idx + 3])
            idx += 4
    return jones


@njit(cache=True)
def _diag_jones_to_params_ap(jones, ref_ant):
    """Diagonal Jones to amp+phase params."""
    n_ant = jones.shape[0]
    n_params = 2 + 4 * (n_ant - 1)
    params = np.zeros(n_params, dtype=np.float64)
    idx = 0
    for a in range(n_ant):
        if a == ref_ant:
            params[idx] = np.abs(jones[a, 0, 0])
            params[idx + 1] = np.abs(jones[a, 1, 1])
            idx += 2
        else:
            params[idx] = np.abs(jones[a, 0, 0])
            params[idx + 1] = np.angle(jones[a, 0, 0])
            params[idx + 2] = np.abs(jones[a, 1, 1])
            params[idx + 3] = np.angle(jones[a, 1, 1])
            idx += 4
    return params


@njit(cache=True)
def _diag_params_to_jones_p(params, n_ant, ref_ant):
    """Phase-only params to diagonal Jones (amp=1)."""
    jones = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    idx = 0
    for a in range(n_ant):
        if a == ref_ant:
            jones[a, 0, 0] = 1.0
            jones[a, 1, 1] = 1.0
        else:
            jones[a, 0, 0] = np.exp(1j * params[idx])
            jones[a, 1, 1] = np.exp(1j * params[idx + 1])
            idx += 2
    return jones


@njit(parallel=True, cache=True)
def _diag_residual_ap(params, vis_obs, vis_model, antenna1, antenna2, n_ant, ref_ant):
    """Diagonal residual (amp+phase mode)."""
    n_bl = len(antenna1)
    jones = _diag_params_to_jones_ap(params, n_ant, ref_ant)
    
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


@njit(parallel=True, cache=True)
def _diag_residual_p(params, vis_obs, vis_model, antenna1, antenna2, n_ant, ref_ant):
    """Diagonal residual (phase-only mode)."""
    n_bl = len(antenna1)
    jones = _diag_params_to_jones_p(params, n_ant, ref_ant)
    
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
def _chain_solve_diagonal(vis_obs, vis_model, antenna1, antenna2, n_ant, ref_ant):
    """
    Chain solver for diagonal gains.
    Returns (n_ant, 2) complex gains.
    """
    n_bl = len(antenna1)
    gains = np.ones((n_ant, 2), dtype=np.complex128)
    solved = np.zeros(n_ant, dtype=np.bool_)
    
    # Ref amplitude from median
    ref_amps_x = np.zeros(n_bl, dtype=np.float64)
    ref_amps_y = np.zeros(n_bl, dtype=np.float64)
    n_ref = 0
    for bl in range(n_bl):
        a1, a2 = antenna1[bl], antenna2[bl]
        if a1 == ref_ant or a2 == ref_ant:
            M_xx = vis_model[bl, 0, 0]
            M_yy = vis_model[bl, 1, 1]
            if np.abs(M_xx) > 1e-10:
                ref_amps_x[n_ref] = np.sqrt(np.abs(vis_obs[bl, 0, 0] / M_xx))
            if np.abs(M_yy) > 1e-10:
                ref_amps_y[n_ref] = np.sqrt(np.abs(vis_obs[bl, 1, 1] / M_yy))
            n_ref += 1
    
    if n_ref > 0:
        gains[ref_ant, 0] = np.median(ref_amps_x[:n_ref])
        gains[ref_ant, 1] = np.median(ref_amps_y[:n_ref])
    solved[ref_ant] = True
    
    # Propagate
    for _ in range(n_ant):
        for bl in range(n_bl):
            a1, a2 = antenna1[bl], antenna2[bl]
            
            if solved[a1] and not solved[a2]:
                for pol in range(2):
                    M = vis_model[bl, pol, pol]
                    V = vis_obs[bl, pol, pol]
                    g_known = gains[a1, pol]
                    if np.abs(M) > 1e-10 and np.abs(g_known) > 1e-10:
                        gains[a2, pol] = np.conj(V / (g_known * M))
                solved[a2] = True
                
            elif solved[a2] and not solved[a1]:
                for pol in range(2):
                    M = vis_model[bl, pol, pol]
                    V = vis_obs[bl, pol, pol]
                    g_known = gains[a2, pol]
                    if np.abs(M) > 1e-10 and np.abs(g_known) > 1e-10:
                        gains[a1, pol] = V / (M * np.conj(g_known))
                solved[a1] = True
    
    return gains


def solve_diagonal(vis_obs, vis_model, antenna1, antenna2, n_ant, ref_ant=0,
                   phase_only=False, max_iter=100, tol=1e-10, verbose=True):
    """
    Solve for diagonal Jones (G or B).
    
    vis_obs, vis_model: (n_bl, 2, 2)
    
    Returns: jones (n_ant, 2, 2) complex, info dict
    """
    n_bl = len(antenna1)
    mode = "phase-only" if phase_only else "amp+phase"
    
    if verbose:
        print(f"[ALAKAZAM] Diagonal solver ({mode}): {n_ant} ants, {n_bl} bl, ref={ref_ant}")
    
    # Chain solve initial
    gains_init = _chain_solve_diagonal(vis_obs, vis_model, antenna1, antenna2, n_ant, ref_ant)
    
    if verbose:
        print(f"[ALAKAZAM] Initial (chain):")
        # Print only antennas with non-zero gains (working antennas)
        working = []
        for a in range(n_ant):
            gx, gy = gains_init[a, 0], gains_init[a, 1]
            if np.abs(gx) > 1e-6 or np.abs(gy) > 1e-6:
                working.append(a)

        for a in working[:10]:  # Show first 10 working antennas
            gx, gy = gains_init[a, 0], gains_init[a, 1]
            ref_mark = "[ref]" if a == ref_ant else ""
            print(f"        Ant {a:2d}: X=({np.abs(gx):.4f}, {np.degrees(np.angle(gx)):+7.1f}°)  "
                  f"Y=({np.abs(gy):.4f}, {np.degrees(np.angle(gy)):+7.1f}°)  {ref_mark}")

        if len(working) > 10:
            print(f"        ... and {len(working) - 10} more working antennas")

        print(f"[ALAKAZAM] Working antennas: {len(working)}/{n_ant}")
    
    # Make contiguous
    vis_obs = np.ascontiguousarray(vis_obs, dtype=np.complex128)
    vis_model = np.ascontiguousarray(vis_model, dtype=np.complex128)
    antenna1 = np.ascontiguousarray(antenna1, dtype=np.int32)
    antenna2 = np.ascontiguousarray(antenna2, dtype=np.int32)
    
    if phase_only:
        # Pack phases (exclude ref)
        p0 = []
        for a in range(n_ant):
            if a != ref_ant:
                p0.extend([np.angle(gains_init[a, 0]), np.angle(gains_init[a, 1])])
        p0 = np.array(p0, dtype=np.float64)
        
        def residual(p):
            return _diag_residual_p(p, vis_obs, vis_model, antenna1, antenna2, n_ant, ref_ant)
        
        cost_init = np.sum(residual(p0)**2)
        result = least_squares(residual, p0, method='lm', ftol=tol, xtol=tol, gtol=tol,
                              max_nfev=max_iter * len(p0))
        jones = _diag_params_to_jones_p(result.x, n_ant, ref_ant)
    else:
        # Build Jones and get params
        jones_init = np.zeros((n_ant, 2, 2), dtype=np.complex128)
        jones_init[:, 0, 0] = gains_init[:, 0]
        jones_init[:, 1, 1] = gains_init[:, 1]
        p0 = _diag_jones_to_params_ap(jones_init, ref_ant)
        
        def residual(p):
            return _diag_residual_ap(p, vis_obs, vis_model, antenna1, antenna2, n_ant, ref_ant)
        
        cost_init = np.sum(residual(p0)**2)
        result = least_squares(residual, p0, method='lm', ftol=tol, xtol=tol, gtol=tol,
                              max_nfev=max_iter * len(p0))
        jones = _diag_params_to_jones_ap(result.x, n_ant, ref_ant)
    
    if verbose:
        print(f"[ALAKAZAM] Cost: {cost_init:.4e} -> {result.cost*2:.4e}")
        print(f"[ALAKAZAM] Final (polished):")
        # Print only working antennas
        for a in working[:10]:  # Use same working list from initial guess
            gx, gy = jones[a, 0, 0], jones[a, 1, 1]
            ref_mark = "[ref]" if a == ref_ant else ""
            print(f"        Ant {a:2d}: X=({np.abs(gx):.4f}, {np.degrees(np.angle(gx)):+7.1f}°)  "
                  f"Y=({np.abs(gy):.4f}, {np.degrees(np.angle(gy)):+7.1f}°)  {ref_mark}")

        if len(working) > 10:
            print(f"        ... and {len(working) - 10} more working antennas")
    
    return jones, {'cost_init': cost_init, 'cost_final': result.cost*2, 'nfev': result.nfev}


# =============================================================================
# Leakage (D) - Off-diagonal
# =============================================================================

@njit(cache=True)
def _leakage_params_to_jones(params, n_ant, ref_ant):
    """
    D-term params to Jones.
    Ref: d=0
    Others: [Re(d_XY), Im(d_XY), Re(d_YX), Im(d_YX)]
    
    J = [[1, d_XY], [d_YX, 1]]
    """
    jones = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    idx = 0
    for a in range(n_ant):
        jones[a, 0, 0] = 1.0
        jones[a, 1, 1] = 1.0
        if a != ref_ant:
            jones[a, 0, 1] = params[idx] + 1j * params[idx + 1]
            jones[a, 1, 0] = params[idx + 2] + 1j * params[idx + 3]
            idx += 4
    return jones


@njit(parallel=True, cache=True)
def _leakage_residual(params, vis_obs, vis_model, antenna1, antenna2, n_ant, ref_ant):
    """Leakage residual - uses all 4 correlations."""
    n_bl = len(antenna1)
    jones = _leakage_params_to_jones(params, n_ant, ref_ant)
    
    out = np.zeros(n_bl * 8, dtype=np.float64)
    for bl in prange(n_bl):
        Ji = jones[antenna1[bl]]
        Jj_H = _herm(jones[antenna2[bl]])
        V_pred = _mm(_mm(Ji, vis_model[bl]), Jj_H)
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


def solve_leakage(vis_obs, vis_model, antenna1, antenna2, n_ant, ref_ant=0,
                  max_iter=100, tol=1e-10, verbose=True):
    """
    Solve for leakage D-terms.
    
    Returns: jones (n_ant, 2, 2), info dict
    """
    n_bl = len(antenna1)
    
    if verbose:
        print(f"[ALAKAZAM] D solver: {n_ant} ants, {n_bl} bl, ref={ref_ant}")
    
    # Initial: d = 0
    n_params = 4 * (n_ant - 1)
    p0 = np.zeros(n_params, dtype=np.float64)
    
    vis_obs = np.ascontiguousarray(vis_obs, dtype=np.complex128)
    vis_model = np.ascontiguousarray(vis_model, dtype=np.complex128)
    antenna1 = np.ascontiguousarray(antenna1, dtype=np.int32)
    antenna2 = np.ascontiguousarray(antenna2, dtype=np.int32)
    
    def residual(p):
        return _leakage_residual(p, vis_obs, vis_model, antenna1, antenna2, n_ant, ref_ant)
    
    cost_init = np.sum(residual(p0)**2)
    result = least_squares(residual, p0, method='lm', ftol=tol, xtol=tol, gtol=tol,
                          max_nfev=max_iter * len(p0))
    
    jones = _leakage_params_to_jones(result.x, n_ant, ref_ant)
    
    if verbose:
        print(f"[ALAKAZAM] Cost: {cost_init:.4e} -> {result.cost*2:.4e}")
        print(f"[ALAKAZAM] D-terms:")
        for a in range(min(5, n_ant)):
            d_xy, d_yx = jones[a, 0, 1], jones[a, 1, 0]
            print(f"        Ant {a}: d_XY={d_xy:.4f}, d_YX={d_yx:.4f}")
    
    return jones, {'cost_init': cost_init, 'cost_final': result.cost*2, 'nfev': result.nfev}


# =============================================================================
# Cross-hand Phase (Xf)
# =============================================================================

@njit(cache=True)
def _xf_params_to_jones(params, n_ant, ref_ant):
    """
    Xf params to Jones.
    J = [[1, 0], [0, exp(iφ)]]
    """
    jones = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    idx = 0
    for a in range(n_ant):
        jones[a, 0, 0] = 1.0
        if a == ref_ant:
            jones[a, 1, 1] = 1.0
        else:
            jones[a, 1, 1] = np.exp(1j * params[idx])
            idx += 1
    return jones


@njit(parallel=True, cache=True)
def _xf_residual(params, vis_obs, vis_model, antenna1, antenna2, n_ant, ref_ant):
    """Xf residual - uses cross-hands."""
    n_bl = len(antenna1)
    jones = _xf_params_to_jones(params, n_ant, ref_ant)
    
    out = np.zeros(n_bl * 4, dtype=np.float64)
    for bl in prange(n_bl):
        Ji = jones[antenna1[bl]]
        Jj_H = _herm(jones[antenna2[bl]])
        V_pred = _mm(_mm(Ji, vis_model[bl]), Jj_H)
        
        R_xy = vis_obs[bl, 0, 1] - V_pred[0, 1]
        R_yx = vis_obs[bl, 1, 0] - V_pred[1, 0]
        
        base = bl * 4
        out[base+0] = R_xy.real
        out[base+1] = R_xy.imag
        out[base+2] = R_yx.real
        out[base+3] = R_yx.imag
    
    return out


def solve_xf(vis_obs, vis_model, antenna1, antenna2, n_ant, ref_ant=0,
             max_iter=100, tol=1e-10, verbose=True):
    """Solve for cross-hand phase Xf."""
    n_bl = len(antenna1)
    
    if verbose:
        print(f"[ALAKAZAM] Xf solver: {n_ant} ants, ref={ref_ant}")
    
    p0 = np.zeros(n_ant - 1, dtype=np.float64)
    
    vis_obs = np.ascontiguousarray(vis_obs, dtype=np.complex128)
    vis_model = np.ascontiguousarray(vis_model, dtype=np.complex128)
    antenna1 = np.ascontiguousarray(antenna1, dtype=np.int32)
    antenna2 = np.ascontiguousarray(antenna2, dtype=np.int32)
    
    def residual(p):
        return _xf_residual(p, vis_obs, vis_model, antenna1, antenna2, n_ant, ref_ant)
    
    cost_init = np.sum(residual(p0)**2)
    result = least_squares(residual, p0, method='lm', ftol=tol, xtol=tol, gtol=tol,
                          max_nfev=max_iter * len(p0))
    
    jones = _xf_params_to_jones(result.x, n_ant, ref_ant)
    
    if verbose:
        print(f"[ALAKAZAM] Cost: {cost_init:.4e} -> {result.cost*2:.4e}")
    
    return jones, {'cost_init': cost_init, 'cost_final': result.cost*2, 'nfev': result.nfev}


# =============================================================================
# Main Dispatcher
# =============================================================================

def solve_jones(jones_type, vis_obs, vis_model, antenna1, antenna2, n_ant, freq=None,
                ref_ant=0, phase_only=False, max_iter=100, tol=1e-10, verbose=True):
    """
    Main solver dispatcher.
    
    jones_type: 'K', 'B', 'G', 'D', 'Xf', 'Kcross'
    vis_obs, vis_model: (n_bl, 2, 2) or (n_bl, n_freq, 2, 2) for K
    
    Returns: jones, params, info
    """
    jt = jones_type.upper()
    
    if jt == 'K':
        if freq is None:
            raise ValueError("K solver requires freq")
        delay, info = solve_delay(vis_obs, vis_model, antenna1, antenna2, freq, n_ant,
                                  ref_ant, max_iter, tol, verbose)
        jones = delay_to_jones(delay, freq)  # (n_freq, n_ant, 2, 2)
        return jones, {'delay': delay}, info
    
    elif jt in ('G', 'B'):
        jones, info = solve_diagonal(vis_obs, vis_model, antenna1, antenna2, n_ant,
                                     ref_ant, phase_only, max_iter, tol, verbose)
        return jones, {}, info
    
    elif jt == 'D':
        jones, info = solve_leakage(vis_obs, vis_model, antenna1, antenna2, n_ant,
                                    ref_ant, max_iter, tol, verbose)
        return jones, {}, info
    
    elif jt == 'XF':
        jones, info = solve_xf(vis_obs, vis_model, antenna1, antenna2, n_ant,
                               ref_ant, max_iter, tol, verbose)
        return jones, {}, info
    
    else:
        raise ValueError(f"Unknown jones_type: {jones_type}")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Types
    'FeedBasis', 'JonesType',
    # Operations
    'jones_multiply', 'jones_inverse', 'jones_apply', 'jones_unapply', 'compute_residual',
    # Delay
    'delay_to_jones', 'solve_delay',
    # Diagonal
    'solve_diagonal',
    # Leakage
    'solve_leakage',
    # Cross-hand
    'solve_xf',
    # Main
    'solve_jones',
]
