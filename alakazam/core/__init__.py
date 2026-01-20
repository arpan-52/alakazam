"""
JACKAL Core Solver.

Handles:
- RFI rejection per baseline (MAD)
- Averaging over time/freq solint
- Dispatches to appropriate Jones solver
"""

import numpy as np
import numba
from numba import njit, prange
from typing import Tuple, Dict, Optional, List
import warnings
from ..jones import solve_jones as _solve_jones, jones_unapply, delay_to_jones


# =============================================================================
# RFI Rejection (MAD per baseline)
# =============================================================================

@njit(cache=True)
def _median(arr):
    """Compute median."""
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    if n % 2 == 0:
        return 0.5 * (sorted_arr[n//2 - 1] + sorted_arr[n//2])
    return sorted_arr[n//2]


@njit(parallel=True, cache=True)
def flag_rfi_mad(vis, existing_flags, sigma=5.0):
    """
    Flag RFI using MAD per baseline.
    
    vis: (n_bl, 2, 2) or (n_bl, n_freq, 2, 2) complex
    existing_flags: same shape, bool
    sigma: threshold
    
    Returns: flags (same shape, bool)
    """
    shape = vis.shape
    flags = existing_flags.copy()
    
    if vis.ndim == 3:
        # (n_bl, 2, 2)
        n_bl = shape[0]
        for pol_i in range(2):
            for pol_j in range(2):
                amps = np.abs(vis[:, pol_i, pol_j])
                
                # Get unflagged
                unflagged = []
                for bl in range(n_bl):
                    if not flags[bl, pol_i, pol_j]:
                        unflagged.append(amps[bl])
                
                if len(unflagged) < 3:
                    continue
                
                unflagged_arr = np.array(unflagged)
                med = _median(unflagged_arr)
                mad = _median(np.abs(unflagged_arr - med))
                
                if mad < 1e-10:
                    continue
                
                thresh = sigma * mad * 1.4826
                for bl in range(n_bl):
                    if np.abs(amps[bl] - med) > thresh:
                        flags[bl, pol_i, pol_j] = True
    else:
        # (n_bl, n_freq, 2, 2)
        n_bl, n_freq = shape[0], shape[1]
        for f in prange(n_freq):
            for pol_i in range(2):
                for pol_j in range(2):
                    amps = np.abs(vis[:, f, pol_i, pol_j])
                    
                    unflagged = []
                    for bl in range(n_bl):
                        if not flags[bl, f, pol_i, pol_j]:
                            unflagged.append(amps[bl])
                    
                    if len(unflagged) < 3:
                        continue
                    
                    unflagged_arr = np.array(unflagged)
                    med = _median(unflagged_arr)
                    mad = _median(np.abs(unflagged_arr - med))
                    
                    if mad < 1e-10:
                        continue
                    
                    thresh = sigma * mad * 1.4826
                    for bl in range(n_bl):
                        if np.abs(amps[bl] - med) > thresh:
                            flags[bl, f, pol_i, pol_j] = True
    
    return flags


# =============================================================================
# Averaging
# =============================================================================

@njit(parallel=True, cache=True)
def average_visibilities(vis, flags):
    """
    Average visibilities per baseline, respecting flags.
    
    vis: (n_row, 2, 2) complex
    flags: (n_row, 2, 2) bool
    
    Returns: (2, 2) complex
    """
    n_row = vis.shape[0]
    result = np.zeros((2, 2), dtype=np.complex128)
    counts = np.zeros((2, 2), dtype=np.float64)
    
    for row in range(n_row):
        for i in range(2):
            for j in range(2):
                if not flags[row, i, j]:
                    result[i, j] += vis[row, i, j]
                    counts[i, j] += 1.0
    
    for i in range(2):
        for j in range(2):
            if counts[i, j] > 0:
                result[i, j] /= counts[i, j]
    
    return result


@njit(parallel=True, cache=True) 
def average_per_baseline(vis, flags, antenna1, antenna2, n_ant):
    """
    Average visibilities per baseline across all rows.
    
    vis: (n_row, 2, 2) complex
    flags: (n_row, 2, 2) bool
    antenna1, antenna2: (n_row,) int
    
    Returns:
        vis_avg: (n_bl_unique, 2, 2) complex
        flags_avg: (n_bl_unique, 2, 2) bool
        ant1_out: (n_bl_unique,) int
        ant2_out: (n_bl_unique,) int
    """
    n_row = vis.shape[0]
    n_bl_max = n_ant * (n_ant - 1) // 2
    
    # Accumulators
    vis_sum = np.zeros((n_bl_max, 2, 2), dtype=np.complex128)
    counts = np.zeros((n_bl_max, 2, 2), dtype=np.float64)
    bl_map = -np.ones((n_ant, n_ant), dtype=np.int32)
    
    ant1_list = np.zeros(n_bl_max, dtype=np.int32)
    ant2_list = np.zeros(n_bl_max, dtype=np.int32)
    n_bl = 0
    
    # Map baselines
    for row in range(n_row):
        a1, a2 = antenna1[row], antenna2[row]
        if a1 > a2:
            a1, a2 = a2, a1
        
        if bl_map[a1, a2] < 0:
            bl_map[a1, a2] = n_bl
            ant1_list[n_bl] = a1
            ant2_list[n_bl] = a2
            n_bl += 1
    
    # Accumulate
    for row in range(n_row):
        a1, a2 = antenna1[row], antenna2[row]
        if a1 > a2:
            a1, a2 = a2, a1
        
        bl_idx = bl_map[a1, a2]
        for i in range(2):
            for j in range(2):
                if not flags[row, i, j]:
                    vis_sum[bl_idx, i, j] += vis[row, i, j]
                    counts[bl_idx, i, j] += 1.0
    
    # Average
    vis_avg = np.zeros((n_bl, 2, 2), dtype=np.complex128)
    flags_avg = np.zeros((n_bl, 2, 2), dtype=np.bool_)
    
    for bl in range(n_bl):
        for i in range(2):
            for j in range(2):
                if counts[bl, i, j] > 0:
                    vis_avg[bl, i, j] = vis_sum[bl, i, j] / counts[bl, i, j]
                else:
                    flags_avg[bl, i, j] = True
    
    return vis_avg, flags_avg, ant1_list[:n_bl], ant2_list[:n_bl]


@njit(parallel=True, cache=True)
def average_per_baseline_freq(vis, flags, antenna1, antenna2, n_ant):
    """
    Average visibilities per baseline, keeping frequency axis.
    
    vis: (n_row, n_freq, 2, 2) complex
    flags: (n_row, n_freq, 2, 2) bool
    
    Returns:
        vis_avg: (n_bl_unique, n_freq, 2, 2) complex
        flags_avg: (n_bl_unique, n_freq, 2, 2) bool
        ant1_out, ant2_out
    """
    n_row, n_freq = vis.shape[0], vis.shape[1]
    n_bl_max = n_ant * (n_ant - 1) // 2
    
    vis_sum = np.zeros((n_bl_max, n_freq, 2, 2), dtype=np.complex128)
    counts = np.zeros((n_bl_max, n_freq, 2, 2), dtype=np.float64)
    bl_map = -np.ones((n_ant, n_ant), dtype=np.int32)
    
    ant1_list = np.zeros(n_bl_max, dtype=np.int32)
    ant2_list = np.zeros(n_bl_max, dtype=np.int32)
    n_bl = 0
    
    for row in range(n_row):
        a1, a2 = antenna1[row], antenna2[row]
        if a1 > a2:
            a1, a2 = a2, a1
        if bl_map[a1, a2] < 0:
            bl_map[a1, a2] = n_bl
            ant1_list[n_bl] = a1
            ant2_list[n_bl] = a2
            n_bl += 1
    
    for row in prange(n_row):
        a1, a2 = antenna1[row], antenna2[row]
        if a1 > a2:
            a1, a2 = a2, a1
        bl_idx = bl_map[a1, a2]
        
        for f in range(n_freq):
            for i in range(2):
                for j in range(2):
                    if not flags[row, f, i, j]:
                        vis_sum[bl_idx, f, i, j] += vis[row, f, i, j]
                        counts[bl_idx, f, i, j] += 1.0
    
    vis_avg = np.zeros((n_bl, n_freq, 2, 2), dtype=np.complex128)
    flags_avg = np.zeros((n_bl, n_freq, 2, 2), dtype=np.bool_)
    
    for bl in prange(n_bl):
        for f in range(n_freq):
            for i in range(2):
                for j in range(2):
                    if counts[bl, f, i, j] > 0:
                        vis_avg[bl, f, i, j] = vis_sum[bl, f, i, j] / counts[bl, f, i, j]
                    else:
                        flags_avg[bl, f, i, j] = True
    
    return vis_avg, flags_avg, ant1_list[:n_bl], ant2_list[:n_bl]


# =============================================================================
# Main Solver
# =============================================================================

def solve(
    jones_type: str,
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    n_ant: int,
    freq: np.ndarray = None,
    flags: np.ndarray = None,
    pre_jones: List[np.ndarray] = None,
    ref_ant: int = 0,
    phase_only: bool = False,
    rfi_sigma: float = 5.0,
    max_iter: int = 100,
    tol: float = 1e-10,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict, Dict]:
    """
    Main JACKAL solver.
    
    Parameters
    ----------
    jones_type : str
        'K', 'B', 'G', 'D', 'Xf'
    vis_obs : (n_row, 2, 2) or (n_row, n_freq, 2, 2) complex
        Observed visibilities
    vis_model : same shape
        Model visibilities
    antenna1, antenna2 : (n_row,) int
        Antenna indices
    n_ant : int
    freq : (n_freq,) float, optional
        Frequencies in Hz (required for K)
    flags : same shape as vis, optional
        Existing flags
    pre_jones : list of (n_ant, 2, 2) arrays, optional
        Pre-solved Jones to apply before solving
    ref_ant : int
        Reference antenna
    phase_only : bool
        For G/B: phase-only mode
    rfi_sigma : float
        MAD threshold for RFI rejection
    max_iter : int
    tol : float
    verbose : bool
    
    Returns
    -------
    jones : (n_ant, 2, 2) or (n_freq, n_ant, 2, 2) complex
    params : dict
        Native parameters (e.g., delay for K)
    info : dict
        Solver info
    """
    jt = jones_type.upper()
    
    # Ensure contiguous
    vis_obs = np.ascontiguousarray(vis_obs, dtype=np.complex128)
    vis_model = np.ascontiguousarray(vis_model, dtype=np.complex128)
    antenna1 = np.ascontiguousarray(antenna1, dtype=np.int32)
    antenna2 = np.ascontiguousarray(antenna2, dtype=np.int32)
    
    # Initialize flags
    if flags is None:
        flags = np.zeros(vis_obs.shape, dtype=np.bool_)
    else:
        flags = np.ascontiguousarray(flags, dtype=np.bool_)
    
    # Verbose logging is now handled by pipeline
    # This suppresses the redundant "JACKAL Solver" header
    
    # Apply pre-solved Jones
    if pre_jones:
        if verbose:
            print(f"[ALAKAZAM] Applying {len(pre_jones)} pre-solved Jones")

        # Composite - handle both freq-averaged and freq-dependent
        J_pre = pre_jones[0].copy()
        for J in pre_jones[1:]:
            # Handle multiplication based on dimensionality
            if J_pre.ndim == 4 and J.ndim == 4:
                # Both freq-dependent: (n_ant, n_freq, 2, 2)
                # Multiply per frequency channel
                n_ant, n_freq = J_pre.shape[:2]
                J_composite = np.zeros_like(J_pre)
                for f in range(n_freq):
                    from ..jones import jones_multiply
                    J_composite[:, f, :, :] = jones_multiply(J[:, f, :, :], J_pre[:, f, :, :])
                J_pre = J_composite
            elif J_pre.ndim == 3 and J.ndim == 3:
                # Both freq-averaged: (n_ant, 2, 2)
                from ..jones import jones_multiply
                J_pre = jones_multiply(J, J_pre)
            else:
                # Mixed - shouldn't happen with proper interpolation, but handle it
                raise ValueError(f"Mismatched Jones dimensions: {J_pre.shape} vs {J.shape}")

        # Unapply from observed - handle both cases
        if J_pre.ndim == 4:
            # Frequency-dependent: (n_ant, n_freq, 2, 2)
            # Need to transpose for _unapply_jones_freq_dependent which expects (n_freq, n_ant, 2, 2)
            from ..jones.delay import _unapply_jones_freq_dependent
            J_pre_transposed = np.transpose(J_pre, (1, 0, 2, 3))  # (n_ant, n_freq, 2, 2) → (n_freq, n_ant, 2, 2)
            vis_obs = _unapply_jones_freq_dependent(J_pre_transposed, vis_obs, antenna1, antenna2)
        else:
            # Frequency-averaged: (n_ant, 2, 2)
            vis_obs = jones_unapply(J_pre, vis_obs, antenna1, antenna2)
    
    # RFI flagging
    if verbose:
        n_flagged_before = np.sum(flags)

    flags = flag_rfi_mad(vis_obs, flags, rfi_sigma)

    if verbose:
        n_flagged_after = np.sum(flags)
        pct = 100.0 * n_flagged_after / flags.size
        print(f"[ALAKAZAM] RFI flagged: {n_flagged_after - n_flagged_before} new ({pct:.1f}% total)")

    # Check if chunk is fully flagged
    if np.all(flags):
        if verbose:
            print(f"[ALAKAZAM] WARNING: Chunk is fully flagged - returning NaN solution")

        # Return NaN Jones based on expected shape
        if jt == 'K':
            if freq is None:
                raise ValueError("K solver requires freq")
            jones = np.full((len(freq), n_ant, 2, 2), np.nan, dtype=np.complex128)
        elif freq is not None and vis_obs.ndim == 4:
            jones = np.full((len(freq), n_ant, 2, 2), np.nan, dtype=np.complex128)
        else:
            jones = np.full((n_ant, 2, 2), np.nan, dtype=np.complex128)

        return jones, {}, {'cost_init': 0.0, 'cost_final': 0.0, 'nfev': 0, 'fully_flagged': True}
    
    # Average and solve based on type
    if jt == 'K':
        # K: average over time, keep freq
        if freq is None:
            raise ValueError("K solver requires freq")
        freq = np.ascontiguousarray(freq, dtype=np.float64)

        vis_avg, flags_avg, ant1, ant2 = average_per_baseline_freq(
            vis_obs, flags, antenna1, antenna2, n_ant
        )
        model_avg, _, _, _ = average_per_baseline_freq(
            vis_model, flags, antenna1, antenna2, n_ant
        )

        if verbose:
            print(f"[ALAKAZAM] Averaged: {vis_obs.shape[0]} rows -> {vis_avg.shape[0]} baselines")

        jones, params, info = _solve_jones(
            jt, vis_avg, model_avg, ant1, ant2, n_ant, freq,
            ref_ant, phase_only, max_iter, tol, verbose
        )

    elif freq is not None and vis_obs.ndim == 4:
        # Frequency-dependent solution (e.g., B with freq_interval='2MHz')
        # Keep frequency axis, average over time only
        freq = np.ascontiguousarray(freq, dtype=np.float64)

        vis_avg, flags_avg, ant1, ant2 = average_per_baseline_freq(
            vis_obs, flags, antenna1, antenna2, n_ant
        )
        model_avg, _, _, _ = average_per_baseline_freq(
            vis_model, flags, antenna1, antenna2, n_ant
        )

        if verbose:
            n_freq = vis_avg.shape[1]
            print(f"[ALAKAZAM] Averaged: {vis_obs.shape[0]} rows -> {vis_avg.shape[0]} baselines, {n_freq} freq channels")

        jones, params, info = _solve_jones(
            jt, vis_avg, model_avg, ant1, ant2, n_ant, freq,
            ref_ant, phase_only, max_iter, tol, verbose
        )

    else:
        # Frequency-averaged solution (G, D, Xf with freq_interval='full')
        # Average over time and freq
        if vis_obs.ndim == 4:
            # Average over freq first - mask flagged data and use nanmean
            vis_obs_masked = np.where(flags, np.nan, vis_obs)
            model_masked = np.where(flags, np.nan, vis_model)

            # Suppress warnings for empty slices (happens when all freq channels are flagged)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
                vis_freq_avg = np.nanmean(vis_obs_masked, axis=1)
                model_freq_avg = np.nanmean(model_masked, axis=1)

            # Flag if ALL channels were flagged
            flags_freq_avg = np.all(flags, axis=1)

            # Log flagging statistics
            if verbose:
                n_total_samples = vis_freq_avg.size
                n_flagged_samples = np.sum(flags_freq_avg)
                pct_flagged = 100.0 * n_flagged_samples / n_total_samples
                print(f"[ALAKAZAM] After freq avg: {n_flagged_samples}/{n_total_samples} samples flagged ({pct_flagged:.1f}%)")
        else:
            vis_freq_avg = vis_obs
            model_freq_avg = vis_model
            flags_freq_avg = flags

        vis_avg, flags_avg, ant1, ant2 = average_per_baseline(
            vis_freq_avg, flags_freq_avg, antenna1, antenna2, n_ant
        )
        model_avg, _, _, _ = average_per_baseline(
            model_freq_avg, flags_freq_avg, antenna1, antenna2, n_ant
        )

        # Log final flagging after baseline averaging
        if verbose:
            n_bl = len(ant1)
            n_total_bl = vis_avg.size
            n_flagged_bl = np.sum(flags_avg)
            pct_flagged_bl = 100.0 * n_flagged_bl / n_total_bl
            print(f"[ALAKAZAM] After baseline avg: {n_flagged_bl}/{n_total_bl} baseline points flagged ({pct_flagged_bl:.1f}%)")

        if verbose:
            print(f"[ALAKAZAM] Averaged: {vis_obs.shape[0]} rows -> {vis_avg.shape[0]} baselines")

        jones, params, info = _solve_jones(
            jt, vis_avg, model_avg, ant1, ant2, n_ant, None,
            ref_ant, phase_only, max_iter, tol, verbose
        )
    
    if verbose:
        print(f"{'='*60}\n")
    
    return jones, params, info


# Import new architecture modules
from .metadata import detect_ms_metadata, detect_non_working_antennas_chunked, MSMetadata
from .chunking import iterate_chunks, DataChunk
from .interpolation import interpolate_jones_to_chunk, interpolate_jones_time, interpolate_jones_freq
from .rfi import flag_rfi_mad as flag_rfi_mad_v2, apply_flags
from .averaging import average_time_freq, average_per_baseline as average_per_baseline_v2, average_visibilities as average_visibilities_v2, average_model_visibilities
from .solver import CalibrationSolver, SolverConfig, SolverResult
from .logging_utils import format_solution_table, log_convergence_summary, log_rfi_summary, log_progress
from .apply import apply_calibration
from .engine import solve_jones
from .resources import get_available_ram, estimate_ms_selection_size, should_use_chunking, cleanup_arrays
from .quality import SolutionQuality, compute_quality_metrics, print_quality_metrics


__all__ = [
    'solve',
    'flag_rfi_mad',
    'average_visibilities',
    'average_per_baseline',
    'average_per_baseline_freq',
    # New architecture
    'solve_jones',
    'CalibrationSolver',
    'SolverConfig',
    'SolverResult',
    'detect_ms_metadata',
    'detect_non_working_antennas_chunked',
    'MSMetadata',
    'iterate_chunks',
    'DataChunk',
    'interpolate_jones_to_chunk',
    'flag_rfi_mad_v2',
    'average_time_freq',
    'average_per_baseline_v2',
    'average_visibilities_v2',
    'average_model_visibilities',
    'format_solution_table',
    'log_convergence_summary',
    'apply_calibration',
    # Resources and quality
    'get_available_ram',
    'estimate_ms_selection_size',
    'should_use_chunking',
    'cleanup_arrays',
    'SolutionQuality',
    'compute_quality_metrics',
    'print_quality_metrics',
]
