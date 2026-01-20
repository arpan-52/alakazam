"""
JACKAL Calibration Engine - Orchestrates complete calibration workflow.

Per Solution Interval:
1. Load MS data
2. Identify working antennas
3. Apply MS flags + RFI flags
4. Average per solint (time/freq)
5. Call solver with chain initial guess
6. Optimize globally
7. Expand to full array with NaNs
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from scipy.optimize import least_squares
import logging

from .quality import compute_quality_metrics, print_quality_metrics

logger = logging.getLogger('jackal')


def solve_jones(
    jones_type: str,
    ms_path: str,
    ref_ant: int,
    field: Optional[str] = None,
    spw: Optional[str] = None,
    scans: Optional[str] = None,
    data_col: str = 'DATA',
    model_col: str = 'MODEL_DATA',
    time_interval: str = 'inf',
    freq_interval: str = 'full',
    rfi_enable: bool = True,
    rfi_threshold: float = 5.0,
    max_iter: int = 100,
    tol: float = 1e-10,
    pre_apply_jones: Optional[Dict[str, np.ndarray]] = None,
    verbose: bool = True,
    **solver_kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Solve for Jones matrices with complete workflow.

    Parameters
    ----------
    jones_type : str
        Jones type ('K', 'G', 'B', 'D', 'Kcross', 'Xf')
    ms_path : str
        Path to measurement set
    ref_ant : int
        Reference antenna (full index)
    field, spw, scans : str, optional
        MS selection
    data_col, model_col : str
        Column names
    time_interval, freq_interval : str
        Solution intervals
    rfi_enable : bool
        Enable RFI flagging
    rfi_threshold : float
        RFI threshold (sigma)
    max_iter : int
        Max optimization iterations
    tol : float
        Convergence tolerance
    pre_apply_jones : dict, optional
        Pre-solved Jones matrices to apply before solving
        Format: {'K': jones_K, 'G': jones_G, ...}
        Jones matrices are (n_ant_total, ...) with NaN for non-working
    verbose : bool
        Print progress
    **solver_kwargs
        Solver-specific parameters (phase_only, d_constraint, etc.)

    Returns
    -------
    jones_full : ndarray
        Jones matrices for ALL antennas (NaN for non-working)
    info : dict
        Solver information
    """
    from ..solvers import get_solver
    from ..pipeline import read_ms
    from .rfi import flag_rfi_mad
    from .averaging import average_per_baseline
    from ..solvers.utils import create_working_antenna_mapping

    # Get solver
    solver = get_solver(jones_type, verbose=verbose)

    # Determine averaging
    avg_freq = (freq_interval != 'full')
    avg_time = True  # Always average time per solint

    # Validate averaging for this Jones type
    solver.validate_averaging(avg_time, avg_freq)

    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info(f"JACKAL {jones_type} Solver")
        logger.info(f"{'='*60}")
        logger.info(f"MS: {ms_path}")
        logger.info(f"Solint: time={time_interval}, freq={freq_interval}")

    # 1. Load MS data
    if verbose:
        logger.info("Loading MS data...")

    ms_data = read_ms(
        ms_path,
        field=field,
        spw=spw,
        scans=scans,
        data_col=data_col,
        model_col=model_col
    )

    vis_obs = ms_data['vis_obs']
    vis_model = ms_data['vis_model']
    antenna1 = ms_data['antenna1']
    antenna2 = ms_data['antenna2']
    freq = ms_data['freq']
    flags_ms = ms_data['flags']
    n_ant_total = ms_data['n_ant']
    time = ms_data.get('time', np.zeros(len(antenna1)))

    if verbose:
        logger.info(f"Loaded: {vis_obs.shape[0]} rows, {vis_obs.shape[1]} chans")

    # 1.5. Apply pre-solved Jones matrices (unapply from observed data)
    if pre_apply_jones:
        if verbose:
            logger.info(f"Applying {len(pre_apply_jones)} pre-solved Jones: {list(pre_apply_jones.keys())}")

        from ..jones import jones_unapply

        # Composite all pre-Jones: J_total = J_N × ... × J_2 × J_1
        # Apply them in order
        for jones_type_pre, jones_pre in pre_apply_jones.items():
            if verbose:
                logger.info(f"  Unapplying {jones_type_pre} from observed data...")

            # Unapply: V_corrected = J^(-1) × V_obs
            # This makes the observed data as if jones_pre was not present
            vis_obs = jones_unapply(jones_pre, vis_obs, antenna1, antenna2)

    # 2. Identify working antennas
    working_ants, full_to_working, working_to_full = create_working_antenna_mapping(
        antenna1, antenna2, n_ant_total
    )
    n_working = len(working_ants)

    if ref_ant not in working_ants:
        raise ValueError(f"Reference antenna {ref_ant} has no data in baselines")

    # Filter to working antennas only
    mask_working = np.isin(antenna1, working_ants) & np.isin(antenna2, working_ants)
    vis_obs = vis_obs[mask_working]
    vis_model = vis_model[mask_working]
    flags_ms = flags_ms[mask_working]
    antenna1 = antenna1[mask_working]
    antenna2 = antenna2[mask_working]
    time = time[mask_working]

    if verbose:
        logger.info(f"Working antennas: {n_working}/{n_ant_total}")
        logger.info(f"  Antenna IDs: {working_ants.tolist()}")

    # 3. RFI flagging
    if rfi_enable:
        if verbose:
            logger.info(f"RFI flagging (threshold={rfi_threshold} sigma)...")

        flags_rfi, rfi_stats = flag_rfi_mad(
            vis_obs, antenna1, antenna2,
            threshold=rfi_threshold,
            existing_flags=flags_ms
        )
    else:
        flags_rfi = np.zeros_like(flags_ms)

    # Combine flags
    flags_combined = flags_ms | flags_rfi

    # 4. Average per baseline and solint
    if verbose:
        logger.info(f"Averaging per baseline (time={avg_time}, freq={avg_freq})...")

    vis_obs_avg, vis_model_avg, ant1_bl, ant2_bl, time_bl, flags_avg = average_per_baseline(
        vis_obs, vis_model, antenna1, antenna2, time, flags_combined,
        avg_time=avg_time, avg_freq=avg_freq
    )

    n_bl = len(ant1_bl)

    if verbose:
        logger.info(f"  Result: {n_bl} baselines")

    # Remap antenna indices to working set
    ant1_work = np.array([full_to_working[a] for a in ant1_bl], dtype=np.int32)
    ant2_work = np.array([full_to_working[a] for a in ant2_bl], dtype=np.int32)
    ref_work = full_to_working[ref_ant]

    # Print solver info
    n_freq = vis_obs_avg.shape[1] if vis_obs_avg.ndim > 2 else 1
    solver.print_solver_info(
        n_working, n_ant_total, n_bl, n_freq, ref_ant,
        time_interval, freq_interval
    )

    # 5. Chain initial guess
    if verbose:
        logger.info("Computing chain initial guess...")

    params_init = solver.chain_initial_guess(
        vis_obs_avg, vis_model_avg, ant1_work, ant2_work, freq,
        flags_avg, ref_work, n_working, **solver_kwargs
    )

    # Pack to optimization parameters
    p0 = solver.pack_params(params_init, ref_work, **solver_kwargs)

    if verbose:
        logger.info(f"Initial parameters: {len(p0)} values")

    # 6. Optimize
    if verbose:
        logger.info("Optimizing...")

    def residual(p):
        return solver.residual(
            p, vis_obs_avg, vis_model_avg, ant1_work, ant2_work, freq,
            flags_avg, n_working, ref_work, **solver_kwargs
        )

    cost_init = np.sum(residual(p0)**2)

    result = least_squares(
        residual, p0, method='lm',
        ftol=tol, xtol=tol, gtol=tol,
        max_nfev=max_iter * max(1, len(p0))
    )

    cost_final = result.cost * 2

    # Unpack results (working antennas only)
    jones_working = solver.unpack_params(result.x, n_working, ref_work, **solver_kwargs)

    # Count data points and parameters for quality metrics
    n_total_vis = flags_avg.size
    n_unflagged_vis = np.sum(~flags_avg)
    n_parameters = len(p0)

    # Compute quality metrics
    quality = compute_quality_metrics(
        cost_init=cost_init,
        cost_final=cost_final,
        n_data_points=n_unflagged_vis,
        n_parameters=n_parameters,
        convergence_info={'nfev': result.nfev, 'success': result.success},
        n_total_vis=n_total_vis
    )

    # Convergence info (include quality metrics)
    convergence_info = {
        'cost_init': cost_init,
        'cost_final': cost_final,
        'nfev': result.nfev,
        'success': result.success,
        'n_working_ants': n_working,
        'quality': quality
    }

    # Print solution
    solver.print_solution(jones_working, working_ants, ref_ant, convergence_info)

    # Print quality metrics
    if verbose:
        print_quality_metrics(quality, verbose=True)

    # 7. Expand to full array with NaNs
    if jones_working.ndim == 2:
        # K delays: (n_working, 2)
        jones_full = np.full((n_ant_total, jones_working.shape[1]), np.nan, dtype=jones_working.dtype)
    elif jones_working.ndim == 3:
        # G gains: (n_working, 2, 2)
        jones_full = np.full((n_ant_total, 2, 2), np.nan, dtype=jones_working.dtype)
    elif jones_working.ndim == 4:
        # B bandpass: (n_working, n_freq, 2, 2)
        jones_full = np.full((n_ant_total, jones_working.shape[1], 2, 2), np.nan, dtype=jones_working.dtype)
    else:
        raise ValueError(f"Unexpected Jones shape: {jones_working.shape}")

    # Fill in working antennas
    for i in range(n_working):
        ant_full = working_to_full[i]
        jones_full[ant_full] = jones_working[i]

    if verbose:
        logger.info(f"\n{'='*60}\n")

    return jones_full, convergence_info


__all__ = ['solve_jones']
