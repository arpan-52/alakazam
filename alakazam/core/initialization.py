"""
Initial Condition Estimation via Direct Chaining.

Computes robust initial guesses for Jones matrix solvers using
the direct chain method from visibility ratios.
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger('ALAKAZAM')


def compute_baseline_quality(
    vis_obs: np.ndarray,
    flags: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute quality metric for each baseline.

    Quality = median(|V|) / MAD(|V|)
    Higher quality = more stable baseline for chaining.

    Parameters
    ----------
    vis_obs : ndarray
        Observed visibilities (n_baseline, [n_freq,] 2, 2)
    flags : ndarray, optional
        Flags (n_baseline, [n_freq,] 2, 2)

    Returns
    -------
    quality : ndarray
        Quality metric per baseline (n_baseline,)
    """
    # Get amplitudes
    if vis_obs.ndim == 4:
        # (n_baseline, n_freq, 2, 2)
        amp = np.abs(vis_obs)
    else:
        # (n_baseline, 2, 2)
        amp = np.abs(vis_obs)

    # Apply flags
    if flags is not None:
        amp = np.where(flags, np.nan, amp)

    # Compute quality per baseline
    n_baseline = vis_obs.shape[0]
    quality = np.zeros(n_baseline)

    for bl in range(n_baseline):
        amp_bl = amp[bl].ravel()
        valid = np.isfinite(amp_bl) & (amp_bl > 0)

        if np.sum(valid) < 2:
            quality[bl] = 0.0
            continue

        amp_valid = amp_bl[valid]
        median_amp = np.median(amp_valid)
        mad = np.median(np.abs(amp_valid - median_amp))

        if mad > 0:
            quality[bl] = median_amp / mad
        else:
            quality[bl] = median_amp

    return quality


def build_chain_greedy(
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    quality: np.ndarray,
    ref_ant: int,
    n_ant: int
) -> list:
    """
    Build chain from reference antenna using greedy algorithm.

    Selects highest quality baselines connecting solved to unsolved antennas.

    Parameters
    ----------
    antenna1 : ndarray
        First antenna index per baseline
    antenna2 : ndarray
        Second antenna index per baseline
    quality : ndarray
        Quality metric per baseline
    ref_ant : int
        Reference antenna index
    n_ant : int
        Total number of antennas

    Returns
    -------
    chain : list of tuples
        List of (bl_idx, ant_solved, ant_new) tuples defining chain
    """
    solved = {ref_ant}
    chain = []

    while len(solved) < n_ant:
        best_bl = None
        best_quality = -1
        best_ant_new = None
        best_ant_solved = None

        # Find best baseline connecting solved to unsolved
        for bl_idx in range(len(antenna1)):
            ant1 = antenna1[bl_idx]
            ant2 = antenna2[bl_idx]

            # Check if exactly one antenna is solved
            ant1_solved = ant1 in solved
            ant2_solved = ant2 in solved

            if ant1_solved and not ant2_solved:
                if quality[bl_idx] > best_quality:
                    best_bl = bl_idx
                    best_quality = quality[bl_idx]
                    best_ant_solved = ant1
                    best_ant_new = ant2
            elif ant2_solved and not ant1_solved:
                if quality[bl_idx] > best_quality:
                    best_bl = bl_idx
                    best_quality = quality[bl_idx]
                    best_ant_solved = ant2
                    best_ant_new = ant1

        if best_bl is None:
            # No connection found - array is disconnected
            logger.warning("Array is disconnected - cannot build complete chain")
            break

        chain.append((best_bl, best_ant_solved, best_ant_new))
        solved.add(best_ant_new)

    return chain


def apply_direct_chain(
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    jones_known: np.ndarray,
    ant_known: int,
    ant_new: int,
    bl_idx: int,
    antenna1: np.ndarray,
    antenna2: np.ndarray
) -> np.ndarray:
    """
    Apply direct chain formula to solve for new antenna Jones.

    Uses Eq. 5 from paper:
        J_j = V_ij (J_i M_ij)^{-1}†

    Parameters
    ----------
    vis_obs : ndarray
        Observed visibility (2, 2) or (n_freq, 2, 2)
    vis_model : ndarray
        Model visibility (2, 2) or (n_freq, 2, 2)
    jones_known : ndarray
        Known Jones matrix (2, 2)
    ant_known : int
        Known antenna index
    ant_new : int
        New antenna index to solve
    bl_idx : int
        Baseline index
    antenna1 : ndarray
        First antenna indices
    antenna2 : ndarray
        Second antenna indices

    Returns
    -------
    jones_new : ndarray
        Solved Jones matrix (2, 2)
    """
    # Get baseline antennas
    ant_i = antenna1[bl_idx]
    ant_j = antenna2[bl_idx]

    # Determine direction
    if ant_i == ant_known and ant_j == ant_new:
        # Forward: V_ij = J_i M_ij J_j†
        # J_j = (V_ij (J_i M_ij)^{-1})†
        forward = True
    elif ant_j == ant_known and ant_i == ant_new:
        # Backward: V_ij = J_i M_ij J_j†, but i is new, j is known
        # Need to transpose: V_ji† = J_j M_ji J_i†
        # J_i = (V_ji† (J_j M_ji)^{-1})†
        forward = False
    else:
        raise ValueError(f"Baseline {bl_idx} does not connect known={ant_known} to new={ant_new}")

    # Average over frequency if needed
    if vis_obs.ndim == 3:
        # (n_freq, 2, 2) → average
        vis_obs_avg = np.nanmean(vis_obs, axis=0)
        vis_model_avg = np.nanmean(vis_model, axis=0)
    else:
        vis_obs_avg = vis_obs
        vis_model_avg = vis_model

    # Apply chain formula
    try:
        if forward:
            # J_j = (V_ij (J_i M_ij)^{-1})†
            temp = jones_known @ vis_model_avg
            temp_inv = np.linalg.inv(temp)
            jones_new = (vis_obs_avg @ temp_inv).conj().T
        else:
            # Transpose baseline: V_ji† = J_j M_ji J_i†
            # J_i = (V_ji† (J_j M_ji)^{-1})†
            vis_obs_T = vis_obs_avg.conj().T
            vis_model_T = vis_model_avg.conj().T
            temp = jones_known @ vis_model_T
            temp_inv = np.linalg.inv(temp)
            jones_new = (vis_obs_T @ temp_inv).conj().T
    except np.linalg.LinAlgError:
        # Singular matrix - return identity
        logger.warning(f"Singular matrix in chain for antenna {ant_new}, using identity")
        jones_new = np.eye(2, dtype=np.complex128)

    return jones_new


def compute_initial_jones_chain(
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    n_ant: int,
    ref_ant: int = 0,
    flags: Optional[np.ndarray] = None,
    ref_jones: Optional[np.ndarray] = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Compute initial Jones matrices using direct chain method.

    Starting from reference antenna, propagates solution through
    all antennas using greedy chain based on baseline quality.

    Parameters
    ----------
    vis_obs : ndarray
        Observed visibilities (n_baseline, [n_freq,] 2, 2)
    vis_model : ndarray
        Model visibilities (n_baseline, [n_freq,] 2, 2)
    antenna1 : ndarray
        First antenna index per baseline
    antenna2 : ndarray
        Second antenna index per baseline
    n_ant : int
        Number of antennas
    ref_ant : int
        Reference antenna index
    flags : ndarray, optional
        Flags (n_baseline, [n_freq,] 2, 2)
    ref_jones : ndarray, optional
        Reference antenna Jones matrix (2, 2)
        If None, uses identity
    verbose : bool
        Print chain information

    Returns
    -------
    jones_init : ndarray
        Initial Jones matrices (n_ant, 2, 2)
    """
    # Initialize all Jones to identity
    jones_init = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    for ant in range(n_ant):
        jones_init[ant] = np.eye(2, dtype=np.complex128)

    # Set reference antenna
    if ref_jones is not None:
        jones_init[ref_ant] = ref_jones.copy()

    # Compute baseline quality
    quality = compute_baseline_quality(vis_obs, flags)

    # Build chain
    chain = build_chain_greedy(antenna1, antenna2, quality, ref_ant, n_ant)

    if verbose:
        logger.info(f"Built chain with {len(chain)} links from reference antenna {ref_ant}")

    # Propagate along chain
    for bl_idx, ant_known, ant_new in chain:
        try:
            jones_new = apply_direct_chain(
                vis_obs[bl_idx],
                vis_model[bl_idx],
                jones_init[ant_known],
                ant_known,
                ant_new,
                bl_idx,
                antenna1,
                antenna2
            )
            jones_init[ant_new] = jones_new

            if verbose:
                amp_x = np.abs(jones_new[0, 0])
                amp_y = np.abs(jones_new[1, 1])
                phase_x = np.angle(jones_new[0, 0], deg=True)
                phase_y = np.angle(jones_new[1, 1], deg=True)
                logger.debug(f"  Ant {ant_new}: |g_X|={amp_x:.3f}, ∠g_X={phase_x:.1f}°, "
                           f"|g_Y|={amp_y:.3f}, ∠g_Y={phase_y:.1f}°")
        except Exception as e:
            logger.warning(f"Failed to chain antenna {ant_new}: {e}")
            # Keep identity for this antenna
            continue

    return jones_init


def normalize_jones_to_reference(
    jones: np.ndarray,
    ref_ant: int,
    phase_only: bool = False
) -> np.ndarray:
    """
    Normalize Jones matrices to reference antenna constraint.

    For phase-only: sets reference phase to zero
    For amplitude+phase: keeps reference amplitude, sets phase to zero

    Parameters
    ----------
    jones : ndarray
        Jones matrices (n_ant, 2, 2)
    ref_ant : int
        Reference antenna index
    phase_only : bool
        If True, only normalize phases

    Returns
    -------
    jones_norm : ndarray
        Normalized Jones matrices
    """
    jones_norm = jones.copy()

    # Get reference phases
    phase_ref_x = np.angle(jones[ref_ant, 0, 0])
    phase_ref_y = np.angle(jones[ref_ant, 1, 1])

    # Remove reference phase from all antennas
    jones_norm[:, 0, 0] *= np.exp(-1j * phase_ref_x)
    jones_norm[:, 1, 1] *= np.exp(-1j * phase_ref_y)

    if not phase_only:
        # For amplitude calibration, reference amplitude should be reasonable
        # but we don't enforce it to be exactly 1.0 (let optimizer decide)
        pass

    return jones_norm


__all__ = [
    'compute_initial_jones_chain',
    'compute_baseline_quality',
    'build_chain_greedy',
    'apply_direct_chain',
    'normalize_jones_to_reference',
]
