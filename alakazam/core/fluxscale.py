"""
Fluxscale Module.

Bootstrap flux density scale from standard calibrators.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger('ALAKAZAM')


def compute_fluxscale(
    jones_ref: np.ndarray,
    jones_transfer: np.ndarray,
    freq: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Compute flux scale factors and apply to transfer gains.

    Scales transfer field gains based on reference field gains by:
    1. Time-averaging gain amplitudes for each field
    2. Computing ratio per antenna, SPW, polarization
    3. Averaging ratio over antennas per SPW, polarization
    4. Applying scale factors to transfer gains

    Parameters
    ----------
    jones_ref : ndarray
        Reference field Jones matrix
        Shape: (n_time, n_ant, 2, 2) or (n_time, n_spw, n_ant, 2, 2)
    jones_transfer : ndarray
        Transfer field Jones matrix (same shape as jones_ref)
    freq : ndarray, optional
        Frequency array for SPW centers (Hz)
    verbose : bool
        Print flux scaling results

    Returns
    -------
    jones_scaled : ndarray
        Scaled transfer Jones (same shape as jones_transfer)
    flux_results : dict
        Flux densities and scale factors per SPW and polarization
    """
    # Determine if freq-dependent
    if jones_ref.ndim == 4:
        # (n_time, n_ant, 2, 2) - freq-averaged
        has_spw = False
        n_time_ref, n_ant = jones_ref.shape[:2]
        n_time_transfer = jones_transfer.shape[0]
        n_spw = 1
    elif jones_ref.ndim == 5:
        # (n_time, n_spw, n_ant, 2, 2) - freq-dependent
        has_spw = True
        n_time_ref, n_spw, n_ant = jones_ref.shape[:3]
        n_time_transfer = jones_transfer.shape[0]
    else:
        raise ValueError(f"Unexpected Jones shape: {jones_ref.shape}")

    # Initialize results
    flux_results = {
        'spw': {},
        'freq': freq if freq is not None else None
    }

    # Compute scale factors per SPW and polarization
    scale_factors = np.ones((n_spw, 2), dtype=np.float64)  # [spw, pol]
    scale_errors = np.zeros((n_spw, 2), dtype=np.float64)

    for spw_idx in range(n_spw):
        for pol_idx in range(2):  # X and Y independently

            # Extract diagonal gains: G[:, ..., pol, pol]
            if has_spw:
                g_ref = jones_ref[:, spw_idx, :, pol_idx, pol_idx]       # (n_time, n_ant)
                g_transfer = jones_transfer[:, spw_idx, :, pol_idx, pol_idx]
            else:
                g_ref = jones_ref[:, :, pol_idx, pol_idx]                # (n_time, n_ant)
                g_transfer = jones_transfer[:, :, pol_idx, pol_idx]

            # Compute amplitudes
            amp_ref = np.abs(g_ref)           # (n_time, n_ant)
            amp_transfer = np.abs(g_transfer)

            # Time-average per antenna, ignoring NaNs
            amp_ref_mean = np.nanmean(amp_ref, axis=0)      # (n_ant,)
            amp_transfer_mean = np.nanmean(amp_transfer, axis=0)

            # Compute ratio per antenna
            ratio = amp_ref_mean / amp_transfer_mean  # (n_ant,)

            # Filter out NaNs and zeros
            valid = np.isfinite(ratio) & (ratio > 0)
            ratio_valid = ratio[valid]

            if len(ratio_valid) == 0:
                logger.warning(f"No valid gain ratios for SPW {spw_idx}, pol {pol_idx}")
                continue

            # Scale factor = mean over antennas
            scale = np.mean(ratio_valid)
            scale_err = np.std(ratio_valid) / np.sqrt(len(ratio_valid))

            scale_factors[spw_idx, pol_idx] = scale
            scale_errors[spw_idx, pol_idx] = scale_err

            # Derived flux density (if assumed 1 Jy model)
            flux = scale ** 2
            flux_err = 2 * scale * scale_err

            # Store results
            spw_key = str(spw_idx)
            if spw_key not in flux_results['spw']:
                flux_results['spw'][spw_key] = {}

            pol_name = 'X' if pol_idx == 0 else 'Y'
            flux_results['spw'][spw_key][pol_name] = {
                'scale': float(scale),
                'scale_err': float(scale_err),
                'flux': float(flux),
                'flux_err': float(flux_err),
                'n_antennas': int(len(ratio_valid))
            }

    # Apply scale factors to transfer gains
    jones_scaled = jones_transfer.copy()

    for spw_idx in range(n_spw):
        for pol_idx in range(2):
            scale = scale_factors[spw_idx, pol_idx]

            if has_spw:
                # (n_time, n_spw, n_ant, 2, 2)
                jones_scaled[:, spw_idx, :, pol_idx, pol_idx] *= scale
            else:
                # (n_time, n_ant, 2, 2)
                jones_scaled[:, :, pol_idx, pol_idx] *= scale

    # Log results
    if verbose:
        log_flux_results(flux_results)

    return jones_scaled, flux_results


def compute_fluxscale_multi_field(
    jones_dict_ref: Dict[str, np.ndarray],
    jones_dict_transfer: Dict[str, np.ndarray],
    freq: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Compute flux scale for multiple reference and transfer fields.

    For multiple reference fields, uses their mean gains.
    For multiple transfer fields, scales each independently.

    Parameters
    ----------
    jones_dict_ref : dict
        Dictionary of {field_name: jones_array} for reference fields
    jones_dict_transfer : dict
        Dictionary of {field_name: jones_array} for transfer fields
    freq : ndarray, optional
        Frequency array
    verbose : bool
        Print results

    Returns
    -------
    jones_scaled_dict : dict
        Scaled transfer Jones per field
    flux_results : dict
        Flux results per transfer field
    """
    # Stack reference field gains and average
    jones_ref_list = list(jones_dict_ref.values())

    if len(jones_ref_list) == 1:
        jones_ref_combined = jones_ref_list[0]
    else:
        # Average reference gains across fields
        jones_ref_combined = np.mean(np.stack(jones_ref_list, axis=0), axis=0)

    # Scale each transfer field
    jones_scaled_dict = {}
    flux_results_all = {}

    for field_name, jones_transfer in jones_dict_transfer.items():
        jones_scaled, flux_results = compute_fluxscale(
            jones_ref_combined,
            jones_transfer,
            freq=freq,
            verbose=False
        )

        jones_scaled_dict[field_name] = jones_scaled
        flux_results_all[field_name] = flux_results

    # Log all results
    if verbose:
        for field_name, results in flux_results_all.items():
            logger.info(f"\nFlux scaling results for transfer field: {field_name}")
            log_flux_results(results)

    return jones_scaled_dict, flux_results_all


def log_flux_results(flux_results: Dict):
    """
    Log flux scaling results in formatted table.

    Parameters
    ----------
    flux_results : dict
        Results from compute_fluxscale
    """
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()

        table = Table(title="Flux Scaling Results", show_header=True)
        table.add_column("SPW", style="cyan")
        table.add_column("Pol", style="magenta")
        table.add_column("Scale Factor", justify="right")
        table.add_column("Flux (Jy)", justify="right", style="green")
        table.add_column("N_ant", justify="right")

        for spw_id, spw_data in flux_results['spw'].items():
            for pol in ['X', 'Y']:
                if pol in spw_data:
                    data = spw_data[pol]
                    table.add_row(
                        spw_id,
                        pol,
                        f"{data['scale']:.4f} ± {data['scale_err']:.4f}",
                        f"{data['flux']:.4f} ± {data['flux_err']:.4f}",
                        str(data['n_antennas'])
                    )

        console.print(table)

    except ImportError:
        # Fallback to simple logging
        logger.info("Flux Scaling Results:")
        for spw_id, spw_data in flux_results['spw'].items():
            logger.info(f"  SPW {spw_id}:")
            for pol in ['X', 'Y']:
                if pol in spw_data:
                    data = spw_data[pol]
                    logger.info(f"    {pol}: scale={data['scale']:.4f}±{data['scale_err']:.4f}, "
                              f"flux={data['flux']:.4f}±{data['flux_err']:.4f} Jy "
                              f"(N_ant={data['n_antennas']})")


__all__ = [
    'compute_fluxscale',
    'compute_fluxscale_multi_field',
    'log_flux_results',
]
