"""
Enhanced HDF5 I/O for calibration solutions.

Stores Jones matrices with comprehensive metadata including:
- Solution intervals (time/freq)
- Working antennas
- Convergence statistics
- RFI statistics
- Feed basis
- Solver configuration
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Union, Dict
from datetime import datetime
import logging

logger = logging.getLogger('jackal')


def write_solution_hdf5(filename: Union[str, Path], solver_result, overwrite: bool = False) -> None:
    """
    Write calibration solution to HDF5 file with comprehensive metadata.

    Parameters
    ----------
    filename : str or Path
        Output HDF5 filename
    solver_result : SolverResult
        Result from CalibrationSolver
    overwrite : bool
        Overwrite existing file
    """
    from ..core.solver import SolverResult  # Import here to avoid circular dependency

    filename = Path(filename)

    if filename.exists() and not overwrite:
        raise FileExistsError(f"File exists: {filename}. Use overwrite=True to replace.")

    logger.info(f"Writing solution to HDF5: {filename}")

    with h5py.File(filename, 'w') as f:
        # Create groups
        jones_grp = f.create_group('jones')
        metadata_grp = f.create_group('metadata')
        config_grp = f.create_group('config')
        convergence_grp = f.create_group('convergence')
        rfi_grp = f.create_group('rfi')

        # Write Jones solutions
        jones_grp.create_dataset('solutions', data=solver_result.jones, compression='gzip')
        jones_grp.attrs['shape_description'] = '(n_sol_time, n_sol_freq, n_ant, 2, 2)'

        # Write metadata
        meta = solver_result.metadata
        metadata_grp.attrs['n_ant'] = meta.n_ant
        metadata_grp.attrs['n_bl'] = meta.n_bl
        metadata_grp.attrs['n_time'] = meta.n_time
        metadata_grp.attrs['n_freq'] = meta.n_freq
        metadata_grp.attrs['n_sol_time'] = meta.n_sol_time
        metadata_grp.attrs['n_sol_freq'] = meta.n_sol_freq
        metadata_grp.attrs['time_min'] = meta.time_min
        metadata_grp.attrs['time_max'] = meta.time_max
        metadata_grp.attrs['freq_min'] = meta.freq_min
        metadata_grp.attrs['freq_max'] = meta.freq_max
        metadata_grp.attrs['sol_time_seconds'] = meta.sol_time_seconds
        metadata_grp.attrs['sol_freq_channels'] = meta.sol_freq_channels
        metadata_grp.attrs['sol_freq_hz'] = meta.sol_freq_hz

        # Write solution intervals
        metadata_grp.create_dataset('time_intervals', data=meta.time_intervals)
        metadata_grp.create_dataset('freq_intervals', data=meta.freq_intervals)
        metadata_grp.create_dataset('frequencies', data=meta.frequencies)
        metadata_grp.create_dataset('antenna1', data=meta.antenna1)
        metadata_grp.create_dataset('antenna2', data=meta.antenna2)

        # Write working antennas
        metadata_grp.create_dataset('working_antennas', data=solver_result.working_ants)

        # Write configuration
        cfg = solver_result.config
        config_grp.attrs['jones_type'] = cfg.jones_type
        config_grp.attrs['solint_time'] = cfg.solint_time
        config_grp.attrs['solint_freq'] = cfg.solint_freq
        config_grp.attrs['ref_ant'] = cfg.ref_ant
        config_grp.attrs['feed_basis'] = cfg.feed_basis
        config_grp.attrs['rfi_enable'] = cfg.rfi_enable
        config_grp.attrs['rfi_threshold'] = cfg.rfi_threshold
        config_grp.attrs['max_iter'] = cfg.max_iter
        config_grp.attrs['tol'] = cfg.tol

        # Write convergence info
        convergence_grp.create_dataset('costs_init', data=solver_result.costs_init)
        convergence_grp.create_dataset('costs_final', data=solver_result.costs_final)
        convergence_grp.create_dataset('success_flags', data=solver_result.success_flags)
        convergence_grp.create_dataset('nfev_counts', data=solver_result.nfev_counts)

        # Convergence statistics
        n_success = np.sum(solver_result.success_flags)
        n_total = solver_result.success_flags.size
        convergence_grp.attrs['n_success'] = int(n_success)
        convergence_grp.attrs['n_total'] = int(n_total)
        convergence_grp.attrs['success_rate'] = float(n_success) / n_total if n_total > 0 else 0.0

        cost_reduction = solver_result.costs_init / np.maximum(solver_result.costs_final, 1e-20)
        valid_reductions = cost_reduction[solver_result.success_flags & np.isfinite(cost_reduction)]
        if len(valid_reductions) > 0:
            convergence_grp.attrs['median_cost_reduction'] = float(np.median(valid_reductions))
            convergence_grp.attrs['mean_cost_reduction'] = float(np.mean(valid_reductions))

        # Write RFI statistics
        rfi_stats = solver_result.rfi_stats
        rfi_grp.attrs['total_samples'] = rfi_stats.get('total_samples', 0)
        rfi_grp.attrs['total_flagged'] = rfi_stats.get('total_flagged', 0)
        if rfi_stats.get('total_samples', 0) > 0:
            frac = rfi_stats['total_flagged'] / rfi_stats['total_samples']
            rfi_grp.attrs['fraction_flagged'] = float(frac)

        # Global attributes
        f.attrs['creation_time'] = datetime.now().isoformat()
        f.attrs['jackal_version'] = '2.0.0'
        f.attrs['file_format_version'] = '2.0'

    logger.info(f"  Written: jones={solver_result.jones.shape}, metadata, convergence, RFI stats")


def read_solution_hdf5(filename: Union[str, Path]) -> Dict:
    """
    Read calibration solution from HDF5 file.

    Parameters
    ----------
    filename : str or Path
        Input HDF5 filename

    Returns
    -------
    solution : dict
        Dictionary containing:
        - 'jones': Jones matrices
        - 'metadata': MS metadata
        - 'config': Solver configuration
        - 'convergence': Convergence information
        - 'rfi_stats': RFI statistics
    """
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    logger.info(f"Reading solution from HDF5: {filename}")

    solution = {}

    with h5py.File(filename, 'r') as f:
        # Read Jones solutions
        solution['jones'] = f['jones/solutions'][:]

        # Read metadata
        meta = {}
        for key, value in f['metadata'].attrs.items():
            meta[key] = value

        # Read datasets
        for dset_name in ['time_intervals', 'freq_intervals', 'frequencies', 'antenna1', 'antenna2', 'working_antennas']:
            if dset_name in f['metadata']:
                meta[dset_name] = f[f'metadata/{dset_name}'][:]

        solution['metadata'] = meta

        # Read configuration
        config = {}
        for key, value in f['config'].attrs.items():
            config[key] = value
        solution['config'] = config

        # Read convergence
        convergence = {}
        if 'costs_init' in f['convergence']:
            convergence['costs_init'] = f['convergence/costs_init'][:]
        if 'costs_final' in f['convergence']:
            convergence['costs_final'] = f['convergence/costs_final'][:]
        if 'success_flags' in f['convergence']:
            convergence['success_flags'] = f['convergence/success_flags'][:]
        if 'nfev_counts' in f['convergence']:
            convergence['nfev_counts'] = f['convergence/nfev_counts'][:]

        for key, value in f['convergence'].attrs.items():
            convergence[key] = value

        solution['convergence'] = convergence

        # Read RFI stats
        rfi_stats = {}
        for key, value in f['rfi'].attrs.items():
            rfi_stats[key] = value
        solution['rfi_stats'] = rfi_stats

        # Read global attributes
        solution['creation_time'] = f.attrs.get('creation_time', 'unknown')
        solution['jackal_version'] = f.attrs.get('jackal_version', 'unknown')

    logger.info(f"  Read: jones={solution['jones'].shape}")

    return solution


__all__ = ['write_solution_hdf5', 'read_solution_hdf5']
