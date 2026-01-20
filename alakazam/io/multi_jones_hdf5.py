"""
Multi-Jones HDF5 I/O for new solver architecture.

Saves/loads multiple Jones solutions (K, G, B, D, etc.) from sequential solving
to a single HDF5 file with complete metadata.

Format:
  /K/jones              - Jones matrices (n_ant, 2) or (n_ant, n_freq, 2, 2)
  /K/convergence/...    - Convergence info
  /K/metadata/...       - MS metadata, working antennas, etc.
  /G/...                - Same structure for G
  /B/...                - Same structure for B
  /config               - Overall config (ms_path, ref_ant, etc.)
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, List
import logging

logger = logging.getLogger('jackal')


def save_multi_jones_hdf5(
    filename: Union[str, Path],
    jones_solutions: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    overwrite: bool = False
) -> None:
    """
    Save multiple Jones solutions to HDF5 file.

    Parameters
    ----------
    filename : str or Path
        Output HDF5 filename
    jones_solutions : dict
        Dictionary mapping Jones type → solution dict:
        {
            'K': {'jones': ndarray, 'info': dict, 'working_ants': ndarray, ...},
            'G': {'jones': ndarray, 'info': dict, 'working_ants': ndarray, ...},
            ...
        }
    config : dict
        Overall configuration (ms_path, ref_ant, field, spw, etc.)
    overwrite : bool
        Overwrite existing file

    Notes
    -----
    jones arrays have NaN for non-working antennas.
    working_ants contains full antenna indices of working antennas.
    """
    filename = Path(filename)

    if filename.exists() and not overwrite:
        raise FileExistsError(f"File exists: {filename}. Use overwrite=True.")

    logger.info(f"Writing multi-Jones solution to: {filename}")
    logger.info(f"  Jones types: {list(jones_solutions.keys())}")

    with h5py.File(filename, 'w') as f:
        # Global config
        config_grp = f.create_group('config')
        config_grp.attrs['ms_path'] = str(config.get('ms_path', ''))
        config_grp.attrs['ref_ant'] = config.get('ref_ant', 0)
        config_grp.attrs['field'] = str(config.get('field', ''))
        config_grp.attrs['spw'] = str(config.get('spw', ''))
        config_grp.attrs['scans'] = str(config.get('scans', ''))
        config_grp.attrs['data_col'] = config.get('data_col', 'DATA')
        config_grp.attrs['model_col'] = config.get('model_col', 'MODEL_DATA')

        # For each Jones type
        for jones_type, solution in jones_solutions.items():
            logger.info(f"  Writing {jones_type}...")

            jones_grp = f.create_group(jones_type)

            # Jones matrices (with NaN for non-working antennas)
            jones_data = solution['jones']
            jones_grp.create_dataset('jones', data=jones_data, compression='gzip')
            jones_grp['jones'].attrs['shape_description'] = str(jones_data.shape)
            jones_grp['jones'].attrs['dtype'] = str(jones_data.dtype)

            # Working antennas
            if 'working_ants' in solution:
                working_ants = solution['working_ants']
                jones_grp.create_dataset('working_antennas', data=working_ants)
                jones_grp.attrs['n_working'] = len(working_ants)

            # Convergence info
            if 'info' in solution:
                conv_grp = jones_grp.create_group('convergence')
                info = solution['info']
                conv_grp.attrs['cost_init'] = info.get('cost_init', np.nan)
                conv_grp.attrs['cost_final'] = info.get('cost_final', np.nan)
                conv_grp.attrs['nfev'] = info.get('nfev', 0)
                conv_grp.attrs['success'] = info.get('success', False)
                conv_grp.attrs['n_working_ants'] = info.get('n_working_ants', 0)

            # Metadata (frequencies, etc.)
            if 'freq' in solution:
                meta_grp = jones_grp.create_group('metadata')
                meta_grp.create_dataset('frequencies', data=solution['freq'])

        f.attrs['creation_date'] = str(np.datetime64('now'))
        f.attrs['jackal_version'] = '1.0'
        f.attrs['n_jones_types'] = len(jones_solutions)

    logger.info(f"✓ Saved {len(jones_solutions)} Jones types to {filename}")


def load_multi_jones_hdf5(
    filename: Union[str, Path],
    jones_types: List[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Load multiple Jones solutions from HDF5 file.

    Parameters
    ----------
    filename : str or Path
        HDF5 file to read
    jones_types : list of str, optional
        Specific Jones types to load (default: all)

    Returns
    -------
    solutions : dict
        Dictionary mapping Jones type → solution dict
    config : dict
        Overall configuration

    Example
    -------
    >>> solutions, config = load_multi_jones_hdf5('cal.h5')
    >>> k_jones = solutions['K']['jones']  # (n_ant, 2)
    >>> g_jones = solutions['G']['jones']  # (n_ant, 2, 2)
    """
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    logger.info(f"Loading multi-Jones solution from: {filename}")

    solutions = {}
    config = {}

    with h5py.File(filename, 'r') as f:
        # Load global config
        if 'config' in f:
            config_grp = f['config']
            config = {key: config_grp.attrs[key] for key in config_grp.attrs}

        # Determine which Jones types to load
        if jones_types is None:
            jones_types = [k for k in f.keys() if k != 'config']

        # Load each Jones type
        for jones_type in jones_types:
            if jones_type not in f:
                logger.warning(f"  Jones type '{jones_type}' not found in file")
                continue

            logger.info(f"  Loading {jones_type}...")

            jones_grp = f[jones_type]
            solution = {}

            # Load Jones matrices
            solution['jones'] = jones_grp['jones'][:]

            # Load working antennas
            if 'working_antennas' in jones_grp:
                solution['working_ants'] = jones_grp['working_antennas'][:]

            # Load convergence info
            if 'convergence' in jones_grp:
                conv_grp = jones_grp['convergence']
                solution['info'] = {key: conv_grp.attrs[key] for key in conv_grp.attrs}

            # Load metadata
            if 'metadata' in jones_grp:
                meta_grp = jones_grp['metadata']
                if 'frequencies' in meta_grp:
                    solution['freq'] = meta_grp['frequencies'][:]

            solutions[jones_type] = solution

    logger.info(f"✓ Loaded {len(solutions)} Jones types")
    return solutions, config


__all__ = ['save_multi_jones_hdf5', 'load_multi_jones_hdf5']
