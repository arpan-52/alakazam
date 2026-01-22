"""
JACKAL I/O Module.

HDF5 storage for Jones solutions.
All stored as (n_time, n_freq, n_ant, 2, 2) complex128.
Additional native params stored for delay etc.
"""

import numpy as np
import h5py
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


def save_jones(
    filename: str,
    jones_type: str,
    jones: np.ndarray,
    time: np.ndarray,
    freq: np.ndarray,
    antenna: np.ndarray = None,
    flags: np.ndarray = None,
    weights: np.ndarray = None,
    params: Dict[str, np.ndarray] = None,
    metadata: Dict[str, Any] = None,
    overwrite: bool = False,
    quality: Any = None,
):
    """
    Save Jones solution to HDF5.

    Structure:
        /{jones_type}/
            jones       (n_time, n_freq, n_ant, 2, 2) complex128
            time        (n_time,) float64
            freq        (n_freq,) float64
            antenna     (n_ant,) int32
            flags       (n_time, n_freq, n_ant) bool
            weights     (n_time, n_freq) float64  [NEW]
            /params/
                delay   (n_ant, 2) float64  [for K]
                d_xy    (n_ant,) complex128 [for D]
                ...
            /metadata/
                ref_antenna, field, mode, ...
            /quality/      [NEW]
                snr, rmse, reduced_chi2, r_squared, etc.

    weights: 1.0 = valid solution, 0.0 = failed/empty chunk (don't use)
    """
    jones = np.asarray(jones, dtype=np.complex128)
    time = np.atleast_1d(np.asarray(time, dtype=np.float64))
    freq = np.atleast_1d(np.asarray(freq, dtype=np.float64))
    
    # Expand to 5D: (n_time, n_freq, n_ant, 2, 2)
    if jones.ndim == 3:
        # (n_ant, 2, 2) -> (1, 1, n_ant, 2, 2)
        jones = jones[np.newaxis, np.newaxis, :, :, :]
    elif jones.ndim == 4:
        # (n_freq, n_ant, 2, 2) -> (1, n_freq, n_ant, 2, 2)
        jones = jones[np.newaxis, :, :, :, :]
    
    n_time, n_freq, n_ant = jones.shape[:3]

    if antenna is None:
        antenna = np.arange(n_ant, dtype=np.int32)

    if flags is None:
        flags = np.zeros((n_time, n_freq, n_ant), dtype=np.bool_)

    if weights is None:
        weights = np.ones((n_time, n_freq), dtype=np.float64)

    # Open file
    mode = 'a' if Path(filename).exists() else 'w'
    with h5py.File(filename, mode) as f:
        if jones_type in f:
            if overwrite:
                del f[jones_type]
            else:
                raise ValueError(f"{jones_type} exists. Use overwrite=True")

        grp = f.create_group(jones_type)

        # Main data
        grp.create_dataset('jones', data=jones, compression='gzip')
        grp.create_dataset('time', data=time)
        grp.create_dataset('freq', data=freq)
        grp.create_dataset('antenna', data=antenna)
        grp.create_dataset('flags', data=flags, compression='gzip')
        grp.create_dataset('weights', data=weights, compression='gzip')
        
        # Native params
        if params:
            pg = grp.create_group('params')
            for k, v in params.items():
                if v is not None:
                    pg.create_dataset(k, data=v)
        
        # Metadata
        mg = grp.create_group('metadata')
        mg.attrs['created'] = datetime.now().isoformat()
        mg.attrs['jones_type'] = jones_type
        mg.attrs['n_time'] = n_time
        mg.attrs['n_freq'] = n_freq
        mg.attrs['n_ant'] = n_ant

        if metadata:
            for k, v in metadata.items():
                if v is None:
                    continue
                if isinstance(v, (list, tuple)):
                    if all(isinstance(x, str) for x in v):
                        mg.attrs[k] = ','.join(v)
                    else:
                        mg.attrs[k] = np.array(v)
                elif isinstance(v, np.ndarray):
                    mg.create_dataset(k, data=v)
                else:
                    mg.attrs[k] = v

        # Quality metrics
        if quality is not None:
            qg = grp.create_group('quality')
            qg.attrs['snr'] = float(quality.snr)
            qg.attrs['rmse'] = float(quality.rmse)
            qg.attrs['reduced_chi2'] = float(quality.reduced_chi2)
            qg.attrs['r_squared'] = float(quality.r_squared)
            qg.attrs['cost_reduction_ratio'] = float(quality.cost_reduction_ratio)
            qg.attrs['unflagged_fraction'] = float(quality.unflagged_fraction)
            qg.attrs['n_iterations'] = int(quality.n_iterations)
            qg.attrs['n_data_points'] = int(quality.n_data_points)
            qg.attrs['n_parameters'] = int(quality.n_parameters)
            qg.attrs['convergence_success'] = bool(quality.convergence_success)


def load_jones(filename: str, jones_type: str) -> Dict[str, Any]:
    """
    Load Jones solution from HDF5.

    Returns dict with:
        jones, time, freq, antenna, flags, weights, params, metadata
    """
    with h5py.File(filename, 'r') as f:
        if jones_type not in f:
            raise KeyError(f"{jones_type} not in {filename}")

        grp = f[jones_type]

        data = {
            'jones': grp['jones'][:],
            'time': grp['time'][:],
            'freq': grp['freq'][:],
            'antenna': grp['antenna'][:],
            'flags': grp['flags'][:],
        }

        # Weights (backward compatible - default to all 1s if not present)
        if 'weights' in grp:
            data['weights'] = grp['weights'][:]
        else:
            n_time, n_freq = grp['jones'].shape[:2]
            data['weights'] = np.ones((n_time, n_freq), dtype=np.float64)

        # Params
        params = {}
        if 'params' in grp:
            for k in grp['params']:
                params[k] = grp['params'][k][:]
        data['params'] = params

        # Metadata
        metadata = {}
        if 'metadata' in grp:
            for k in grp['metadata'].attrs:
                metadata[k] = grp['metadata'].attrs[k]
        data['metadata'] = metadata

    return data


def list_jones(filename: str) -> List[str]:
    """List Jones types in file."""
    with h5py.File(filename, 'r') as f:
        return [k for k in f.keys() if isinstance(f[k], h5py.Group)]


def print_summary(filename: str):
    """Print summary of Jones file."""
    print(f"\nJones File: {filename}")
    print("-" * 50)
    
    with h5py.File(filename, 'r') as f:
        for jt in f.keys():
            if not isinstance(f[jt], h5py.Group):
                continue
            
            grp = f[jt]
            shape = grp['jones'].shape
            
            meta = {}
            if 'metadata' in grp:
                for k in grp['metadata'].attrs:
                    meta[k] = grp['metadata'].attrs[k]
            
            print(f"\n{jt}:")
            print(f"  Shape: {shape}")
            print(f"  n_time={shape[0]}, n_freq={shape[1]}, n_ant={shape[2]}")
            
            if 'ref_antenna' in meta:
                print(f"  ref_antenna: {meta['ref_antenna']}")
            if 'field' in meta:
                print(f"  field: {meta['field']}")
            if 'cost_init' in meta and 'cost_final' in meta:
                print(f"  cost: {meta['cost_init']:.2e} -> {meta['cost_final']:.2e}")
            
            # Show params
            if 'params' in grp:
                for pk in grp['params']:
                    pshape = grp['params'][pk].shape
                    print(f"  params/{pk}: {pshape}")


# Import enhanced HDF5 I/O
from .hdf5_io import write_solution_hdf5, read_solution_hdf5


__all__ = ['save_jones', 'load_jones', 'list_jones', 'print_summary', 'write_solution_hdf5', 'read_solution_hdf5']
