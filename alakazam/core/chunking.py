"""
Chunked Data Loading Module.

Loads MS data in solint-sized chunks for memory-efficient processing.
Each chunk corresponds to one solution interval in time and frequency.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterator
import logging
from .metadata import MSMetadata

logger = logging.getLogger('jackal')


@dataclass
class DataChunk:
    """
    Data for a single solution interval.

    This represents the data for one (time_idx, freq_idx) solution cell.
    """
    # Solution indices
    time_idx: int  # Index in solution array time dimension
    freq_idx: int  # Index in solution array frequency dimension

    # Time/freq ranges for this chunk
    time_range: Tuple[float, float]  # (t_start, t_end) in MJD seconds
    freq_range: Tuple[float, float]  # (f_start, f_end) in Hz
    freq_indices: np.ndarray  # Channel indices in this chunk

    # Data arrays
    vis_obs: np.ndarray  # (n_row, n_chan_chunk, 2, 2) or (n_row, 2, 2) if freq-averaged
    vis_model: np.ndarray  # Same shape as vis_obs
    antenna1: np.ndarray  # (n_row,) int32
    antenna2: np.ndarray  # (n_row,) int32
    flags: np.ndarray  # Same shape as vis_obs
    time: np.ndarray  # (n_row,) float64 - time stamps for each row
    freq: np.ndarray  # (n_chan_chunk,) float64 - frequencies for this chunk

    # Statistics
    n_rows: int
    n_baselines: int
    flag_fraction: float  # Fraction of flagged data in this chunk


def compute_chunk_ranges(
    metadata: MSMetadata
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute time and frequency ranges for each solution interval.

    Parameters
    ----------
    metadata : MSMetadata
        MS metadata including solution intervals

    Returns
    -------
    time_edges : ndarray
        Time edges (n_sol_time + 1,) defining solution intervals
    freq_edges : ndarray
        Frequency edges (n_sol_freq + 1,) defining solution intervals
    """
    t_min, t_max = metadata.time_range
    f_min, f_max = metadata.freq_range

    # Time edges
    if metadata.n_sol_time == 1:
        time_edges = np.array([t_min, t_max])
    else:
        time_edges = np.linspace(t_min, t_max, metadata.n_sol_time + 1)

    # Frequency edges
    if metadata.n_sol_freq == 1:
        freq_edges = np.array([f_min, f_max])
    else:
        # Use actual frequency channel boundaries
        n_chan_per_chunk = metadata.n_freq // metadata.n_sol_freq
        freq_indices = np.arange(0, metadata.n_freq + 1, n_chan_per_chunk)
        if freq_indices[-1] < metadata.n_freq:
            freq_indices = np.append(freq_indices, metadata.n_freq)

        freq_edges = np.zeros(len(freq_indices))
        for i, idx in enumerate(freq_indices):
            if idx < metadata.n_freq:
                freq_edges[i] = metadata.freq[idx]
            else:
                freq_edges[i] = metadata.freq[-1] + (metadata.freq[-1] - metadata.freq[-2])

    return time_edges, freq_edges


def load_chunk(
    ms_path: str,
    time_range: Tuple[float, float],
    freq_indices: np.ndarray,
    field: Optional[str] = None,
    spw: Optional[str] = None,
    scans: Optional[str] = None,
    data_col: str = "DATA",
    model_col: str = "MODEL_DATA",
    working_ants: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data for a single solution interval chunk.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set
    time_range : tuple
        (t_start, t_end) in MJD seconds
    freq_indices : ndarray
        Channel indices to load
    field : str, optional
        Field selection
    spw : str, optional
        SPW selection
    scans : str, optional
        Scan selection
    data_col : str
        Data column
    model_col : str
        Model column
    working_ants : ndarray, optional
        Working antenna indices (for filtering)

    Returns
    -------
    vis_obs : ndarray (n_row, n_chan, 2, 2)
    vis_model : ndarray (n_row, n_chan, 2, 2)
    antenna1 : ndarray (n_row,)
    antenna2 : ndarray (n_row,)
    flags : ndarray (n_row, n_chan, 2, 2)
    time : ndarray (n_row,)
    freq : ndarray (n_chan,)
    """
    from casacore.tables import table, taql

    t_start, t_end = time_range

    # Open MS
    ms = table(ms_path, readonly=True, ack=False)

    # Build selection
    conditions = []

    if field is not None:
        field_tab = table(f"{ms_path}::FIELD", readonly=True, ack=False)
        field_names = list(field_tab.getcol("NAME"))
        field_tab.close()
        if field in field_names:
            field_id = field_names.index(field)
            conditions.append(f"FIELD_ID=={field_id}")

    if spw is not None:
        if "~" in spw:
            start, end = spw.split("~")
            spw_ids = list(range(int(start), int(end) + 1))
        elif "," in spw:
            spw_ids = [int(s) for s in spw.split(",")]
        else:
            spw_ids = [int(spw)]
        conditions.append(f"DATA_DESC_ID IN [{','.join(map(str, spw_ids))}]")

    if scans is not None:
        if "~" in scans:
            start, end = scans.split("~")
            scan_ids = list(range(int(start), int(end) + 1))
        elif "," in scans:
            scan_ids = [int(s) for s in scans.split(",")]
        else:
            scan_ids = [int(scans)]
        conditions.append(f"SCAN_NUMBER IN [{','.join(map(str, scan_ids))}]")

    # Add time condition
    conditions.append(f"TIME >= {t_start} AND TIME < {t_end}")

    # Select data
    if conditions:
        query = f"SELECT * FROM $ms WHERE {' AND '.join(conditions)}"
        sel = taql(query)
    else:
        sel = taql(f"SELECT * FROM $ms WHERE TIME >= {t_start} AND TIME < {t_end}")

    if sel.nrows() == 0:
        # Empty chunk
        ms.close()
        sel.close()
        return None, None, None, None, None, None, None

    # Read antenna indices and time
    antenna1 = sel.getcol("ANTENNA1").astype(np.int32)
    antenna2 = sel.getcol("ANTENNA2").astype(np.int32)
    time_arr = sel.getcol("TIME")

    # Filter to working antennas if specified
    if working_ants is not None:
        working_set = set(working_ants)
        valid_rows = np.array([a1 in working_set and a2 in working_set for a1, a2 in zip(antenna1, antenna2)])

        if not np.any(valid_rows):
            # No valid rows
            ms.close()
            sel.close()
            return None, None, None, None, None, None, None

        # Filter
        antenna1 = antenna1[valid_rows]
        antenna2 = antenna2[valid_rows]
        time_arr = time_arr[valid_rows]

    # Read data and flags
    if data_col in sel.colnames():
        data = sel.getcol(data_col)
        if working_ants is not None:
            data = data[valid_rows]
    else:
        raise ValueError(f"Column {data_col} not found")

    if model_col in sel.colnames():
        model = sel.getcol(model_col)
        if working_ants is not None:
            model = model[valid_rows]
    else:
        # Use unity model
        model = np.ones_like(data)

    flags = sel.getcol("FLAG")
    if working_ants is not None:
        flags = flags[valid_rows]

    sel.close()

    # Get frequencies for this chunk
    spw_tab = table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True, ack=False)
    all_freq = spw_tab.getcol("CHAN_FREQ")[0]
    spw_tab.close()

    ms.close()

    freq = all_freq[freq_indices]

    # Extract channels for this chunk
    data = data[:, freq_indices, :]
    model = model[:, freq_indices, :]
    flags = flags[:, freq_indices, :]

    # Reshape to (n_row, n_chan, 2, 2)
    n_row, n_chan, n_corr = data.shape

    if n_corr == 4:
        vis_obs = data.reshape(n_row, n_chan, 2, 2)
        vis_model = model.reshape(n_row, n_chan, 2, 2)
        flags_out = flags.reshape(n_row, n_chan, 2, 2)
    elif n_corr == 2:
        vis_obs = np.zeros((n_row, n_chan, 2, 2), dtype=np.complex128)
        vis_model = np.zeros((n_row, n_chan, 2, 2), dtype=np.complex128)
        flags_out = np.ones((n_row, n_chan, 2, 2), dtype=bool)

        vis_obs[:, :, 0, 0] = data[:, :, 0]
        vis_obs[:, :, 1, 1] = data[:, :, 1]
        vis_model[:, :, 0, 0] = model[:, :, 0]
        vis_model[:, :, 1, 1] = model[:, :, 1]
        flags_out[:, :, 0, 0] = flags[:, :, 0]
        flags_out[:, :, 1, 1] = flags[:, :, 1]
    else:
        raise ValueError(f"Unexpected number of correlations: {n_corr}")

    return (
        np.ascontiguousarray(vis_obs, dtype=np.complex128),
        np.ascontiguousarray(vis_model, dtype=np.complex128),
        np.ascontiguousarray(antenna1, dtype=np.int32),
        np.ascontiguousarray(antenna2, dtype=np.int32),
        np.ascontiguousarray(flags_out, dtype=bool),
        time_arr,
        np.ascontiguousarray(freq, dtype=np.float64)
    )


def iterate_chunks(
    metadata: MSMetadata,
    data_col: str = "DATA",
    model_col: str = "MODEL_DATA"
) -> Iterator[DataChunk]:
    """
    Iterator over solution interval chunks.

    Yields DataChunk objects for each (time_idx, freq_idx) solution cell.

    Parameters
    ----------
    metadata : MSMetadata
        MS metadata
    data_col : str
        Data column name
    model_col : str
        Model column name

    Yields
    ------
    chunk : DataChunk
        Data for one solution interval
    """
    # Compute chunk ranges
    time_edges, freq_edges = compute_chunk_ranges(metadata)

    # Total chunks
    total_chunks = metadata.n_sol_time * metadata.n_sol_freq

    logger.info(f"Loading data in {total_chunks} chunks ({metadata.n_sol_time} time × {metadata.n_sol_freq} freq)")

    chunk_num = 0

    # Iterate over solution intervals
    for t_idx in range(metadata.n_sol_time):
        time_range = (time_edges[t_idx], time_edges[t_idx + 1])

        for f_idx in range(metadata.n_sol_freq):
            chunk_num += 1

            # Determine frequency channels for this chunk
            if metadata.n_sol_freq == 1:
                freq_indices = np.arange(metadata.n_freq)
            else:
                n_chan_per_chunk = metadata.n_freq // metadata.n_sol_freq
                f_start = f_idx * n_chan_per_chunk
                f_end = f_start + n_chan_per_chunk if f_idx < metadata.n_sol_freq - 1 else metadata.n_freq
                freq_indices = np.arange(f_start, f_end)

            freq_range = (metadata.freq[freq_indices[0]], metadata.freq[freq_indices[-1]])

            logger.info(f"  Loading chunk {chunk_num}/{total_chunks} "
                       f"(t_idx={t_idx}, f_idx={f_idx}): "
                       f"time [{time_range[0]:.1f}, {time_range[1]:.1f}]s, "
                       f"freq [{freq_range[0]/1e9:.3f}, {freq_range[1]/1e9:.3f}] GHz")

            # Load data
            result = load_chunk(
                metadata.ms_path,
                time_range,
                freq_indices,
                field=metadata.field,
                spw=metadata.spw,
                scans=None,
                data_col=data_col,
                model_col=model_col,
                working_ants=metadata.working_antennas
            )

            if result[0] is None:
                # Empty chunk, skip
                logger.warning(f"    Chunk {chunk_num} is empty, skipping")
                continue

            vis_obs, vis_model, antenna1, antenna2, flags, time, freq = result

            # Compute statistics
            n_rows = len(antenna1)
            n_baselines = len(np.unique(list(zip(antenna1, antenna2))))
            flag_fraction = np.sum(flags) / flags.size

            logger.info(f"    Loaded: {n_rows} rows, {n_baselines} baselines, "
                       f"{flag_fraction * 100:.1f}% flagged")

            chunk = DataChunk(
                time_idx=t_idx,
                freq_idx=f_idx,
                time_range=time_range,
                freq_range=freq_range,
                freq_indices=freq_indices,
                vis_obs=vis_obs,
                vis_model=vis_model,
                antenna1=antenna1,
                antenna2=antenna2,
                flags=flags,
                time=time,
                freq=freq,
                n_rows=n_rows,
                n_baselines=n_baselines,
                flag_fraction=flag_fraction
            )

            yield chunk


__all__ = [
    'DataChunk',
    'iterate_chunks',
    'load_chunk',
    'compute_chunk_ranges',
]
