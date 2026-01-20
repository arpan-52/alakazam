"""
Metadata Detection Module.

Detects MS metadata and non-working antennas using chunked loading.
Never loads entire MS into memory - uses 5-minute chunks.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger('jackal')


@dataclass
class MSMetadata:
    """Metadata extracted from Measurement Set."""
    # Basic structure
    n_ant: int
    n_freq: int
    n_time: int
    n_baselines: int
    n_rows: int

    # Ranges
    time_range: Tuple[float, float]  # (t_min, t_max) in MJD seconds
    freq_range: Tuple[float, float]  # (f_min, f_max) in Hz

    # Arrays
    freq: np.ndarray  # (n_freq,) in Hz
    unique_times: np.ndarray  # (n_time,) in MJD seconds
    antenna_names: List[str]  # Length n_ant

    # Working antennas
    working_antennas: np.ndarray  # Indices of working antennas
    non_working_antennas: np.ndarray  # Indices of non-working antennas

    # Baseline info
    all_baselines: np.ndarray  # (n_unique_bl, 2) - unique (ant1, ant2) pairs
    valid_baselines: np.ndarray  # (n_valid_bl, 2) - baselines with working antennas only

    # Solution intervals
    n_sol_time: int  # Number of solution intervals in time
    n_sol_freq: int  # Number of solution intervals in frequency
    solint_time: float  # Time interval in seconds
    solint_freq: float  # Frequency interval in Hz

    # Field info
    feed_basis: str  # 'LINEAR' or 'CIRCULAR'
    ms_path: str
    field: Optional[str]
    spw: Optional[str]

    # Statistics
    total_valid_data_fraction: float  # Fraction of unflagged data


def detect_non_working_antennas_chunked(
    ms_path: str,
    field: Optional[str] = None,
    spw: Optional[str] = None,
    scans: Optional[str] = None,
    chunk_time_seconds: float = 300.0,  # 5 minutes
    data_col: str = "DATA",
    model_col: str = "MODEL_DATA"
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Detect non-working antennas by loading MS in time chunks.

    An antenna is considered "working" if it appears in at least one
    baseline with unflagged data across all chunks.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set
    field : str, optional
        Field name to select
    spw : str, optional
        Spectral window selection
    scans : str, optional
        Scan selection
    chunk_time_seconds : float
        Size of time chunks to load (default 5 minutes)
    data_col : str
        Data column name
    model_col : str
        Model column name

    Returns
    -------
    working_ants : ndarray
        Array of working antenna indices
    non_working_ants : ndarray
        Array of non-working antenna indices
    n_ant : int
        Total number of antennas
    """
    from casacore.tables import table, taql

    logger.info("Detecting non-working antennas using chunked loading (5-min chunks)...")

    # Open MS
    ms = table(ms_path, readonly=True, ack=False)

    # Get total number of antennas
    ant_tab = table(f"{ms_path}::ANTENNA", readonly=True, ack=False)
    n_ant = ant_tab.nrows()
    ant_tab.close()

    # Build TaQL selection
    conditions = []

    if field is not None:
        field_tab = table(f"{ms_path}::FIELD", readonly=True, ack=False)
        field_names = list(field_tab.getcol("NAME"))
        field_tab.close()

        if field in field_names:
            field_id = field_names.index(field)
            conditions.append(f"FIELD_ID=={field_id}")
        else:
            logger.warning(f"Field '{field}' not found, using all fields")

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

    # Select data
    if conditions:
        query = f"SELECT * FROM $ms WHERE {' AND '.join(conditions)}"
        sel = taql(query)
    else:
        sel = ms

    # Get time range
    all_times = sel.getcol("TIME")
    time_min, time_max = np.min(all_times), np.max(all_times)

    # Track which antennas have valid data
    antenna_has_data = np.zeros(n_ant, dtype=bool)

    # Process in chunks
    n_chunks = int(np.ceil((time_max - time_min) / chunk_time_seconds))
    logger.info(f"  Total time range: {(time_max - time_min):.1f}s, processing in {n_chunks} chunks")

    for chunk_idx in range(n_chunks):
        t_start = time_min + chunk_idx * chunk_time_seconds
        t_end = min(t_start + chunk_time_seconds, time_max)

        # Select this time chunk
        if conditions:
            time_cond = f"TIME >= {t_start} AND TIME < {t_end}"
            chunk_query = f"SELECT * FROM $sel WHERE {time_cond}"
        else:
            chunk_query = f"SELECT * FROM $sel WHERE TIME >= {t_start} AND TIME < {t_end}"

        chunk = taql(chunk_query)

        if chunk.nrows() == 0:
            chunk.close()
            continue

        # Read antenna indices and flags
        ant1 = chunk.getcol("ANTENNA1").astype(np.int32)
        ant2 = chunk.getcol("ANTENNA2").astype(np.int32)
        flags = chunk.getcol("FLAG")

        chunk.close()

        # Check which antennas have unflagged data in this chunk
        for i in range(len(ant1)):
            # Check if this baseline has any unflagged data
            has_unflagged = not np.all(flags[i])

            if has_unflagged:
                antenna_has_data[ant1[i]] = True
                antenna_has_data[ant2[i]] = True

        logger.info(f"  Chunk {chunk_idx + 1}/{n_chunks}: {len(ant1)} rows, "
                   f"{np.sum(antenna_has_data)}/{n_ant} antennas with data so far")

    sel.close()
    ms.close()

    # Determine working and non-working antennas
    working_ants = np.where(antenna_has_data)[0]
    non_working_ants = np.where(~antenna_has_data)[0]

    logger.info(f"  Working antennas: {len(working_ants)}/{n_ant}")
    if len(non_working_ants) > 0:
        logger.warning(f"  Non-working antennas: {list(non_working_ants)}")

    return working_ants, non_working_ants, n_ant


def compute_solution_intervals(
    time_range: Tuple[float, float],
    freq: np.ndarray,
    solint_time: str,
    solint_freq: str
) -> Tuple[int, int, float, float]:
    """
    Compute number of solution intervals based on solint specifications.

    Parameters
    ----------
    time_range : tuple
        (t_min, t_max) in seconds
    freq : ndarray
        Frequency array in Hz
    solint_time : str
        Time interval: 'inf', 'Ns', 'Nmin', 'Nh'
    solint_freq : str
        Frequency interval: 'full', 'NMHz', 'Nchan', 'inf'

    Returns
    -------
    n_sol_time : int
        Number of solution intervals in time
    n_sol_freq : int
        Number of solution intervals in frequency
    solint_time_sec : float
        Time interval in seconds
    solint_freq_hz : float
        Frequency interval in Hz
    """
    t_min, t_max = time_range
    total_time = t_max - t_min

    # Parse time interval
    if solint_time == 'inf':
        n_sol_time = 1
        solint_time_sec = np.inf
    elif solint_time.endswith('s'):
        solint_time_sec = float(solint_time[:-1])
        n_sol_time = max(1, int(np.ceil(total_time / solint_time_sec)))
    elif solint_time.endswith('min'):
        solint_time_sec = float(solint_time[:-3]) * 60.0
        n_sol_time = max(1, int(np.ceil(total_time / solint_time_sec)))
    elif solint_time.endswith('h'):
        solint_time_sec = float(solint_time[:-1]) * 3600.0
        n_sol_time = max(1, int(np.ceil(total_time / solint_time_sec)))
    else:
        # Try to parse as float (seconds)
        try:
            solint_time_sec = float(solint_time)
            n_sol_time = max(1, int(np.ceil(total_time / solint_time_sec)))
        except ValueError:
            logger.warning(f"Cannot parse solint_time '{solint_time}', using 'inf'")
            n_sol_time = 1
            solint_time_sec = np.inf

    # Parse frequency interval
    n_freq = len(freq)
    freq_min, freq_max = freq[0], freq[-1]
    total_bw = freq_max - freq_min
    chan_width = (freq[1] - freq[0]) if n_freq > 1 else 1.0

    if solint_freq in ('full', 'inf'):
        n_sol_freq = 1
        solint_freq_hz = total_bw
    elif solint_freq.endswith('MHz'):
        solint_freq_hz = float(solint_freq[:-3]) * 1e6
        n_sol_freq = max(1, int(np.ceil(total_bw / solint_freq_hz)))
    elif solint_freq.endswith('chan'):
        n_chan = int(solint_freq[:-4])
        solint_freq_hz = n_chan * chan_width
        n_sol_freq = max(1, int(np.ceil(n_freq / n_chan)))
    else:
        # Try to parse as int (number of channels)
        try:
            n_chan = int(solint_freq)
            solint_freq_hz = n_chan * chan_width
            n_sol_freq = max(1, int(np.ceil(n_freq / n_chan)))
        except ValueError:
            logger.warning(f"Cannot parse solint_freq '{solint_freq}', using 'full'")
            n_sol_freq = 1
            solint_freq_hz = total_bw

    return n_sol_time, n_sol_freq, solint_time_sec, solint_freq_hz


def detect_ms_metadata(
    ms_path: str,
    field: Optional[str] = None,
    spw: Optional[str] = None,
    scans: Optional[str] = None,
    solint_time: str = 'inf',
    solint_freq: str = 'full',
    data_col: str = "DATA",
    model_col: str = "MODEL_DATA"
) -> MSMetadata:
    """
    Detect all MS metadata including structure, working antennas, and solution intervals.

    Uses chunked loading for non-working antenna detection to handle large MS files.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set
    field : str, optional
        Field name
    spw : str, optional
        Spectral window selection
    scans : str, optional
        Scan selection
    solint_time : str
        Time solution interval
    solint_freq : str
        Frequency solution interval
    data_col : str
        Data column name
    model_col : str
        Model column name

    Returns
    -------
    metadata : MSMetadata
        Complete metadata
    """
    from casacore.tables import table, taql

    logger.info("=" * 70)
    logger.info("Detecting MS Metadata")
    logger.info("=" * 70)
    logger.info(f"MS: {ms_path}")
    if field:
        logger.info(f"Field: {field}")
    if spw:
        logger.info(f"SPW: {spw}")
    if scans:
        logger.info(f"Scans: {scans}")

    # Detect non-working antennas first (chunked)
    working_ants, non_working_ants, n_ant = detect_non_working_antennas_chunked(
        ms_path, field, spw, scans, data_col=data_col, model_col=model_col
    )

    # Now load metadata (lightweight)
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

    if conditions:
        query = f"SELECT * FROM $ms WHERE {' AND '.join(conditions)}"
        sel = taql(query)
    else:
        sel = ms

    n_rows = sel.nrows()

    # Read basic info (just indices, not data)
    antenna1 = sel.getcol("ANTENNA1").astype(np.int32)
    antenna2 = sel.getcol("ANTENNA2").astype(np.int32)
    time_arr = sel.getcol("TIME")
    flags = sel.getcol("FLAG")

    sel.close()
    ms.close()

    # Get frequency info
    spw_tab = table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True, ack=False)
    freq = spw_tab.getcol("CHAN_FREQ")[0]
    spw_tab.close()

    n_freq = len(freq)
    freq_range = (freq[0], freq[-1])

    # Get time info
    unique_times = np.unique(time_arr)
    n_time = len(unique_times)
    time_range = (unique_times[0], unique_times[-1])

    # Get feed basis
    pol_tab = table(f"{ms_path}::POLARIZATION", readonly=True, ack=False)
    corr_type = pol_tab.getcol("CORR_TYPE")[0]
    pol_tab.close()

    if corr_type[0] in [9, 10, 11, 12]:
        feed_basis = "LINEAR"
    elif corr_type[0] in [5, 6, 7, 8]:
        feed_basis = "CIRCULAR"
    else:
        feed_basis = "LINEAR"

    # Get antenna names
    ant_tab = table(f"{ms_path}::ANTENNA", readonly=True, ack=False)
    antenna_names = list(ant_tab.getcol("NAME"))
    ant_tab.close()

    # Get unique baselines
    baseline_set = set()
    for a1, a2 in zip(antenna1, antenna2):
        if a1 < a2:
            baseline_set.add((a1, a2))
        else:
            baseline_set.add((a2, a1))

    all_baselines = np.array(sorted(list(baseline_set)), dtype=np.int32)
    n_baselines = len(all_baselines)

    # Filter baselines to only working antennas
    valid_bl_list = []
    for a1, a2 in all_baselines:
        if a1 in working_ants and a2 in working_ants:
            valid_bl_list.append((a1, a2))

    valid_baselines = np.array(valid_bl_list, dtype=np.int32) if valid_bl_list else np.array([], dtype=np.int32).reshape(0, 2)

    # Compute valid data fraction
    total_valid_data_fraction = 1.0 - (np.sum(flags) / flags.size)

    # Compute solution intervals
    n_sol_time, n_sol_freq, solint_time_sec, solint_freq_hz = compute_solution_intervals(
        time_range, freq, solint_time, solint_freq
    )

    metadata = MSMetadata(
        n_ant=n_ant,
        n_freq=n_freq,
        n_time=n_time,
        n_baselines=n_baselines,
        n_rows=n_rows,
        time_range=time_range,
        freq_range=freq_range,
        freq=freq,
        unique_times=unique_times,
        antenna_names=antenna_names,
        working_antennas=working_ants,
        non_working_antennas=non_working_ants,
        all_baselines=all_baselines,
        valid_baselines=valid_baselines,
        n_sol_time=n_sol_time,
        n_sol_freq=n_sol_freq,
        solint_time=solint_time_sec,
        solint_freq=solint_freq_hz,
        feed_basis=feed_basis,
        ms_path=ms_path,
        field=field,
        spw=spw,
        total_valid_data_fraction=total_valid_data_fraction
    )

    # Log summary
    logger.info("")
    logger.info("MS Structure:")
    logger.info(f"  Antennas: {n_ant} total, {len(working_ants)} working, {len(non_working_ants)} non-working")
    logger.info(f"  Frequency: {n_freq} channels, {freq_range[0]/1e9:.3f}-{freq_range[1]/1e9:.3f} GHz")
    logger.info(f"  Time: {n_time} timestamps, {(time_range[1] - time_range[0]):.1f}s span")
    logger.info(f"  Baselines: {n_baselines} total, {len(valid_baselines)} valid (working ants only)")
    logger.info(f"  Rows: {n_rows}")
    logger.info(f"  Feed basis: {feed_basis}")
    logger.info(f"  Valid data: {total_valid_data_fraction * 100:.1f}%")
    logger.info("")
    logger.info("Solution Intervals:")
    logger.info(f"  Time: {n_sol_time} intervals of {solint_time_sec}s" if solint_time_sec != np.inf else f"  Time: {n_sol_time} interval (full)")
    logger.info(f"  Frequency: {n_sol_freq} intervals of {solint_freq_hz/1e6:.2f} MHz" if n_sol_freq > 1 else f"  Frequency: {n_sol_freq} interval (full)")
    logger.info(f"  Solution array shape: ({n_sol_time}, {n_sol_freq}, {n_ant}, 2, 2)")
    logger.info("=" * 70)
    logger.info("")

    return metadata


__all__ = [
    'MSMetadata',
    'detect_ms_metadata',
    'detect_non_working_antennas_chunked',
    'compute_solution_intervals',
]
