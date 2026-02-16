"""
ALAKAZAM MS I/O.

Memory-safe Measurement Set reading. Loads data in solint-sized chunks.
Multi-SPW support. Detects metadata, working antennas, feed basis.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
from dataclasses import dataclass, field as dc_field
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger("alakazam")


@dataclass
class MSMetadata:
    """Metadata extracted from a Measurement Set."""
    ms_path: str
    n_ant: int
    n_freq: int
    n_time: int
    n_rows: int
    freq: np.ndarray
    unique_times: np.ndarray
    antenna_names: List[str]
    working_antennas: np.ndarray
    non_working_antennas: np.ndarray
    feed_basis: str
    field: Optional[str]
    spw_id: int
    all_spw_ids: List[int]
    all_spw_freqs: Dict[int, np.ndarray] = dc_field(default_factory=dict)


def parse_spw_selection(spw_str: str, n_spw: int) -> List[int]:
    """Parse SPW selection string: '*', '0', '0~3', '0,2,4'."""
    spw_str = str(spw_str).strip()
    if spw_str in ("*", "all", ""):
        return list(range(n_spw))
    if "~" in spw_str:
        parts = spw_str.split("~")
        return list(range(int(parts[0]), int(parts[1]) + 1))
    if "," in spw_str:
        return [int(s.strip()) for s in spw_str.split(",")]
    return [int(spw_str)]


def detect_metadata(
    ms_path: str,
    field: Optional[str] = None,
    spw_id: int = 0,
    scans: Optional[str] = None,
    chunk_time_sec: float = 300.0,
) -> MSMetadata:
    """Detect MS metadata using chunked loading for antenna detection.

    Never loads entire MS into memory.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set.
    field : str, optional
        Field name to select.
    spw_id : int
        Spectral window ID for this metadata.
    scans : str, optional
        Scan selection string.
    chunk_time_sec : float
        Time chunk size for antenna detection (seconds).

    Returns
    -------
    meta : MSMetadata
    """
    from casacore.tables import table, taql

    ms = table(ms_path, readonly=True, ack=False)

    # Antenna info
    ant_tab = table(f"{ms_path}::ANTENNA", readonly=True, ack=False)
    n_ant = ant_tab.nrows()
    antenna_names = list(ant_tab.getcol("NAME"))
    ant_tab.close()

    # All SPW freqs
    spw_tab = table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True, ack=False)
    n_spw = spw_tab.nrows()
    all_spw_ids = list(range(n_spw))
    all_spw_freqs = {}
    for sid in range(n_spw):
        all_spw_freqs[sid] = spw_tab.getcol("CHAN_FREQ", startrow=sid, nrow=1)[0]
    spw_tab.close()

    freq = all_spw_freqs[spw_id]
    n_freq = len(freq)

    # Feed basis
    pol_tab = table(f"{ms_path}::POLARIZATION", readonly=True, ack=False)
    corr_type = pol_tab.getcol("CORR_TYPE")[0]
    pol_tab.close()
    feed_basis = "CIRCULAR" if corr_type[0] in (5, 6, 7, 8) else "LINEAR"

    # Field ID
    field_id = None
    if field is not None:
        field_tab = table(f"{ms_path}::FIELD", readonly=True, ack=False)
        field_names = list(field_tab.getcol("NAME"))
        field_tab.close()
        if field in field_names:
            field_id = field_names.index(field)
        else:
            logger.warning(f"Field '{field}' not found. Using all fields.")

    # Build selection query
    conditions = [f"DATA_DESC_ID=={spw_id}"]
    if field_id is not None:
        conditions.append(f"FIELD_ID=={field_id}")
    if scans is not None:
        scan_ids = _parse_scan_selection(scans)
        conditions.append(f"SCAN_NUMBER IN [{','.join(map(str, scan_ids))}]")

    where = " AND ".join(conditions)
    sel = taql(f"SELECT * FROM $ms WHERE {where}")
    n_rows = sel.nrows()

    if n_rows == 0:
        ms.close()
        sel.close()
        raise ValueError(
            f"No data found for SPW={spw_id}, field={field}, scans={scans}"
        )

    time_arr = sel.getcol("TIME")
    unique_times = np.unique(time_arr)
    n_time = len(unique_times)

    # Detect working antennas in chunks
    antenna_has_data = np.zeros(n_ant, dtype=bool)
    t_min, t_max = np.min(time_arr), np.max(time_arr)
    n_chunks = max(1, int(np.ceil((t_max - t_min) / chunk_time_sec)))

    for ci in range(n_chunks):
        t_start = t_min + ci * chunk_time_sec
        t_end = min(t_start + chunk_time_sec, t_max + 1.0)
        chunk_q = taql(
            f"SELECT ANTENNA1, ANTENNA2, FLAG FROM $sel "
            f"WHERE TIME >= {t_start} AND TIME < {t_end}"
        )
        if chunk_q.nrows() == 0:
            chunk_q.close()
            continue
        a1 = chunk_q.getcol("ANTENNA1")
        a2 = chunk_q.getcol("ANTENNA2")
        flags = chunk_q.getcol("FLAG")
        chunk_q.close()

        for i in range(len(a1)):
            if not np.all(flags[i]):
                antenna_has_data[a1[i]] = True
                antenna_has_data[a2[i]] = True

    sel.close()
    ms.close()

    working = np.where(antenna_has_data)[0]
    non_working = np.where(~antenna_has_data)[0]

    logger.info(
        f"SPW {spw_id}: {n_ant} antennas ({len(working)} working), "
        f"{n_freq} channels, {n_time} times, {n_rows} rows, {feed_basis}"
    )

    return MSMetadata(
        ms_path=ms_path,
        n_ant=n_ant,
        n_freq=n_freq,
        n_time=n_time,
        n_rows=n_rows,
        freq=freq,
        unique_times=unique_times,
        antenna_names=antenna_names,
        working_antennas=working,
        non_working_antennas=non_working,
        feed_basis=feed_basis,
        field=field,
        spw_id=spw_id,
        all_spw_ids=all_spw_ids,
        all_spw_freqs=all_spw_freqs,
    )


def load_chunk(
    ms_path: str,
    spw_id: int,
    time_range: Tuple[float, float],
    freq_indices: np.ndarray,
    field: Optional[str] = None,
    scans: Optional[str] = None,
    data_col: str = "DATA",
    model_col: str = "MODEL_DATA",
    working_ants: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    """Load one solint chunk from the MS.

    Returns None if chunk is empty.
    Returns dict with keys: vis_obs, vis_model, ant1, ant2, flags, time, freq
    """
    from casacore.tables import table, taql

    t_start, t_end = time_range
    ms = table(ms_path, readonly=True, ack=False)

    conditions = [
        f"DATA_DESC_ID=={spw_id}",
        f"TIME >= {t_start}",
        f"TIME < {t_end}",
    ]
    if field is not None:
        field_tab = table(f"{ms_path}::FIELD", readonly=True, ack=False)
        field_names = list(field_tab.getcol("NAME"))
        field_tab.close()
        if field in field_names:
            conditions.append(f"FIELD_ID=={field_names.index(field)}")

    if scans is not None:
        scan_ids = _parse_scan_selection(scans)
        conditions.append(f"SCAN_NUMBER IN [{','.join(map(str, scan_ids))}]")

    where = " AND ".join(conditions)
    sel = taql(f"SELECT * FROM $ms WHERE {where}")

    if sel.nrows() == 0:
        sel.close()
        ms.close()
        return None

    ant1 = sel.getcol("ANTENNA1").astype(np.int32)
    ant2 = sel.getcol("ANTENNA2").astype(np.int32)
    time_arr = sel.getcol("TIME")

    # Filter working antennas
    if working_ants is not None:
        ws = set(int(a) for a in working_ants)
        valid = np.array([int(a1) in ws and int(a2) in ws for a1, a2 in zip(ant1, ant2)])
        if not np.any(valid):
            sel.close()
            ms.close()
            return None
        ant1 = ant1[valid]
        ant2 = ant2[valid]
        time_arr = time_arr[valid]
    else:
        valid = np.ones(len(ant1), dtype=bool)

    # Read data
    if data_col in sel.colnames():
        data = sel.getcol(data_col)
        if working_ants is not None:
            data = data[valid]
    else:
        sel.close()
        ms.close()
        raise ValueError(f"Column '{data_col}' not found in MS")

    if model_col in sel.colnames():
        model = sel.getcol(model_col)
        if working_ants is not None:
            model = model[valid]
    else:
        model = np.ones_like(data)
        logger.warning(f"Column '{model_col}' not found. Using unity model.")

    flags = sel.getcol("FLAG")
    if working_ants is not None:
        flags = flags[valid]

    sel.close()

    # Get freqs for this SPW
    spw_tab = table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True, ack=False)
    all_freq = spw_tab.getcol("CHAN_FREQ", startrow=spw_id, nrow=1)[0]
    spw_tab.close()
    ms.close()

    freq = all_freq[freq_indices]
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
    elif n_corr == 1:
        vis_obs = np.zeros((n_row, n_chan, 2, 2), dtype=np.complex128)
        vis_model = np.zeros((n_row, n_chan, 2, 2), dtype=np.complex128)
        flags_out = np.ones((n_row, n_chan, 2, 2), dtype=bool)
        vis_obs[:, :, 0, 0] = data[:, :, 0]
        vis_model[:, :, 0, 0] = model[:, :, 0]
        flags_out[:, :, 0, 0] = flags[:, :, 0]
    else:
        raise ValueError(f"Unexpected number of correlations: {n_corr}")

    return {
        "vis_obs": np.ascontiguousarray(vis_obs, dtype=np.complex128),
        "vis_model": np.ascontiguousarray(vis_model, dtype=np.complex128),
        "ant1": np.ascontiguousarray(ant1, dtype=np.int32),
        "ant2": np.ascontiguousarray(ant2, dtype=np.int32),
        "flags": np.ascontiguousarray(flags_out, dtype=bool),
        "time": time_arr,
        "freq": np.ascontiguousarray(freq, dtype=np.float64),
    }


def _parse_scan_selection(scans_str: str) -> List[int]:
    """Parse scan selection string: '10', '10~20', '10,15,20'."""
    scans_str = str(scans_str).strip()
    if "~" in scans_str:
        parts = scans_str.split("~")
        return list(range(int(parts[0]), int(parts[1]) + 1))
    if "," in scans_str:
        return [int(s.strip()) for s in scans_str.split(",")]
    return [int(scans_str)]


def compute_solint_grid(
    unique_times: np.ndarray,
    freq: np.ndarray,
    time_interval: str,
    freq_interval: str,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Compute solution interval grid.

    Returns
    -------
    time_edges : ndarray (n_sol_time+1,)
    freq_chunk_indices : list of ndarray
        Each element is array of channel indices for that freq chunk.
    """
    import re

    t_min, t_max = unique_times[0], unique_times[-1]
    total_time = t_max - t_min

    # Parse time interval
    ti = time_interval.strip().lower()
    if ti in ("inf", "full"):
        n_sol_time = 1
    else:
        match = re.match(r'^([\d.]+)\s*(s|sec|min|m|h|hour)$', ti)
        if match:
            val = float(match.group(1))
            unit = match.group(2)
            if unit in ("min", "m"):
                val *= 60.0
            elif unit in ("h", "hour"):
                val *= 3600.0
            n_sol_time = max(1, int(np.ceil(total_time / val)))
        else:
            n_sol_time = 1

    if n_sol_time == 1:
        time_edges = np.array([t_min, t_max + 1.0])
    else:
        time_edges = np.linspace(t_min, t_max + 1.0, n_sol_time + 1)

    # Parse freq interval
    n_chan = len(freq)
    fi = freq_interval.strip().lower()
    if fi in ("full", "inf", "spw"):
        freq_chunk_indices = [np.arange(n_chan)]
    else:
        chan_width = float(freq[1] - freq[0]) if n_chan > 1 else 1.0
        match = re.match(r'^([\d.]+)\s*(hz|khz|mhz|ghz|chan|channels?)$', fi)
        if match:
            val = float(match.group(1))
            unit = match.group(2)
            if unit == "khz":
                val *= 1e3
            elif unit == "mhz":
                val *= 1e6
            elif unit == "ghz":
                val *= 1e9
            elif unit.startswith("chan"):
                val *= abs(chan_width)

            n_chan_per_chunk = max(1, int(np.round(val / abs(chan_width))))
        else:
            n_chan_per_chunk = n_chan

        freq_chunk_indices = []
        for start in range(0, n_chan, n_chan_per_chunk):
            end = min(start + n_chan_per_chunk, n_chan)
            freq_chunk_indices.append(np.arange(start, end))

    return time_edges, freq_chunk_indices
