"""
Solint (Solution Interval) Chunking.

Handles parsing and creating time/frequency chunks based on user-specified
solution intervals. The solver itself is agnostic to solints - it just solves
for a single Jones matrix. This module handles the chunking logic.
"""

import numpy as np
from typing import Tuple, List, Optional
import re


def parse_time_interval(time_interval: str) -> Optional[float]:
    """
    Parse time interval string to seconds.

    Parameters
    ----------
    time_interval : str
        Examples: '60s', '2min', '0.5h', 'inf', 'full'

    Returns
    -------
    interval_sec : float or None
        Interval in seconds, or None for 'inf'/'full'
    """
    time_interval = time_interval.strip().lower()

    if time_interval in ['inf', 'full', 'infinite']:
        return None

    # Match number + unit
    match = re.match(r'([\d.]+)\s*([a-z]+)', time_interval)
    if not match:
        raise ValueError(f"Cannot parse time_interval: {time_interval}")

    value = float(match.group(1))
    unit = match.group(2)

    if unit in ['s', 'sec', 'second', 'seconds']:
        return value
    elif unit in ['m', 'min', 'minute', 'minutes']:
        return value * 60.0
    elif unit in ['h', 'hour', 'hours']:
        return value * 3600.0
    else:
        raise ValueError(f"Unknown time unit: {unit}")


def parse_freq_interval(freq_interval: str, chan_width: float) -> Optional[float]:
    """
    Parse frequency interval string to Hz.

    Parameters
    ----------
    freq_interval : str
        Examples: '2MHz', '128kHz', 'full', 'spw'
    chan_width : float
        Channel width in Hz (needed for 'chan' units)

    Returns
    -------
    interval_hz : float or None
        Interval in Hz, or None for 'full'
    """
    freq_interval = freq_interval.strip().lower()

    if freq_interval in ['full', 'inf', 'infinite']:
        return None

    if freq_interval == 'spw':
        # SPW handled at higher level
        return None

    # Match number + unit
    match = re.match(r'([\d.]+)\s*([a-z]+)', freq_interval)
    if not match:
        # Try just number (assume Hz)
        try:
            return float(freq_interval)
        except:
            raise ValueError(f"Cannot parse freq_interval: {freq_interval}")

    value = float(match.group(1))
    unit = match.group(2)

    if unit in ['hz']:
        return value
    elif unit in ['khz']:
        return value * 1e3
    elif unit in ['mhz']:
        return value * 1e6
    elif unit in ['ghz']:
        return value * 1e9
    elif unit in ['chan', 'channel', 'channels']:
        return value * chan_width
    else:
        raise ValueError(f"Unknown frequency unit: {unit}")


def create_time_chunks(time: np.ndarray, time_interval_sec: Optional[float]) -> List[np.ndarray]:
    """
    Create time chunk indices.

    Parameters
    ----------
    time : ndarray (n_row,)
        Time stamps in seconds
    time_interval_sec : float or None
        Chunk size in seconds, or None for single chunk

    Returns
    -------
    chunks : list of ndarray
        Each element is array of row indices for that chunk
    """
    if time_interval_sec is None:
        # Single chunk with all data
        return [np.arange(len(time))]

    t_min = np.min(time)
    t_max = np.max(time)
    n_chunks = int(np.ceil((t_max - t_min) / time_interval_sec))

    if n_chunks == 0:
        n_chunks = 1

    chunks = []
    for i in range(n_chunks):
        t_start = t_min + i * time_interval_sec
        t_end = t_start + time_interval_sec
        mask = (time >= t_start) & (time < t_end)
        indices = np.where(mask)[0]
        if len(indices) > 0:
            chunks.append(indices)

    return chunks


def create_freq_chunks(freq: np.ndarray, freq_interval_hz: Optional[float]) -> List[np.ndarray]:
    """
    Create frequency chunk indices.

    Parameters
    ----------
    freq : ndarray (n_chan,)
        Frequencies in Hz
    freq_interval_hz : float or None
        Chunk bandwidth in Hz, or None for single chunk

    Returns
    -------
    chunks : list of ndarray
        Each element is array of channel indices for that chunk
    """
    if freq_interval_hz is None:
        # Single chunk with all channels
        return [np.arange(len(freq))]

    f_min = np.min(freq)
    f_max = np.max(freq)
    n_chunks = int(np.ceil((f_max - f_min) / freq_interval_hz))

    if n_chunks == 0:
        n_chunks = 1

    chunks = []
    for i in range(n_chunks):
        f_start = f_min + i * freq_interval_hz
        f_end = f_start + freq_interval_hz
        mask = (freq >= f_start) & (freq < f_end)
        indices = np.where(mask)[0]
        if len(indices) > 0:
            chunks.append(indices)

    return chunks


def extract_chunk_data(
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    flags: np.ndarray,
    time_indices: np.ndarray,
    freq_indices: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract data for a specific time/freq chunk.

    Parameters
    ----------
    vis_obs : ndarray (n_row, n_chan, 2, 2) or (n_row, 2, 2)
    vis_model : same shape
    flags : same shape
    time_indices : ndarray
        Row indices for this time chunk
    freq_indices : ndarray or None
        Channel indices for this freq chunk (None if no freq axis)

    Returns
    -------
    vis_obs_chunk : ndarray
    vis_model_chunk : ndarray
    flags_chunk : ndarray
    """
    if vis_obs.ndim == 4:
        # Has frequency axis
        vis_obs_chunk = vis_obs[time_indices, :, :, :]
        vis_model_chunk = vis_model[time_indices, :, :, :]
        flags_chunk = flags[time_indices, :, :, :]

        if freq_indices is not None:
            vis_obs_chunk = vis_obs_chunk[:, freq_indices, :, :]
            vis_model_chunk = vis_model_chunk[:, freq_indices, :, :]
            flags_chunk = flags_chunk[:, freq_indices, :, :]
    else:
        # No frequency axis (already averaged)
        vis_obs_chunk = vis_obs[time_indices, :, :]
        vis_model_chunk = vis_model[time_indices, :, :]
        flags_chunk = flags[time_indices, :, :]

    return vis_obs_chunk, vis_model_chunk, flags_chunk


def average_chunk_data(
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    flags: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Average data within a chunk (over time and freq if present).

    Solver expects data averaged within each solint chunk.

    Parameters
    ----------
    vis_obs : ndarray (n_row, n_chan, 2, 2) or (n_row, 2, 2)
    vis_model : same shape
    flags : same shape

    Returns
    -------
    vis_obs_avg : ndarray (1, 2, 2) or (1, n_chan, 2, 2)
        Averaged within chunk (keep single "row" for baseline structure)
    vis_model_avg : same
    flags_avg : same
    """
    # Mask flagged data
    vis_obs_masked = np.where(flags, np.nan, vis_obs)
    vis_model_masked = np.where(flags, np.nan, vis_model)

    # Average over time (axis 0)
    with np.errstate(invalid='ignore'):
        vis_obs_avg = np.nanmean(vis_obs_masked, axis=0, keepdims=True)
        vis_model_avg = np.nanmean(vis_model_masked, axis=0, keepdims=True)

    # Flags: True if ALL time samples flagged
    flags_avg = np.all(flags, axis=0, keepdims=True)

    return vis_obs_avg, vis_model_avg, flags_avg


__all__ = [
    'parse_time_interval',
    'parse_freq_interval',
    'create_time_chunks',
    'create_freq_chunks',
    'extract_chunk_data',
    'average_chunk_data',
]
