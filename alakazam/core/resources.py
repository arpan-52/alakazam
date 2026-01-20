"""
Resource Monitoring Module.

Monitors RAM/CPU and estimates MS memory usage for chunking decisions.
"""

import os
import gc
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger('jackal')

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil not available - RAM monitoring disabled")


def get_available_ram() -> float:
    """
    Get available system RAM in GB.

    Returns
    -------
    float
        Available RAM in gigabytes
    """
    if not HAS_PSUTIL:
        # Fallback: assume 8 GB available
        return 8.0

    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    return available_gb


def get_total_ram() -> float:
    """
    Get total system RAM in GB.

    Returns
    -------
    float
        Total RAM in gigabytes
    """
    if not HAS_PSUTIL:
        return 16.0  # Default assumption

    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    return total_gb


def estimate_ms_selection_size(
    ms_path: str,
    field: Optional[str] = None,
    spw: Optional[str] = None,
    scans: Optional[str] = None
) -> Tuple[float, int, int, int]:
    """
    Estimate memory usage for SELECTED MS data ONLY.

    CRITICAL: This estimates ONLY the data matching the selection criteria
    (field/spw/scans), NOT the entire MS file!

    Example:
    - MS file: 100 GB total with 10 fields, 100 scans
    - Selection: field="3C147", spw="0:250~1800", scans="10"
    - Estimate: ~2 GB (only the selected subset)

    We compare THIS estimated size vs available RAM, not the full MS size.

    Parameters
    ----------
    ms_path : str
        Path to MS
    field : str, optional
        Field selection (e.g., "3C147")
    spw : str, optional
        SPW selection (may include channel selection like "0:250~1800")
    scans : str, optional
        Scan selection (e.g., "10" or "10~20")

    Returns
    -------
    size_gb : float
        Estimated memory usage in GB for SELECTED data only
    n_rows : int
        Number of rows in selection
    n_chan : int
        Number of channels (after channel selection if any)
    n_corr : int
        Number of correlations
    """
    from casacore.tables import table, taql

    # Log what we're selecting
    selection_desc = []
    if field:
        selection_desc.append(f"field={field}")
    if spw:
        selection_desc.append(f"spw={spw}")
    if scans:
        selection_desc.append(f"scans={scans}")

    selection_str = ", ".join(selection_desc) if selection_desc else "all data"
    logger.info(f"Estimating memory for SELECTED data: {selection_str}")

    ms = table(ms_path, readonly=True, ack=False)

    # Get total MS size for comparison
    total_rows = ms.nrows()

    # Build TaQL selection (same as read_ms)
    conditions = []

    if field is not None:
        field_tab = table(f"{ms_path}::FIELD", readonly=True, ack=False)
        field_names = list(field_tab.getcol("NAME"))
        field_tab.close()

        if field in field_names:
            field_id = field_names.index(field)
            conditions.append(f"FIELD_ID=={field_id}")

    # Parse SPW (handle channel selection)
    chan_start, chan_end = None, None
    if spw is not None:
        spw = str(spw)

        if ":" in spw:
            spw_part, chan_part = spw.split(":", 1)
            if "~" in spw_part:
                start, end = spw_part.split("~")
                spw_ids = list(range(int(start), int(end) + 1))
            elif "," in spw_part:
                spw_ids = [int(s.strip()) for s in spw_part.split(",")]
            else:
                spw_ids = [int(spw_part)]

            # Parse channel range
            if "~" in chan_part:
                chan_start, chan_end = map(int, chan_part.split("~"))
            else:
                chan_start = chan_end = int(chan_part)
        else:
            if "~" in spw:
                start, end = spw.split("~")
                spw_ids = list(range(int(start), int(end) + 1))
            elif "," in spw:
                spw_ids = [int(s.strip()) for s in spw.split(",")]
            else:
                spw_ids = [int(spw)]

        conditions.append(f"DATA_DESC_ID IN [{','.join(map(str, spw_ids))}]")

    if scans is not None:
        scans = str(scans)
        if "~" in scans:
            start, end = scans.split("~")
            scan_ids = list(range(int(start), int(end) + 1))
        elif "," in scans:
            scan_ids = [int(s.strip()) for s in scans.split(",")]
        else:
            scan_ids = [int(scans)]
        conditions.append(f"SCAN_NUMBER IN [{','.join(map(str, scan_ids))}]")

    # Select data
    if conditions:
        query = f"SELECT * FROM $ms WHERE {' AND '.join(conditions)}"
        sel = taql(query)
    else:
        sel = ms

    # Get dimensions of SELECTED data
    n_rows = sel.nrows()

    # Get number of channels
    spw_tab = table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True, ack=False)
    n_chan_full = spw_tab.getcol("CHAN_FREQ")[0].shape[0]
    spw_tab.close()

    # Apply channel selection if present
    if chan_start is not None and chan_end is not None:
        n_chan = chan_end - chan_start + 1
    else:
        n_chan = n_chan_full

    # Get number of correlations
    n_corr = 4  # Assume full pol

    sel.close()
    ms.close()

    # Log selection statistics
    row_fraction = n_rows / max(total_rows, 1) * 100
    chan_fraction = n_chan / max(n_chan_full, 1) * 100
    logger.info(f"Selection: {n_rows}/{total_rows} rows ({row_fraction:.1f}%), {n_chan}/{n_chan_full} channels ({chan_fraction:.1f}%)")

    # Estimate memory:
    # - vis_obs: n_rows × n_chan × 2 × 2 × 16 bytes (complex128)
    # - vis_model: same
    # - flags: n_rows × n_chan × 2 × 2 × 1 byte (bool)
    # - antenna1, antenna2: n_rows × 4 bytes each
    # - time: n_rows × 8 bytes
    # - freq: n_chan × 8 bytes

    bytes_per_vis = n_rows * n_chan * 4 * 16  # 4 = 2×2 Jones
    bytes_vis_obs = bytes_per_vis
    bytes_vis_model = bytes_per_vis
    bytes_flags = n_rows * n_chan * 4 * 1
    bytes_meta = n_rows * (4 + 4 + 8) + n_chan * 8

    total_bytes = bytes_vis_obs + bytes_vis_model + bytes_flags + bytes_meta

    # Add 50% overhead for intermediate arrays
    total_bytes = int(total_bytes * 1.5)

    size_gb = total_bytes / (1024**3)

    logger.info(f"Estimated memory for SELECTED data: {size_gb:.2f} GB (not full MS!)")

    return size_gb, n_rows, n_chan, n_corr


def should_use_chunking(
    ms_size_gb: float,
    available_ram_gb: float,
    threshold_fraction: float = 0.1
) -> Tuple[bool, str]:
    """
    Decide if chunking is needed.

    Parameters
    ----------
    ms_size_gb : float
        Estimated MS selection size in GB
    available_ram_gb : float
        Available RAM in GB
    threshold_fraction : float
        Threshold: if MS > threshold × RAM, use chunking
        Default: 0.1 (10% of RAM)

    Returns
    -------
    use_chunking : bool
        Whether to use chunking
    reason : str
        Explanation for decision
    """
    threshold_gb = available_ram_gb * threshold_fraction

    if ms_size_gb <= threshold_gb:
        return False, f"MS size ({ms_size_gb:.2f} GB) <= {threshold_fraction*100:.0f}% RAM ({threshold_gb:.2f} GB)"
    else:
        return True, f"MS size ({ms_size_gb:.2f} GB) > {threshold_fraction*100:.0f}% RAM ({threshold_gb:.2f} GB)"


def cleanup_arrays(*arrays):
    """
    Delete arrays and force garbage collection.

    Parameters
    ----------
    *arrays
        Variable number of arrays to delete
    """
    for arr in arrays:
        if arr is not None:
            del arr
    gc.collect()


def print_memory_info(label: str = ""):
    """Print current memory usage."""
    if not HAS_PSUTIL:
        return

    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024**3)
    available_gb = mem.available / (1024**3)
    percent = mem.percent

    prefix = f"[{label}] " if label else ""
    logger.info(f"{prefix}Memory: {used_gb:.2f} GB used, {available_gb:.2f} GB available ({percent:.1f}%)")


__all__ = [
    'get_available_ram',
    'get_total_ram',
    'estimate_ms_selection_size',
    'should_use_chunking',
    'cleanup_arrays',
    'print_memory_info',
]
