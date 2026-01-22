"""
ALAKAZAM Pipeline.

YAML config parsing and execution with MS reading.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Rich logging imports
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn
    from rich.logging import RichHandler
    import logging
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    import logging

from ..core import solve
from ..core.interpolation import interpolate_jones_freq, interpolate_jones_time_array, fill_flagged_from_valid
from ..core.fluxscale import compute_fluxscale, compute_fluxscale_multi_field
from ..core.solint import (
    parse_time_interval,
    parse_freq_interval,
    create_time_chunks,
    create_freq_chunks,
    extract_chunk_data,
    average_chunk_data,
)
from ..io import save_jones, load_jones, list_jones
from ..jones import jones_multiply, jones_unapply, delay_to_jones, FeedBasis

# Create console for rich output
if RICH_AVAILABLE:
    console = Console()
else:
    console = None


def interpolate_jones_for_chaining(
    jones: np.ndarray,
    freq_src: Optional[np.ndarray],
    freq_tgt: np.ndarray,
    jones_type: str,
    verbose: bool = False
) -> np.ndarray:
    """
    Interpolate or broadcast Jones matrix for chaining with next solver.

    Parameters
    ----------
    jones : ndarray
        Jones matrix from previous solver
        - Shape (n_ant, 2, 2) if freq-averaged
        - Shape (n_ant, n_freq_src, 2, 2) if freq-dependent
    freq_src : ndarray or None
        Source frequencies (None if freq-averaged)
    freq_tgt : ndarray
        Target frequencies for next solver
    jones_type : str
        Jones type (for logging)
    verbose : bool
        Print interpolation info

    Returns
    -------
    jones_interp : ndarray
        Interpolated Jones (n_ant, n_freq_tgt, 2, 2)
    """
    # Check Jones shape
    if jones.ndim == 3:
        # Freq-averaged: (n_ant, 2, 2) → broadcast to all frequencies
        n_ant = jones.shape[0]
        n_freq_tgt = len(freq_tgt)
        jones_interp = np.zeros((n_ant, n_freq_tgt, 2, 2), dtype=jones.dtype)
        for f in range(n_freq_tgt):
            jones_interp[:, f, :, :] = jones

        if verbose:
            if RICH_AVAILABLE and console:
                console.print(f"      [dim]Broadcasting {jones_type} (freq-averaged) → {n_freq_tgt} channels[/dim]")
            else:
                print(f"      Broadcasting {jones_type} (freq-averaged) → {n_freq_tgt} channels")

        return jones_interp

    elif jones.ndim == 4:
        # Freq-dependent: (n_ant, n_freq_src, 2, 2)
        n_ant, n_freq_src = jones.shape[:2]
        n_freq_tgt = len(freq_tgt)

        if n_freq_src == n_freq_tgt and freq_src is not None and np.allclose(freq_src, freq_tgt):
            # Frequencies match, no interpolation needed
            return jones

        # Need interpolation
        if freq_src is None:
            # No source freq info, assume uniform spacing
            freq_src = np.linspace(freq_tgt[0], freq_tgt[-1], n_freq_src)

        # Transpose for interpolation: (n_ant, n_freq, 2, 2) → (n_freq, n_ant, 2, 2)
        jones_transposed = np.transpose(jones, (1, 0, 2, 3))

        # Interpolate
        jones_interp_transposed = interpolate_jones_freq(
            jones_transposed, freq_src, freq_tgt, method='linear'
        )

        # Transpose back: (n_freq, n_ant, 2, 2) → (n_ant, n_freq, 2, 2)
        jones_interp = np.transpose(jones_interp_transposed, (1, 0, 2, 3))

        if verbose:
            if RICH_AVAILABLE and console:
                console.print(f"      [dim]Interpolating {jones_type}: {n_freq_src} → {n_freq_tgt} channels[/dim]")
            else:
                print(f"      Interpolating {jones_type}: {n_freq_src} → {n_freq_tgt} channels")

        return jones_interp

    else:
        raise ValueError(f"Unexpected Jones shape: {jones.shape}")


def setup_logging(log_dir: str = ".", use_rich: bool = True) -> logging.Logger:
    """
    Setup logging with optional rich formatting and file output.

    Parameters
    ----------
    log_dir : str
        Directory to write log files
    use_rich : bool
        Use rich formatting if available

    Returns
    -------
    logger : logging.Logger
        Configured logger
    """
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"alakazam_run_{timestamp}.log"

    # Create logger
    logger = logging.getLogger("ALAKAZAM")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler (always plain text)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter('[%(asctime)s] %(levelname)-8s %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # Console handler (rich if available)
    if use_rich and RICH_AVAILABLE:
        ch = RichHandler(console=console, show_time=True, show_path=False)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
    else:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter('[%(asctime)s] %(levelname)-8s %(message)s', datefmt='%H:%M:%S')
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")

    return logger


# =============================================================================
# MS Helper Functions
# =============================================================================

def get_all_spws(ms_path: str) -> List[int]:
    """
    Get all SPW IDs from Measurement Set.

    Parameters
    ----------
    ms_path : str
        Path to MS

    Returns
    -------
    spw_ids : list of int
        All SPW IDs in MS
    """
    from casacore.tables import table

    # Read DATA_DESCRIPTION table to get all SPW IDs
    ddtab = table(f"{ms_path}::DATA_DESCRIPTION", readonly=True, ack=False)
    spw_ids = list(ddtab.getcol("SPECTRAL_WINDOW_ID"))
    ddtab.close()

    return sorted(set(spw_ids))


def parse_spw_selection(spw: Optional[str], ms_path: str) -> List[int]:
    """
    Parse SPW selection string and return list of SPW IDs.

    Parameters
    ----------
    spw : str or None
        SPW selection string:
        - None, "all", "auto": all SPWs
        - "0": SPW 0
        - "0~3": SPWs 0, 1, 2, 3
        - "0,2,4": SPWs 0, 2, 4
        - "0:250~1800": SPW 0 with channel selection (returns [0])
    ms_path : str
        Path to MS (to auto-detect SPWs)

    Returns
    -------
    spw_ids : list of int
        SPW IDs to process
    """
    if spw is None or spw in ["all", "auto", ""]:
        # Auto-detect all SPWs
        return get_all_spws(ms_path)

    spw = str(spw)

    # Handle channel selection: "0:250~1800" → just SPW 0
    if ":" in spw:
        spw_part = spw.split(":", 1)[0]
    else:
        spw_part = spw

    # Parse SPW IDs
    if "~" in spw_part:
        start, end = spw_part.split("~")
        return list(range(int(start), int(end) + 1))
    elif "," in spw_part:
        return [int(s.strip()) for s in spw_part.split(",")]
    else:
        return [int(spw_part)]


def format_spw_string(spw: Optional[str], spw_id: int) -> str:
    """
    Format SPW string for a specific SPW ID.

    Parameters
    ----------
    spw : str or None
        Original SPW specification
    spw_id : int
        Specific SPW ID to use

    Returns
    -------
    spw_str : str
        SPW string with channel selection if present

    Examples
    --------
    >>> format_spw_string("0~3:250~1800", 2)
    "2:250~1800"
    >>> format_spw_string("all", 1)
    "1"
    """
    if spw is None or spw in ["all", "auto", ""]:
        return str(spw_id)

    spw = str(spw)

    # If channel selection present, preserve it
    if ":" in spw:
        _, chan_part = spw.split(":", 1)
        return f"{spw_id}:{chan_part}"
    else:
        return str(spw_id)


# =============================================================================
# MS Reading (casacore)
# =============================================================================

def get_ms_time_array(
    ms_path: str,
    field: str = None,
    spw: str = None,
    scans: str = None,
) -> np.ndarray:
    """
    Read just the TIME column from MS (lightweight operation).
    Used for pre-computing time chunks before batched loading.
    """
    from casacore.tables import table, taql

    ms = table(ms_path, readonly=True, ack=False)

    # Build TaQL selection (same logic as read_ms)
    conditions = []

    if field is not None:
        field_tab = table(f"{ms_path}::FIELD", readonly=True, ack=False)
        field_names = list(field_tab.getcol("NAME"))
        field_tab.close()

        if field in field_names:
            field_id = field_names.index(field)
            conditions.append(f"FIELD_ID=={field_id}")
        else:
            raise ValueError(f"Field '{field}' not found. Available: {field_names}")

    if spw is not None:
        spw = str(spw)
        # Parse SPW (ignore channel selection for time array)
        if ":" in spw:
            spw_part = spw.split(":", 1)[0]
        else:
            spw_part = spw

        if "~" in spw_part:
            start, end = spw_part.split("~")
            spw_ids = list(range(int(start), int(end) + 1))
        elif "," in spw_part:
            spw_ids = [int(s.strip()) for s in spw_part.split(",")]
        else:
            spw_ids = [int(spw_part)]
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

    # Select and read only TIME column
    if conditions:
        query = f"SELECT TIME FROM $ms WHERE {' AND '.join(conditions)}"
        sel = taql(query)
    else:
        sel = ms

    time_arr = sel.getcol("TIME")
    sel.close()
    ms.close()

    return time_arr


def read_ms(
    ms_path: str,
    field: str = None,
    spw: str = None,
    scans: str = None,
    time_min: float = None,
    time_max: float = None,
    data_col: str = "DATA",
    model_col: str = "MODEL_DATA",
) -> Dict[str, Any]:
    """
    Read visibilities from Measurement Set.
    
    Returns dict with:
        vis_obs: (n_row, n_chan, 2, 2) complex128
        vis_model: (n_row, n_chan, 2, 2) complex128
        antenna1: (n_row,) int32
        antenna2: (n_row,) int32
        time: (n_row,) float64
        freq: (n_chan,) float64
        flags: (n_row, n_chan, 2, 2) bool
        n_ant: int
        feed_basis: FeedBasis
    """
    from casacore.tables import table, taql
    
    ms = table(ms_path, readonly=True, ack=False)
    
    # Build TaQL selection
    conditions = []
    
    if field is not None:
        # Get field ID from name
        field_tab = table(f"{ms_path}::FIELD", readonly=True, ack=False)
        field_names = list(field_tab.getcol("NAME"))
        field_tab.close()
        
        if field in field_names:
            field_id = field_names.index(field)
            conditions.append(f"FIELD_ID=={field_id}")
        else:
            raise ValueError(f"Field '{field}' not found. Available: {field_names}")
    
    # Track channel selection for later
    chan_start = None
    chan_end = None

    if spw is not None:
        # Convert to string if needed
        spw = str(spw)

        # Parse spw string like "0", "0~3", "0,2,4", "0:250~1800"
        if ":" in spw:
            # SPW with channel selection: "0:250~1800"
            spw_part, chan_part = spw.split(":", 1)

            # Parse SPW ID(s)
            if "~" in spw_part:
                start, end = spw_part.split("~")
                spw_ids = list(range(int(start), int(end) + 1))
            elif "," in spw_part:
                spw_ids = [int(s.strip()) for s in spw_part.split(",")]
            else:
                spw_ids = [int(spw_part)]

            # Parse channel range: "250~1800"
            if "~" in chan_part:
                chan_start, chan_end = map(int, chan_part.split("~"))
            else:
                # Single channel
                chan_start = chan_end = int(chan_part)
        else:
            # No channel selection, just SPW IDs
            if "~" in spw:
                start, end = spw.split("~")
                spw_ids = list(range(int(start), int(end) + 1))
            elif "," in spw:
                spw_ids = [int(s.strip()) for s in spw.split(",")]
            else:
                spw_ids = [int(spw)]

        conditions.append(f"DATA_DESC_ID IN [{','.join(map(str, spw_ids))}]")

    if scans is not None:
        # Convert to string if needed (YAML may parse as int)
        scans = str(scans)

        # Parse scan selection like "10", "10~20", "10,15,20"
        if "~" in scans:
            start, end = scans.split("~")
            scan_ids = list(range(int(start), int(end) + 1))
        elif "," in scans:
            scan_ids = [int(s.strip()) for s in scans.split(",")]
        else:
            scan_ids = [int(scans)]
        conditions.append(f"SCAN_NUMBER IN [{','.join(map(str, scan_ids))}]")

    if time_min is not None:
        conditions.append(f"TIME >= {time_min}")

    if time_max is not None:
        conditions.append(f"TIME <= {time_max}")  # Inclusive upper bound

    # Select data
    if conditions:
        query = f"SELECT * FROM $ms WHERE {' AND '.join(conditions)}"
        sel = taql(query)
    else:
        sel = ms

    # Check if selection is empty
    if sel.nrows() == 0:
        sel.close()
        ms.close()
        raise ValueError(
            f"No data found for selection. "
            f"Field: {field}, SPW: {spw}, Scans: {scans}, "
            f"Time range: [{time_min}, {time_max}]"
        )

    # Read columns
    antenna1 = sel.getcol("ANTENNA1").astype(np.int32)
    antenna2 = sel.getcol("ANTENNA2").astype(np.int32)
    time = sel.getcol("TIME")
    flags = sel.getcol("FLAG")
    
    # Data
    if data_col in sel.colnames():
        data = sel.getcol(data_col)
    else:
        raise ValueError(f"Column {data_col} not in MS")
    
    # Model
    if model_col in sel.colnames():
        model = sel.getcol(model_col)
    else:
        # Default to unity model
        model = np.ones_like(data)
    
    sel.close()
    ms.close()
    
    # Get frequencies from SPW table
    spw_tab = table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True, ack=False)
    freq = spw_tab.getcol("CHAN_FREQ")[0]  # First SPW
    spw_tab.close()

    # Apply channel selection if specified
    if chan_start is not None and chan_end is not None:
        # Slice data to selected channels
        data = data[:, chan_start:chan_end+1, :]
        model = model[:, chan_start:chan_end+1, :]
        flags = flags[:, chan_start:chan_end+1, :]
        freq = freq[chan_start:chan_end+1]

    # Get feed basis from FEED or POLARIZATION table
    pol_tab = table(f"{ms_path}::POLARIZATION", readonly=True, ack=False)
    corr_type = pol_tab.getcol("CORR_TYPE")[0]
    pol_tab.close()
    
    # Correlation types: 9,10,11,12 = XX,XY,YX,YY (linear)
    #                    5,6,7,8 = RR,RL,LR,LL (circular)
    if corr_type[0] in [9, 10, 11, 12]:
        feed_basis = FeedBasis.LINEAR
    elif corr_type[0] in [5, 6, 7, 8]:
        feed_basis = FeedBasis.CIRCULAR
    else:
        feed_basis = FeedBasis.LINEAR  # Default
    
    # Number of antennas
    ant_tab = table(f"{ms_path}::ANTENNA", readonly=True, ack=False)
    n_ant = ant_tab.nrows()
    ant_tab.close()
    
    # Reshape to (n_row, n_chan, 2, 2)
    n_row, n_chan, n_corr = data.shape
    
    if n_corr == 4:
        # Already 4 correlations
        vis_obs = data.reshape(n_row, n_chan, 2, 2)
        vis_model = model.reshape(n_row, n_chan, 2, 2)
        flags_out = flags.reshape(n_row, n_chan, 2, 2)
    elif n_corr == 2:
        # Only XX, YY - expand to 2x2 diagonal
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
    
    return {
        'vis_obs': np.ascontiguousarray(vis_obs, dtype=np.complex128),
        'vis_model': np.ascontiguousarray(vis_model, dtype=np.complex128),
        'antenna1': np.ascontiguousarray(antenna1, dtype=np.int32),
        'antenna2': np.ascontiguousarray(antenna2, dtype=np.int32),
        'time': time,
        'freq': np.ascontiguousarray(freq, dtype=np.float64),
        'flags': np.ascontiguousarray(flags_out, dtype=bool),
        'n_ant': n_ant,
        'feed_basis': feed_basis,
    }


def write_corrected(
    ms_path: str,
    jones_list: List[np.ndarray],
    tables: List[str],
    jones_types: List[str],
    output_col: str = "CORRECTED_DATA",
    spw: str = None,
    field: str = None,
    verbose: bool = True,
):
    """
    Apply Jones corrections and write to MS.

    Properly handles:
    - Multi-dimensional Jones (time/freq dependent)
    - Interpolation to MS time AND frequency grids
    - Freq-dependent Jones (K, B)
    - Time-dependent Jones (G)
    - Composite Jones multiplication in user-specified order
    - Different Jones from different H5 files
    """
    from casacore.tables import table

    if verbose:
        if RICH_AVAILABLE and console:
            console.print(f"\n[bold]Applying calibration to {ms_path}[/bold]")
            console.print(f"   Output column: [cyan]{output_col}[/cyan]")
        else:
            print(f"\nApplying calibration to {ms_path}")
            print(f"  Output column: {output_col}")

    # Read MS metadata to get frequencies and times
    ms_data = read_ms(ms_path, field=field, spw=spw, data_col="DATA", model_col="DATA")
    ms_freq = ms_data['freq']
    ms_time = ms_data['time']
    n_ant = ms_data['n_ant']

    # Get unique times for interpolation
    ms_time_unique = np.unique(ms_time)

    if verbose:
        if RICH_AVAILABLE and console:
            console.print(f"   MS: {ms_data['vis_obs'].shape[0]} rows, {len(ms_freq)} channels, {n_ant} antennas")
            console.print(f"   Time samples: {len(ms_time_unique)}, Frequency channels: {len(ms_freq)}")
        else:
            print(f"  MS: {ms_data['vis_obs'].shape[0]} rows, {len(ms_freq)} channels, {n_ant} antennas")
            print(f"  Time samples: {len(ms_time_unique)}, Frequency channels: {len(ms_freq)}")

    # Load and prepare all Jones matrices
    jones_all = []
    jones_info = []

    for i, (jt, tbl) in enumerate(zip(jones_types, tables)):
        if verbose:
            if RICH_AVAILABLE and console:
                console.print(f"\n   [bold]Loading {jt}[/bold] from {tbl}")
            else:
                print(f"\n  Loading {jt} from {tbl}")

        # Load Jones data
        data = load_jones(tbl, jt)
        jones_loaded = data['jones']
        freq_loaded = data.get('freq', None)
        time_loaded = data.get('time', None)
        weights_loaded = data.get('weights', None)

        # Check weights and warn if any chunks are flagged
        if weights_loaded is not None:
            n_flagged = np.sum(weights_loaded == 0.0)
            if n_flagged > 0 and verbose:
                if RICH_AVAILABLE and console:
                    console.print(f"      [yellow]Warning: {n_flagged} of {weights_loaded.size} chunks have weight=0 (flagged)[/yellow]")
                else:
                    print(f"    Warning: {n_flagged} of {weights_loaded.size} chunks have weight=0 (flagged)")

        # Get shape info
        if verbose:
            if RICH_AVAILABLE and console:
                console.print(f"      Loaded shape: [cyan]{jones_loaded.shape}[/cyan]")
            else:
                print(f"    Loaded shape: {jones_loaded.shape}")

        # Determine Jones dimensionality
        is_time_dependent = False
        is_freq_dependent = False
        jones_work = None

        if jones_loaded.ndim == 5:
            # (n_time, n_freq, n_ant, 2, 2) - both time and freq dependent
            is_time_dependent = True
            is_freq_dependent = True
            jones_work = jones_loaded

        elif jones_loaded.ndim == 4:
            # Could be (n_time, n_ant, 2, 2) or (n_freq, n_ant, 2, 2)
            if freq_loaded is not None and len(freq_loaded) == jones_loaded.shape[0]:
                # (n_freq, n_ant, 2, 2) - freq dependent only
                is_freq_dependent = True
                jones_work = jones_loaded
            elif time_loaded is not None and len(time_loaded) == jones_loaded.shape[0]:
                # (n_time, n_ant, 2, 2) - time dependent only
                is_time_dependent = True
                jones_work = jones_loaded
            else:
                # Fallback: assume freq-dependent if freq is provided
                if freq_loaded is not None:
                    is_freq_dependent = True
                    jones_work = jones_loaded
                else:
                    # Assume time-dependent
                    is_time_dependent = True
                    jones_work = jones_loaded

        elif jones_loaded.ndim == 3:
            # (n_ant, 2, 2) - neither time nor freq dependent
            jones_work = jones_loaded

        else:
            raise ValueError(f"Unexpected Jones shape: {jones_loaded.shape}")

        # Fill flagged chunks by interpolating from valid neighbors (weight-aware)
        if weights_loaded is not None and jones_work is not None:
            n_flagged = np.sum(weights_loaded == 0.0)
            if n_flagged > 0:
                # Need to reshape to 5D for fill_flagged_from_valid
                jones_reshaped = None
                weights_reshaped = None

                if jones_work.ndim == 5:
                    # (n_time, n_freq, n_ant, 2, 2) - already 5D
                    jones_reshaped = jones_work
                    weights_reshaped = weights_loaded
                elif jones_work.ndim == 4:
                    if is_time_dependent and not is_freq_dependent:
                        # (n_time, n_ant, 2, 2) → (n_time, 1, n_ant, 2, 2)
                        jones_reshaped = jones_work[:, np.newaxis, :, :, :]
                        weights_reshaped = weights_loaded[:, :1]  # (n_time, 1)
                    elif is_freq_dependent and not is_time_dependent:
                        # (n_freq, n_ant, 2, 2) → (1, n_freq, n_ant, 2, 2)
                        jones_reshaped = jones_work[np.newaxis, :, :, :, :]
                        weights_reshaped = weights_loaded[:1, :]  # (1, n_freq)
                elif jones_work.ndim == 3:
                    # (n_ant, 2, 2) → (1, 1, n_ant, 2, 2) - no interpolation needed
                    jones_reshaped = None

                if jones_reshaped is not None:
                    # Fill flagged chunks by interpolating from valid neighbors
                    jones_filled = fill_flagged_from_valid(
                        jones_reshaped,
                        weights_reshaped,
                        time=time_loaded if is_time_dependent else None,
                        freq=freq_loaded if is_freq_dependent else None,
                        method='linear'
                    )

                    # Reshape back to original shape
                    if jones_work.ndim == 5:
                        jones_work = jones_filled
                    elif jones_work.ndim == 4:
                        if is_time_dependent and not is_freq_dependent:
                            jones_work = jones_filled[:, 0, :, :, :]
                        elif is_freq_dependent and not is_time_dependent:
                            jones_work = jones_filled[0, :, :, :, :]

                    if verbose:
                        if RICH_AVAILABLE and console:
                            console.print(f"      [yellow]Filled {n_flagged} flagged chunks by interpolation[/yellow]")
                        else:
                            print(f"    Filled {n_flagged} flagged chunks by interpolation")

        # Interpolate in TIME if needed
        if is_time_dependent:
            if time_loaded is None:
                raise ValueError(f"Time-dependent Jones {jt} has no time metadata")

            # Check if interpolation is needed
            if len(time_loaded) != len(ms_time_unique) or not np.allclose(time_loaded, ms_time_unique):
                if verbose:
                    if RICH_AVAILABLE and console:
                        console.print(f"      [yellow]Interpolating time: {len(time_loaded)} → {len(ms_time_unique)} samples[/yellow]")
                    else:
                        print(f"    Interpolating time: {len(time_loaded)} → {len(ms_time_unique)} samples")

                # Interpolate to MS time grid
                jones_work = interpolate_jones_time_array(jones_work, time_loaded, ms_time_unique, method='linear')

        # After time interpolation, shape is:
        # - (n_time_ms, n_freq, n_ant, 2, 2) if both time & freq dependent
        # - (n_time_ms, n_ant, 2, 2) if time dependent only
        # - (n_freq, n_ant, 2, 2) if freq dependent only
        # - (n_ant, 2, 2) if neither

        # Interpolate in FREQUENCY if needed
        if is_freq_dependent:
            if freq_loaded is None:
                raise ValueError(f"Freq-dependent Jones {jt} has no freq metadata")

            # Check if interpolation is needed
            if len(freq_loaded) != len(ms_freq) or not np.allclose(freq_loaded, ms_freq):
                if verbose:
                    if RICH_AVAILABLE and console:
                        console.print(f"      [yellow]Interpolating freq: {len(freq_loaded)} → {len(ms_freq)} channels[/yellow]")
                    else:
                        print(f"    Interpolating freq: {len(freq_loaded)} → {len(ms_freq)} channels")

                # Handle both cases: with and without time dimension
                if is_time_dependent:
                    # (n_time_ms, n_freq, n_ant, 2, 2) → interpolate each time slice
                    n_time_ms = jones_work.shape[0]
                    jones_interp = np.zeros((n_time_ms, len(ms_freq), n_ant, 2, 2), dtype=np.complex128)
                    for t in range(n_time_ms):
                        jones_interp[t] = interpolate_jones_freq(jones_work[t], freq_loaded, ms_freq, method='linear')
                    jones_work = jones_interp
                else:
                    # (n_freq, n_ant, 2, 2) → interpolate directly
                    jones_work = interpolate_jones_freq(jones_work, freq_loaded, ms_freq, method='linear')

        # Broadcast to time dimension if not time-dependent
        if not is_time_dependent:
            if is_freq_dependent:
                # (n_freq, n_ant, 2, 2) → (n_time_ms, n_freq, n_ant, 2, 2)
                jones_final = np.tile(jones_work[np.newaxis, :, :, :, :], (len(ms_time_unique), 1, 1, 1, 1))
            else:
                # (n_ant, 2, 2) → (n_time_ms, n_freq, n_ant, 2, 2)
                jones_final = np.tile(jones_work[np.newaxis, np.newaxis, :, :, :], (len(ms_time_unique), len(ms_freq), 1, 1, 1))

            if verbose:
                if RICH_AVAILABLE and console:
                    console.print(f"      Time-independent: broadcasting to {len(ms_time_unique)} time samples")
                else:
                    print(f"    Time-independent: broadcasting to {len(ms_time_unique)} time samples")
        else:
            # Already time-dependent, just need to broadcast frequency if needed
            if not is_freq_dependent:
                # (n_time_ms, n_ant, 2, 2) → (n_time_ms, n_freq, n_ant, 2, 2)
                jones_final = np.tile(jones_work[:, np.newaxis, :, :, :], (1, len(ms_freq), 1, 1, 1))

                if verbose:
                    if RICH_AVAILABLE and console:
                        console.print(f"      Freq-averaged: broadcasting to {len(ms_freq)} channels")
                    else:
                        print(f"    Freq-averaged: broadcasting to {len(ms_freq)} channels")
            else:
                # Already both time and freq dependent
                jones_final = jones_work

        # Final shape should be (n_time_ms, n_freq, n_ant, 2, 2)
        jones_all.append(jones_final)
        jones_info.append({'type': jt, 'table': tbl})

    # Composite Jones multiplication in user-specified order
    if verbose:
        if RICH_AVAILABLE and console:
            console.print(f"\n   [bold]Compositing Jones matrices:[/bold]")
            for i, info in enumerate(jones_info):
                console.print(f"      {i+1}. [cyan]{info['type']}[/cyan]")
        else:
            print(f"\n  Compositing Jones matrices:")
            for i, info in enumerate(jones_info):
                print(f"    {i+1}. {info['type']}")

    # Composite: J_total = J_N @ ... @ J_2 @ J_1
    # Shape: (n_time_ms, n_freq, n_ant, 2, 2)
    J_composite = jones_all[0].copy()
    for Jones_i in jones_all[1:]:
        # Multiply per time and frequency
        n_time_ms, n_freq = J_composite.shape[:2]
        for t in range(n_time_ms):
            for f in range(n_freq):
                J_composite[t, f] = jones_multiply(Jones_i[t, f], J_composite[t, f])

    # Read and correct MS data
    if verbose:
        if RICH_AVAILABLE and console:
            console.print(f"\n   [bold]Applying composite Jones to visibilities[/bold]")
        else:
            print(f"\n  Applying composite Jones to visibilities")

    ms = table(ms_path, readonly=False, ack=False)
    data = ms.getcol("DATA")
    antenna1 = ms.getcol("ANTENNA1").astype(np.int32)
    antenna2 = ms.getcol("ANTENNA2").astype(np.int32)

    n_row, n_chan, n_corr = data.shape

    if n_corr == 4:
        vis = data.reshape(n_row, n_chan, 2, 2)
    else:
        vis = np.zeros((n_row, n_chan, 2, 2), dtype=np.complex128)
        vis[:, :, 0, 0] = data[:, :, 0]
        vis[:, :, 1, 1] = data[:, :, 1]

    # Create mapping from MS row times to unique time indices
    time_indices = np.searchsorted(ms_time_unique, ms_time)

    # Apply correction per row and channel: V_corrected = J^{-1}_i @ V @ J^{-H}_j
    for row in range(n_row):
        t_idx = time_indices[row]
        for c in range(n_chan):
            # Get Jones for this time and channel
            J = J_composite[t_idx, c]
            # Apply to this row, channel
            vis[row:row+1, c] = jones_unapply(J, vis[row:row+1, c], antenna1[row:row+1], antenna2[row:row+1])

    # Write corrected data
    if n_corr == 4:
        corrected = vis.reshape(n_row, n_chan, 4)
    else:
        corrected = np.zeros((n_row, n_chan, 2), dtype=np.complex128)
        corrected[:, :, 0] = vis[:, :, 0, 0]
        corrected[:, :, 1] = vis[:, :, 1, 1]

    # Create column if doesn't exist
    if output_col not in ms.colnames():
        from casacore.tables import makecoldesc
        desc = ms.getcoldesc("DATA")
        desc['name'] = output_col
        ms.addcols(makecoldesc(output_col, desc))

    ms.putcol(output_col, corrected)
    ms.close()

    if verbose:
        if RICH_AVAILABLE and console:
            console.print(f"   [green]✓ Written to column {output_col}[/green]")
        else:
            print(f"  ✓ Written to column {output_col}")


# =============================================================================
# Config Parsing
# =============================================================================

def parse_config(config_path: str) -> Dict[str, Any]:
    """Parse YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def expand_list(value, n: int) -> List:
    """Expand single value to list of length n."""
    if isinstance(value, list):
        if len(value) == n:
            return value
        elif len(value) == 1:
            return value * n
        else:
            raise ValueError(f"List length {len(value)} != {n}")
    return [value] * n


# =============================================================================
# Parallel Chunk Solving with Shared Memory
# =============================================================================

# Global batch data for workers (set before Pool creation, shared via fork)
_batch_data = {}

def _solve_chunk_worker(args):
    """
    Worker function for parallel chunk solving with shared memory.

    Uses global _batch_data to access large arrays without copying.
    Only small parameters are passed via args.

    Returns: (t_idx, f_idx, jones_chunk, params_chunk, info_chunk, weight)
        weight: 1.0 if solve succeeded, 0.0 if failed/empty
    """
    (
        t_idx, f_idx, t_start_chunk, t_end_chunk, freq_chunk_indices,
        jt, n_ant_working, ref_ant, phase_only,
        rfi_sigma, max_iter, tol
    ) = args

    # Access batch data from global (no copy, shared via fork)
    vis_obs_batch = _batch_data['vis_obs']
    vis_model_batch = _batch_data['vis_model']
    flags_batch = _batch_data['flags']
    antenna1_batch_remapped = _batch_data['antenna1_remapped']
    antenna2_batch_remapped = _batch_data['antenna2_remapped']
    time_arr_batch = _batch_data['time']
    freq = _batch_data['freq']
    pre_jones_list = _batch_data['pre_jones_list']

    # Filter batch data to this time chunk
    time_mask = (time_arr_batch >= t_start_chunk) & (time_arr_batch <= t_end_chunk)
    time_chunk_indices = np.where(time_mask)[0]

    # Check for empty chunk
    if len(time_chunk_indices) == 0:
        jones_empty = np.zeros((n_ant_working, 2, 2), dtype=np.complex128)
        return (t_idx, f_idx, jones_empty, None, {'cost_init': 0.0, 'cost_final': 0.0, 'nfev': 0}, 0.0)

    # Extract chunk data
    vis_obs_chunk, vis_model_chunk, flags_chunk = extract_chunk_data(
        vis_obs_batch, vis_model_batch, flags_batch,
        time_chunk_indices, freq_chunk_indices
    )

    # Get baseline arrays (use REMAPPED indices)
    ant1_chunk = antenna1_batch_remapped[time_chunk_indices]
    ant2_chunk = antenna2_batch_remapped[time_chunk_indices]

    # Determine frequency array
    if freq_chunk_indices is not None:
        freq_chunk = freq[freq_chunk_indices]
    else:
        freq_chunk = freq

    # For K: keep full freq axis; others: average
    if jt.upper() == 'K':
        freq_solve = freq_chunk
    else:
        freq_solve = None

    # Prepare pre_jones for this chunk
    pre_jones_chunk = None
    if pre_jones_list:
        pre_jones_chunk = []
        for pj in pre_jones_list:
            if pj.ndim == 4:
                if freq_chunk_indices is not None:
                    pj_chunk = pj[:, freq_chunk_indices, :, :]
                else:
                    pj_chunk = pj
            else:
                pj_chunk = pj
            pre_jones_chunk.append(pj_chunk)

    # Solve for this chunk
    try:
        jones_chunk, params_chunk, info_chunk = solve(
            jt,
            vis_obs_chunk,
            vis_model_chunk,
            ant1_chunk,
            ant2_chunk,
            n_ant_working,
            freq=freq_solve,
            flags=flags_chunk,
            pre_jones=pre_jones_chunk if pre_jones_chunk else None,
            ref_ant=ref_ant,
            phase_only=phase_only,
            rfi_sigma=rfi_sigma,
            max_iter=max_iter,
            tol=tol,
            verbose=False,
        )
        weight = 1.0  # Success
    except Exception as e:
        # Solve failed - return zeros with weight=0
        jones_chunk = np.zeros((n_ant_working, 2, 2), dtype=np.complex128)
        params_chunk = None
        info_chunk = {'cost_init': 0.0, 'cost_final': 0.0, 'nfev': 0, 'error': str(e)}
        weight = 0.0

    return (t_idx, f_idx, jones_chunk, params_chunk, info_chunk, weight)


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(config_path: str, verbose: bool = True):
    """
    Run calibration pipeline from YAML config.
    
    Config format:
    ```yaml
    ms_files:
      - path: flux_cal.ms
        fields: [3C147, 3C286]
    
    solve:
      - jones: [K, B, G]
        ms: flux_cal.ms
        field: [3C147, 3C147, 3C147]
        ref_ant: 0
        output: cal.h5
    
    apply:
      - ms: target.ms
        jones: [K, B, G]
        tables: [cal.h5]
        output_col: CORRECTED_DATA
    ```
    """
    cfg = parse_config(config_path)

    # Setup logging
    logger = setup_logging(log_dir=".", use_rich=(verbose and RICH_AVAILABLE))

    if verbose:
        if RICH_AVAILABLE and console:
            # Rich formatted header
            console.print(Panel.fit(
                "[bold cyan]ALAKAZAM[/bold cyan]: A Radio Interferometry Calibrator\n"
                "[dim]Developed by Arpan Pal, 2026[/dim]",
                border_style="cyan"
            ))
        else:
            print("\n" + "="*60)
            print("ALAKAZAM: A Radio Interferometry Calibrator")
            print("Developed by Arpan Pal, 2026")
            print("="*60)

    # Process solve entries
    for solve_entry in cfg.get('solve', []):
        jones_types = solve_entry.get('jones', [])
        n_terms = len(jones_types)
        
        if n_terms == 0:
            continue
        
        ms_path = solve_entry.get('ms')
        fields = expand_list(solve_entry.get('field', ''), n_terms)
        ref_ant = solve_entry.get('ref_ant', 0)
        phase_only = expand_list(solve_entry.get('phase_only', False), n_terms)
        freq_interval = expand_list(solve_entry.get('freq_interval', 'full'), n_terms)
        time_interval = expand_list(solve_entry.get('time_interval', 'inf'), n_terms)
        spw = solve_entry.get('spw', None)
        scans = solve_entry.get('scans', None)
        rfi_sigma = solve_entry.get('rfi_sigma', 5.0)
        max_iter = solve_entry.get('max_iter', 100)
        tol = solve_entry.get('tol', 1e-10)
        n_workers = solve_entry.get('n_workers', None)  # None = auto-detect
        output_table = solve_entry.get('output', 'cal.h5')
        
        pre_apply = solve_entry.get('pre_apply', [])
        pre_tables = solve_entry.get('pre_tables', [])
        
        # Parse SPW selection
        spw_ids = parse_spw_selection(spw, ms_path)
        n_spws = len(spw_ids)

        if verbose:
            if RICH_AVAILABLE and console:
                # Rich formatted MS info
                console.print(f"\n[bold]MS:[/bold] {ms_path}")
                console.print(f"   [bold]Field:[/bold] {fields[0] if len(set(fields)) == 1 else fields}")
                if spw is None or spw in ["all", "auto", ""]:
                    console.print(f"   [bold]SPWs:[/bold] auto → detected {n_spws} SPW(s): {spw_ids}")
                else:
                    console.print(f"   [bold]SPWs:[/bold] {spw} → {n_spws} SPW(s): {spw_ids}")
                console.print(f"   [bold]Solving:[/bold] {', '.join(jones_types)}")
                console.print(f"   [bold]Output:[/bold] {output_table}")
            else:
                print(f"\nMS: {ms_path}")
                if spw is None or spw in ["all", "auto", ""]:
                    print(f"SPW: auto (detected {n_spws} SPWs: {spw_ids})")
                else:
                    print(f"SPW: {spw} → {n_spws} SPW(s): {spw_ids}")
                print(f"Solving: {', '.join(jones_types)}")
                print(f"Output: {output_table}")

        # Import cleanup utilities (for future batched loading)
        from ..core import cleanup_arrays

        # Iterate over SPWs
        for spw_idx, spw_id in enumerate(spw_ids):
            # Format SPW string for this specific SPW
            spw_str = format_spw_string(spw, spw_id)

            if verbose and n_spws > 1:
                if RICH_AVAILABLE and console:
                    console.print(f"\n[bold cyan]{'━'*60}[/bold cyan]")
                    console.print(f"[bold cyan]Processing SPW {spw_id} ({spw_idx+1}/{n_spws})[/bold cyan]")
                    console.print(f"[bold cyan]{'━'*60}[/bold cyan]")
                else:
                    print(f"\n{'='*60}")
                    print(f"Processing SPW {spw_id} ({spw_idx+1}/{n_spws})")
                    print(f"{'='*60}")

            # Track solved Jones for chaining (per SPW)
            # Store as dict with 'jones' and 'freq' keys for interpolation
            solved_jones = {}

            for idx, jt in enumerate(jones_types):
                field = fields[idx]

                if verbose:
                    if RICH_AVAILABLE and console:
                        if n_spws > 1:
                            console.print(f"\n[bold yellow]Solving {jt}[/bold yellow] on field=[cyan]{field}[/cyan], SPW [cyan]{spw_id}[/cyan]")
                        else:
                            console.print(f"\n[bold yellow]Solving {jt}[/bold yellow] on field=[cyan]{field}[/cyan]")
                    else:
                        if n_spws > 1:
                            print(f"\n--- {jt} on field={field}, SPW {spw_id} ---")
                        else:
                            print(f"\n--- {jt} on field={field} ---")

                # STEP 1: Load small sample to determine ACTUAL data characteristics
                # Don't use header info - use actual data to determine working antennas and integration time
                from casacore.tables import table
                ms_temp = table(ms_path, readonly=True, ack=False)

                # Build selection for sample
                conditions_sample = []
                if field is not None:
                    field_tab = table(f"{ms_path}::FIELD", readonly=True, ack=False)
                    field_names = list(field_tab.getcol("NAME"))
                    field_tab.close()
                    if field in field_names:
                        field_id = field_names.index(field)
                        conditions_sample.append(f"FIELD_ID=={field_id}")

                # Get SPW ID for this specific SPW string
                if spw_str is not None:
                    spw_part = spw_str.split(":")[0] if ":" in spw_str else spw_str
                    spw_id_int = int(spw_part)
                    conditions_sample.append(f"DATA_DESC_ID=={spw_id_int}")

                # Select first 1000 rows to analyze
                if conditions_sample:
                    from casacore.tables import taql
                    query = f"SELECT * FROM $ms_temp WHERE {' AND '.join(conditions_sample)} LIMIT 1000"
                    sel_sample = taql(query)
                else:
                    sel_sample = ms_temp.query("", limit=1000)

                if sel_sample.nrows() == 0:
                    sel_sample.close()
                    ms_temp.close()
                    raise ValueError(f"No data found for field={field}, SPW={spw_str}")

                # Get ACTUAL working antennas from data
                ant1_sample = sel_sample.getcol("ANTENNA1")
                ant2_sample = sel_sample.getcol("ANTENNA2")
                working_ants = np.unique(np.concatenate([ant1_sample, ant2_sample]))
                n_ant_working = len(working_ants)

                # Create mapping: MS antenna index -> contiguous 0-based index
                # e.g., if working_ants = [0, 5, 10, 15], map: 0->0, 5->1, 10->2, 15->3
                ant_map = {int(ms_idx): i for i, ms_idx in enumerate(working_ants)}

                # Get antenna names
                ant_tab = table(f"{ms_path}::ANTENNA", readonly=True, ack=False)
                ant_names_all = ant_tab.getcol("NAME")
                ant_tab.close()
                # Convert to numpy array for proper indexing
                ant_names_all = np.array(ant_names_all)
                working_ants = np.array(working_ants, dtype=np.int32)
                ant_names = [ant_names_all[i] for i in working_ants]

                # Get ACTUAL integration time from data
                time_sample = sel_sample.getcol("TIME")
                unique_times_sample = np.unique(time_sample)
                if len(unique_times_sample) > 1:
                    time_diffs = np.diff(unique_times_sample)
                    actual_int_time = np.median(time_diffs[time_diffs > 0])
                else:
                    actual_int_time = 1.0

                # Get frequency and other metadata
                spw_tab = table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True, ack=False)
                freq = spw_tab.getcol("CHAN_FREQ")[spw_id_int]
                spw_tab.close()
                n_chan = len(freq)

                # Get feed basis
                pol_tab = table(f"{ms_path}::POLARIZATION", readonly=True, ack=False)
                corr_type = pol_tab.getcol("CORR_TYPE")[0]
                pol_tab.close()
                if corr_type[0] in [9, 10, 11, 12]:
                    feed_basis = FeedBasis.LINEAR
                elif corr_type[0] in [5, 6, 7, 8]:
                    feed_basis = FeedBasis.CIRCULAR
                else:
                    feed_basis = FeedBasis.LINEAR

                sel_sample.close()
                ms_temp.close()

                # STEP 2: Parse solint intervals and use ACTUAL integration time for chunking
                current_time_interval = time_interval[idx]
                current_freq_interval = freq_interval[idx]
                time_interval_sec = parse_time_interval(current_time_interval)

                # If user specified interval is smaller than actual integration, use actual
                if time_interval_sec < actual_int_time:
                    if verbose:
                        if RICH_AVAILABLE and console:
                            console.print(f"   [yellow]Warning:[/yellow] Requested time interval ({time_interval_sec:.1f}s) < actual integration time ({actual_int_time:.1f}s)")
                            console.print(f"   [yellow]Using actual integration time for chunking[/yellow]")
                    time_interval_sec = actual_int_time

                # STEP 3: Get full time array and create chunk boundaries
                time_arr_full = get_ms_time_array(ms_path, field=field, spw=spw_str, scans=scans)
                unique_times = np.unique(time_arr_full)

                # Group by actual unique timestamps for solution intervals
                # If solint >= actual int time, group multiple timestamps
                t_min, t_max = unique_times.min(), unique_times.max()
                chunk_assignments = ((unique_times - t_min) / time_interval_sec).astype(int)
                n_time_chunks = chunk_assignments.max() + 1

                # Create chunk boundaries from actual timestamps
                time_chunk_boundaries = []
                for chunk_id in range(n_time_chunks):
                    mask = chunk_assignments == chunk_id
                    if np.any(mask):
                        chunk_times = unique_times[mask]
                        time_chunk_boundaries.append((chunk_times.min(), chunk_times.max()))

                n_time_chunks = len(time_chunk_boundaries)
                n_baseline = n_ant_working * (n_ant_working - 1) // 2

                # STEP 5: Calculate batch plan based on memory constraints
                from ..core.resources import compute_batch_plan, get_available_ram
                total_time_sec = t_max - t_min
                available_ram_gb = get_available_ram()
                n_chunks_total, n_chunks_per_batch, plan_str = compute_batch_plan(
                    total_time_sec, time_interval_sec, n_baseline, n_chan,
                    available_ram_gb, safety_factor=0.3  # Conservative: use only 30% of available RAM
                )

                # STEP 6: Report plan (ONCE, professional)
                if verbose:
                    if RICH_AVAILABLE and console:
                        console.print(f"   [bold]Data:[/bold] {n_time_chunks} time chunks, {n_chan} chans, {n_ant_working} working ants, {n_baseline} baselines")
                        console.print(f"   [bold]Int time:[/bold] {actual_int_time:.2f}s")
                        console.print(f"   [bold]Memory:[/bold] {plan_str}")
                        console.print(f"   [bold]Feed:[/bold] {feed_basis.value}")
                    else:
                        print(f"  Data: {n_time_chunks} time chunks, {n_chan} chans, {n_ant_working} working ants")
                        print(f"  Int time: {actual_int_time:.2f}s")
                        print(f"  Memory: {plan_str}")
                        print(f"  Feed: {feed_basis.value}")

                # Build pre-apply Jones list
                pre_jones_list = []

                # From previous steps in this entry (same SPW)
                for prev_jt in jones_types[:idx]:
                    if prev_jt in solved_jones:
                        # Retrieve stored Jones and interpolate to current frequency grid
                        prev_jones_data = solved_jones[prev_jt]
                        prev_jones = prev_jones_data['jones']
                        prev_freq = prev_jones_data['freq']

                        # Interpolate to current freq grid if needed
                        prev_jones_interp = interpolate_jones_for_chaining(
                            prev_jones, prev_freq, freq, prev_jt, verbose=verbose
                        )
                        pre_jones_list.append(prev_jones_interp)

                # From external tables
                for i, pjt in enumerate(pre_apply):
                    if i < len(pre_tables):
                        tbl = pre_tables[i]
                        # Try to load with SPW suffix first, then without
                        pjt_with_spw = f"{pjt}_spw{spw_id}"
                        if Path(tbl).exists():
                            loaded_jones = None
                            loaded_freq = None

                            if pjt_with_spw in list_jones(tbl):
                                data = load_jones(tbl, pjt_with_spw)
                                loaded_jones = data['jones'][0]  # Remove time dimension
                                loaded_freq = data.get('freq', None)
                            elif pjt in list_jones(tbl):
                                data = load_jones(tbl, pjt)
                                loaded_jones = data['jones'][0]  # Remove time dimension
                                loaded_freq = data.get('freq', None)

                            if loaded_jones is not None:
                                # Interpolate to current freq grid
                                loaded_jones_interp = interpolate_jones_for_chaining(
                                    loaded_jones, loaded_freq, freq, pjt, verbose=verbose
                                )
                                pre_jones_list.append(loaded_jones_interp)

                # Show Jones chain information
                if pre_jones_list and verbose:
                    if RICH_AVAILABLE and console:
                        console.print(f"\n   [bold]Jones chain:[/bold]")
                        # Show each pre-applied Jones term
                        chain_idx = 0
                        for prev_jt in jones_types[:idx]:
                            if prev_jt in solved_jones:
                                prev_jones_data = solved_jones[prev_jt]
                                prev_freq = prev_jones_data['freq']

                                # Determine if interpolation happened
                                interpolated = False
                                if prev_freq is not None and len(freq) > 0:
                                    if len(prev_freq) != len(freq) or not np.allclose(prev_freq, freq):
                                        interpolated = True

                                # Show the term
                                if prev_freq is None:
                                    freq_info = "freq-averaged"
                                else:
                                    freq_info = f"{len(prev_freq)} channels"

                                if interpolated:
                                    console.print(f"      {chain_idx+1}. [cyan]{prev_jt}[/cyan] ({freq_info}) [yellow]→ interpolated to {len(freq)} channels[/yellow]")
                                else:
                                    console.print(f"      {chain_idx+1}. [cyan]{prev_jt}[/cyan] ({freq_info})")

                                chain_idx += 1

                        # Show external pre-apply
                        for i, pjt in enumerate(pre_apply):
                            if i < len(pre_tables):
                                console.print(f"      {chain_idx+1}. [cyan]{pjt}[/cyan] (external from {pre_tables[i]})")
                                chain_idx += 1

                        console.print(f"      → Solving [bold cyan]{jt}[/bold cyan]")
                    else:
                        print(f"  Pre-applying: {len(pre_jones_list)} Jones terms")

                # Parse freq interval (time interval already parsed above)
                from ..core.solint import create_freq_chunks
                chan_width = np.median(np.diff(freq)) if len(freq) > 1 else freq[0] * 0.01
                freq_interval_hz = parse_freq_interval(current_freq_interval, chan_width)

                # Create frequency chunks (determine based on Jones type)
                # Note: For K term, we keep full freq axis for delay estimation
                if jt.upper() != 'K':
                    # For B/G/D/Xf: Create freq chunks
                    freq_chunks = create_freq_chunks(freq, freq_interval_hz)
                    n_freq_chunks = len(freq_chunks)
                else:
                    # K term: single freq chunk (all channels, needed for delay)
                    freq_chunks = [np.arange(len(freq))]
                    n_freq_chunks = 1

                total_blocks = n_time_chunks * n_freq_chunks

                if verbose:
                    if RICH_AVAILABLE and console:
                        console.print(f"   [bold]Solint:[/bold] [cyan]{current_time_interval}[/cyan] (time), [cyan]{current_freq_interval}[/cyan] (freq)")
                        console.print(f"   [bold]Solving:[/bold] [cyan]{n_time_chunks}[/cyan] time × [cyan]{n_freq_chunks}[/cyan] freq = [bold cyan]{total_blocks}[/bold cyan] solution blocks")
                    else:
                        print(f"  Solint: {current_time_interval} (time), {current_freq_interval} (freq)")
                        print(f"  Solving: {n_time_chunks} time × {n_freq_chunks} freq = {total_blocks} solution blocks")

                # STEP 7: Batched loading and solving
                jones_solutions = []
                weights_solutions = []
                block_idx = 0

                # Track convergence info for all blocks
                all_block_info = []
                all_costs_init = []
                all_costs_final = []
                all_nfev = []

                # Track number of workers used (for timing output)
                n_workers_auto = 1  # Default to serial

                # Start timer for this Jones solve
                import time
                solve_start_time = time.time()

                # Progress tracking
                if verbose and total_blocks > 1:
                    if RICH_AVAILABLE and console:
                        progress = Progress(
                            TextColumn("[bold blue]{task.description}"),
                            BarColumn(bar_width=40),
                            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                            TextColumn("({task.completed}/{task.total})"),
                            console=console
                        )
                        task = progress.add_task("  Progress:", total=total_blocks)
                        progress.start()

                # Loop over batches of time chunks
                n_batches = int(np.ceil(n_time_chunks / n_chunks_per_batch))

                for batch_idx in range(n_batches):
                    batch_start_idx = batch_idx * n_chunks_per_batch
                    batch_end_idx = min(batch_start_idx + n_chunks_per_batch, n_time_chunks)

                    # Determine time range for this batch (union of all chunks in batch)
                    t_min_batch = time_chunk_boundaries[batch_start_idx][0]
                    t_max_batch = time_chunk_boundaries[batch_end_idx - 1][1]

                    # Load MS data for this batch ONLY
                    ms_data = read_ms(
                        ms_path, field=field, spw=spw_str, scans=scans,
                        time_min=t_min_batch, time_max=t_max_batch
                    )

                    vis_obs_batch = ms_data['vis_obs']
                    vis_model_batch = ms_data['vis_model']
                    antenna1_batch = ms_data['antenna1']
                    antenna2_batch = ms_data['antenna2']
                    flags_batch = ms_data['flags']
                    time_arr_batch = ms_data['time']

                    # Remap antenna indices from MS indices to contiguous 0-based indices
                    # This is CRITICAL: solver expects antenna indices 0, 1, 2, ..., n_ant_working-1
                    # but MS might have gaps like 0, 5, 10, 15, ...
                    antenna1_batch_remapped = np.array([ant_map[int(a)] for a in antenna1_batch], dtype=np.int32)
                    antenna2_batch_remapped = np.array([ant_map[int(a)] for a in antenna2_batch], dtype=np.int32)

                    # Set global batch data for workers (shared via fork, no copy!)
                    global _batch_data
                    _batch_data = {
                        'vis_obs': vis_obs_batch,
                        'vis_model': vis_model_batch,
                        'flags': flags_batch,
                        'antenna1_remapped': antenna1_batch_remapped,
                        'antenna2_remapped': antenna2_batch_remapped,
                        'time': time_arr_batch,
                        'freq': freq,
                        'pre_jones_list': pre_jones_list,
                    }

                    # Prepare lightweight chunk tasks (no large arrays!)
                    chunk_tasks = []
                    for t_idx in range(batch_start_idx, batch_end_idx):
                        t_start_chunk, t_end_chunk = time_chunk_boundaries[t_idx]
                        for f_idx, freq_chunk_indices in enumerate(freq_chunks):
                            chunk_tasks.append((
                                t_idx, f_idx, t_start_chunk, t_end_chunk, freq_chunk_indices,
                                jt, n_ant_working, ref_ant, phase_only[idx],
                                rfi_sigma, max_iter, tol
                            ))

                    # Solve chunks in parallel using multiprocessing
                    import multiprocessing as mp
                    import os
                    import time

                    # Determine number of workers
                    if n_workers is None:
                        # Auto-detect: default to 8, but don't exceed available cores or tasks
                        n_workers_auto = min(os.cpu_count() or 1, len(chunk_tasks), 8)
                    else:
                        # User specified: respect their choice (no artificial cap)
                        n_workers_auto = min(n_workers, len(chunk_tasks))

                    # Start timer for this batch
                    batch_start_time = time.time()

                    if n_workers_auto > 1 and len(chunk_tasks) > 1:
                        # Parallel solving
                        with mp.Pool(processes=n_workers_auto) as pool:
                            chunk_results = pool.map(_solve_chunk_worker, chunk_tasks)
                    else:
                        # Serial fallback (single chunk or single core)
                        chunk_results = [_solve_chunk_worker(task) for task in chunk_tasks]

                    # Process parallel results
                    # Organize results by (t_idx, f_idx)
                    result_dict = {}
                    weight_dict = {}
                    for t_idx_r, f_idx_r, jones_chunk, params_chunk, info_chunk, weight in chunk_results:
                        if t_idx_r not in result_dict:
                            result_dict[t_idx_r] = {}
                            weight_dict[t_idx_r] = {}
                        result_dict[t_idx_r][f_idx_r] = (jones_chunk, params_chunk, info_chunk)
                        weight_dict[t_idx_r][f_idx_r] = weight

                        # Store convergence info
                        all_block_info.append({
                            't_idx': t_idx_r,
                            'f_idx': f_idx_r,
                            'info': info_chunk,
                        })
                        all_costs_init.append(info_chunk.get('cost_init', 0.0))
                        all_costs_final.append(info_chunk.get('cost_final', 0.0))
                        all_nfev.append(info_chunk.get('nfev', 0))

                        # Update progress
                        block_idx += 1
                        if verbose and total_blocks > 1 and RICH_AVAILABLE and console:
                            progress.update(task, completed=block_idx)

                    # Build jones_solutions and weights_solutions structure in correct order
                    for t_idx in range(batch_start_idx, batch_end_idx):
                        jones_freq_solutions = []
                        weights_freq_solutions = []
                        for f_idx in range(n_freq_chunks):
                            if t_idx in result_dict and f_idx in result_dict[t_idx]:
                                jones_chunk, params_chunk, info_chunk = result_dict[t_idx][f_idx]
                                jones_freq_solutions.append(jones_chunk)
                                weights_freq_solutions.append(weight_dict[t_idx][f_idx])
                            else:
                                # Empty chunk (no data)
                                jones_empty = np.zeros((n_ant_working, 2, 2), dtype=np.complex128)
                                jones_freq_solutions.append(jones_empty)
                                weights_freq_solutions.append(0.0)
                        jones_solutions.append(jones_freq_solutions)
                        weights_solutions.append(weights_freq_solutions)

                    # Free batch memory before loading next batch
                    cleanup_arrays(vis_obs_batch, vis_model_batch, flags_batch,
                                   antenna1_batch, antenna2_batch, time_arr_batch,
                                   antenna1_batch_remapped, antenna2_batch_remapped)
                    del ms_data

                    # Clear global batch data
                    _batch_data.clear()

                    # Force garbage collection to free memory immediately
                    import gc
                    gc.collect()

                # Stop progress bar
                if verbose and total_blocks > 1 and RICH_AVAILABLE and console:
                    progress.stop()

                # Reshape solutions to (n_time, n_freq, n_ant, 2, 2) or appropriate shape
                # Also reshape weights to (n_time, n_freq)
                if n_time_chunks == 1 and n_freq_chunks == 1:
                    # Single solution
                    jones = jones_solutions[0][0]
                    weights = np.array([[weights_solutions[0][0]]], dtype=np.float64)
                    params = params_chunk
                    info = info_chunk
                elif n_time_chunks > 1 and n_freq_chunks == 1:
                    # Multiple time solutions, single freq
                    jones = np.array([sol[0] for sol in jones_solutions])
                    weights = np.array([[w[0]] for w in weights_solutions], dtype=np.float64)
                    params = params_chunk
                    info = info_chunk
                elif n_time_chunks == 1 and n_freq_chunks > 1:
                    # Single time, multiple freq solutions
                    jones = np.array(jones_solutions[0])
                    weights = np.array([weights_solutions[0]], dtype=np.float64)
                    params = params_chunk
                    info = info_chunk
                else:
                    # Multiple time and freq solutions
                    jones = np.array([[sol for sol in time_sol] for time_sol in jones_solutions])
                    weights = np.array(weights_solutions, dtype=np.float64)
                    params = params_chunk
                    info = info_chunk

                # Free jones_solutions and weights_solutions immediately after reshaping to save memory
                del jones_solutions, weights_solutions
                import gc
                gc.collect()

                # Comprehensive statistics output
                if verbose:
                    if RICH_AVAILABLE and console:
                        console.print(f"\n")

                        # Fit quality (global)
                        if len(all_costs_init) > 0:
                            total_cost_init = sum(all_costs_init)
                            total_cost_final = sum(all_costs_final)
                            cost_reduction = total_cost_init / max(total_cost_final, 1e-20)
                            console.print(f"   [bold]Fit quality (global):[/bold]")
                            console.print(f"      Cost reduction: {cost_reduction:.2f}×")
                            console.print(f"      Final cost: {total_cost_final:.6e}")

                        # Gain statistics (all antennas)
                        console.print(f"\n   [bold]Gain statistics (all antennas):[/bold]")
                        console.print(f"      Solution shape: [cyan]{jones.shape}[/cyan]")

                        # Extract gains for statistics
                        if jones.ndim == 3:
                            # (n_ant, 2, 2) - single solution
                            g_xx = jones[:, 0, 0]
                            g_yy = jones[:, 1, 1]
                        elif jones.ndim == 4:
                            # (n_time, n_ant, 2, 2) or (n_freq, n_ant, 2, 2)
                            g_xx = jones[:, :, 0, 0].flatten()
                            g_yy = jones[:, :, 1, 1].flatten()
                        elif jones.ndim == 5:
                            # (n_time, n_freq, n_ant, 2, 2)
                            g_xx = jones[:, :, :, 0, 0].flatten()
                            g_yy = jones[:, :, :, 1, 1].flatten()
                        else:
                            g_xx = jones.flatten()
                            g_yy = jones.flatten()

                        # Filter NaN
                        g_xx_valid = g_xx[~np.isnan(g_xx)]
                        g_yy_valid = g_yy[~np.isnan(g_yy)]

                        if len(g_xx_valid) > 0:
                            console.print(f"      Median |g| = {np.median(np.abs(np.concatenate([g_xx_valid, g_yy_valid]))):.3f}")
                            console.print(f"      Median phase scatter = {np.std(np.angle(np.concatenate([g_xx_valid, g_yy_valid])))*180/np.pi:.1f}°")
                            console.print(f"      Max |g| = {np.max(np.abs(np.concatenate([g_xx_valid, g_yy_valid]))):.3f}")

                        # Reference antenna gains
                        console.print(f"\n   [bold]Reference antenna gains (ant {ref_ant}):[/bold]")
                        if jones.ndim == 3:
                            # Single solution
                            ref_g_xx = jones[ref_ant, 0, 0]
                            ref_g_yy = jones[ref_ant, 1, 1]
                            console.print(f"      Polarization X: |g_X| = {np.abs(ref_g_xx):.4f}, arg(g_X) = {np.angle(ref_g_xx)*180/np.pi:+.2f}°")
                            console.print(f"      Polarization Y: |g_Y| = {np.abs(ref_g_yy):.4f}, arg(g_Y) = {np.angle(ref_g_yy)*180/np.pi:+.2f}°")
                        elif jones.ndim == 4:
                            # Time or freq dependent
                            ref_g_xx = jones[:, ref_ant, 0, 0]
                            ref_g_yy = jones[:, ref_ant, 1, 1]
                            console.print(f"      Polarization X:")
                            console.print(f"         ⟨|g_X|⟩ = {np.nanmean(np.abs(ref_g_xx)):.4f} ± {np.nanstd(np.abs(ref_g_xx)):.4f}")
                            console.print(f"         ⟨arg(g_X)⟩ = {np.nanmean(np.angle(ref_g_xx))*180/np.pi:+.2f}° ± {np.nanstd(np.angle(ref_g_xx))*180/np.pi:.2f}°")
                            console.print(f"      Polarization Y:")
                            console.print(f"         ⟨|g_Y|⟩ = {np.nanmean(np.abs(ref_g_yy)):.4f} ± {np.nanstd(np.abs(ref_g_yy)):.4f}")
                            console.print(f"         ⟨arg(g_Y)⟩ = {np.nanmean(np.angle(ref_g_yy))*180/np.pi:+.2f}° ± {np.nanstd(np.angle(ref_g_yy))*180/np.pi:.2f}°")
                        elif jones.ndim == 5:
                            # Time and freq dependent
                            ref_g_xx = jones[:, :, ref_ant, 0, 0]
                            ref_g_yy = jones[:, :, ref_ant, 1, 1]
                            console.print(f"      Polarization X:")
                            console.print(f"         ⟨|g_X|⟩ = {np.nanmean(np.abs(ref_g_xx)):.4f} ± {np.nanstd(np.abs(ref_g_xx)):.4f}")
                            console.print(f"         ⟨arg(g_X)⟩ = {np.nanmean(np.angle(ref_g_xx))*180/np.pi:+.2f}° ± {np.nanstd(np.angle(ref_g_xx))*180/np.pi:.2f}°")
                            console.print(f"      Polarization Y:")
                            console.print(f"         ⟨|g_Y|⟩ = {np.nanmean(np.abs(ref_g_yy)):.4f} ± {np.nanstd(np.abs(ref_g_yy)):.4f}")
                            console.print(f"         ⟨arg(g_Y)⟩ = {np.nanmean(np.angle(ref_g_yy))*180/np.pi:+.2f}° ± {np.nanstd(np.angle(ref_g_yy))*180/np.pi:.2f}°")

                        # Convergence (global)
                        console.print(f"\n   [bold]Convergence (global):[/bold]")
                        if len(all_nfev) > 0:
                            console.print(f"      Iterations: {int(np.mean(all_nfev))}")
                            console.print(f"      Tolerance: {tol:.1e}")

                        # Worst block convergence (if multi-block)
                        if total_blocks > 1 and len(all_block_info) > 0:
                            # Find worst block (highest final cost)
                            worst_idx = np.argmax(all_costs_final)
                            worst_block = all_block_info[worst_idx]
                            console.print(f"\n   [bold]Worst block convergence:[/bold]")
                            console.print(f"      Chunk (t={worst_block['t_idx']+1}/{n_time_chunks}, f={worst_block['f_idx']+1}/{n_freq_chunks}):")
                            console.print(f"         Iterations: {worst_block['info'].get('nfev', 0)}")
                            console.print(f"         Final cost: {worst_block['info'].get('cost_final', 0):.6e}")
                            console.print(f"         Cost reduction: {worst_block['info'].get('cost_init', 1) / max(worst_block['info'].get('cost_final', 1), 1e-20):.2f}×")

                        # Timing
                        solve_time = time.time() - solve_start_time
                        console.print(f"\n   [bold]Timing:[/bold]")
                        if solve_time < 60:
                            console.print(f"      Total time: {solve_time:.1f}s")
                        else:
                            console.print(f"      Total time: {solve_time/60:.1f} min ({solve_time:.1f}s)")
                        if n_workers_auto > 1:
                            console.print(f"      Workers: {n_workers_auto} (parallel)")
                        else:
                            console.print(f"      Workers: 1 (serial)")

                    else:
                        print(f"  Jones solution shape: {jones.shape}")

                # Save with SPW suffix if multiple SPWs
                jt_save = f"{jt}_spw{spw_id}" if n_spws > 1 else jt

                # Create time grid for saved solutions (use chunk midpoints)
                time_grid = np.array([(t_start + t_end) / 2.0 for t_start, t_end in time_chunk_boundaries])

                # Create frequency grid for saved solutions
                if n_freq_chunks == 1:
                    # Single freq chunk: use full freq array or None
                    if jt.upper() == 'K':
                        freq_grid = freq
                    else:
                        freq_grid = None
                else:
                    # Multiple freq chunks: use center freq of each chunk
                    freq_grid = np.array([np.mean(freq[chunk]) for chunk in freq_chunks])

                quality_obj = info.get('quality', None)

                # Apply fluxscale if requested for G mode
                flux_results = None
                if jt.upper() == 'G' and 'fluxscale' in cfg:
                    fluxscale_cfg = cfg['fluxscale']
                    if fluxscale_cfg.get('enable', False):
                        if verbose:
                            if RICH_AVAILABLE and console:
                                console.print(f"\n   [bold]Applying flux scaling[/bold]")
                            else:
                                print(f"\n  Applying flux scaling")

                        # Get reference gains
                        ref_fields = fluxscale_cfg.get('reference_field', [])
                        if isinstance(ref_fields, str):
                            ref_fields = [ref_fields]

                        # Check if external reference table provided
                        ref_table = fluxscale_cfg.get('reference_table', None)

                        if ref_table:
                            # Load from external file
                            ref_data = load_jones(ref_table, 'G')
                            jones_ref = ref_data['jones']
                            freq_ref = ref_data.get('freq', None)

                            if verbose:
                                if RICH_AVAILABLE and console:
                                    console.print(f"      Reference: [cyan]{ref_table}[/cyan]")
                                else:
                                    print(f"    Reference: {ref_table}")
                        else:
                            # On-the-fly: assume all fields solved in current MS
                            # This would require multi-field solving support
                            # For now, raise error
                            raise ValueError("On-the-fly fluxscale (multi-field in same solve) not yet implemented. "
                                           "Please provide reference_table.")

                        # Apply fluxscale
                        jones_scaled, flux_results = compute_fluxscale(
                            jones_ref,
                            jones,
                            freq=freq_grid,
                            verbose=verbose
                        )

                        # Replace jones with scaled version
                        jones = jones_scaled

                        if verbose:
                            if RICH_AVAILABLE and console:
                                console.print(f"      [green]Flux scaling applied successfully[/green]")
                            else:
                                print(f"    Flux scaling applied successfully")

                save_jones(
                    output_table, jt_save, jones,
                    time=time_grid,
                    freq=freq_grid,
                    antenna=working_ants,
                    weights=weights,
                    params=params,
                    metadata={
                        'ref_antenna': ref_ant,
                        'field': field,
                        'ms': ms_path,
                        'spw_id': spw_id,
                        'phase_only': phase_only[idx],
                        'time_interval': current_time_interval,
                        'freq_interval': current_freq_interval,
                        'actual_int_time': actual_int_time,
                        'n_time_chunks': n_time_chunks,
                        'n_freq_chunks': n_freq_chunks,
                        'antenna_names': ant_names,
                        'n_ant_total': len(ant_names_all),
                        'n_ant_working': n_ant_working,
                        'cost_init': info.get('cost_init'),
                        'cost_final': info.get('cost_final'),
                        'fluxscale': flux_results if flux_results else None,
                    },
                    quality=quality_obj,
                    overwrite=True,
                )

                if verbose:
                    if RICH_AVAILABLE and console:
                        console.print(f"   [green]Saved to {output_table}[/green] as [cyan]{jt_save}[/cyan]")
                    else:
                        print(f"  Saved to {output_table} as {jt_save}")

                # Store for chaining (within this SPW)
                # For chaining, use the first time chunk's solution
                # Freq dimension depends on whether we have freq chunks
                if n_time_chunks == 1 and n_freq_chunks == 1:
                    # Single solution: (n_ant, 2, 2) or (n_ant, n_freq, 2, 2) for K
                    if jt.upper() == 'K' and jones.ndim == 4:
                        # K with freq axis: (n_ant, n_freq, 2, 2)
                        solved_jones[jt] = {'jones': jones, 'freq': freq}
                    else:
                        # Freq-averaged: (n_ant, 2, 2)
                        solved_jones[jt] = {'jones': jones, 'freq': None}
                elif n_time_chunks > 1 and n_freq_chunks == 1:
                    # Multiple time: (n_time, n_ant, 2, 2) or (n_time, n_ant, n_freq, 2, 2) for K
                    # Use first time for chaining
                    if jt.upper() == 'K' and jones[0].ndim == 4:
                        solved_jones[jt] = {'jones': jones[0], 'freq': freq}
                    else:
                        solved_jones[jt] = {'jones': jones[0], 'freq': None}
                elif n_time_chunks == 1 and n_freq_chunks > 1:
                    # Multiple freq: (n_freq, n_ant, 2, 2)
                    # Transpose to (n_ant, n_freq, 2, 2) for chaining
                    jones_for_chain = np.transpose(jones, (1, 0, 2, 3))
                    solved_jones[jt] = {'jones': jones_for_chain, 'freq': freq_grid}
                else:
                    # Multiple time and freq: (n_time, n_freq, n_ant, 2, 2)
                    # Use first time, transpose freq
                    jones_for_chain = np.transpose(jones[0], (1, 0, 2, 3))
                    solved_jones[jt] = {'jones': jones_for_chain, 'freq': freq_grid}

    # Process apply entries
    for apply_entry in cfg.get('apply', []):
        # Get parameters
        ms_path = apply_entry.get('ms')
        jones_types = apply_entry.get('jones', [])
        tables = apply_entry.get('tables', [])
        output_col = apply_entry.get('output_col', 'CORRECTED_DATA')
        spw = apply_entry.get('spw', None)
        field = apply_entry.get('field', None)

        # Handle single values or lists
        if isinstance(jones_types, str):
            jones_types = [jones_types]
        if isinstance(tables, str):
            tables = [tables]

        if not jones_types:
            continue

        # Expand tables if single table for multiple Jones types
        if len(tables) == 1 and len(jones_types) > 1:
            tables = tables * len(jones_types)

        if len(tables) != len(jones_types):
            raise ValueError(f"Number of tables ({len(tables)}) must match number of Jones types ({len(jones_types)}) or be 1")

        # Apply calibration
        write_corrected(
            ms_path,
            [],
            tables,
            jones_types,
            output_col,
            spw=spw,
            field=field,
            verbose=verbose
        )

    if verbose:
        if RICH_AVAILABLE and console:
            console.print(Panel.fit(
                "[bold green]Pipeline complete![/bold green]",
                border_style="green"
            ))
        else:
            print("\n" + "="*60)
            print("Pipeline complete")
            print("="*60 + "\n")


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_solve(
    jones_type: str,
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    n_ant: int,
    freq: np.ndarray = None,
    flags: np.ndarray = None,
    ref_ant: int = 0,
    phase_only: bool = False,
    output_file: str = None,
    verbose: bool = True,
):
    """Quick solve for a single Jones type."""
    jones, params, info = solve(
        jones_type, vis_obs, vis_model, antenna1, antenna2, n_ant,
        freq=freq, flags=flags, ref_ant=ref_ant, phase_only=phase_only,
        verbose=verbose
    )
    
    if output_file:
        time = np.array([0.0])
        freq_out = freq if freq is not None else np.array([1e9])
        
        save_jones(
            output_file, jones_type, jones, time, freq_out,
            params=params,
            metadata={
                'ref_antenna': ref_ant,
                'phase_only': phase_only,
                'cost_init': info.get('cost_init'),
                'cost_final': info.get('cost_final'),
            },
            overwrite=True
        )
        
        if verbose:
            print(f"Saved to {output_file}")
    
    return jones, params, info


__all__ = ['parse_config', 'run_pipeline', 'quick_solve', 'read_ms', 'write_corrected']
