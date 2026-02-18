"""ALAKAZAM MS I/O utilities.

Read metadata, data, flags and model from Measurement Sets.
All heavy reading is done via casacore.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("alakazam")


@dataclass
class MSMetadata:
    ms_path: str
    n_ant: int
    ant_names: List[str]
    n_spw: int
    spw_freqs: List[np.ndarray]   # per SPW: (n_chan,) Hz
    n_corr: int
    corr_types: List[int]
    field_names: List[str]
    field_ids: List[int]
    field_ra: List[float]         # radians
    field_dec: List[float]        # radians
    n_baseline: int
    ant1: np.ndarray              # (n_baseline,) int
    ant2: np.ndarray
    times: np.ndarray             # all unique timestamps MJD s
    scans: np.ndarray             # per row: scan number


def detect_metadata(ms_path: str) -> MSMetadata:
    """Extract all metadata needed for calibration from an MS."""
    from casacore.tables import table

    ant_tab = table(f"{ms_path}::ANTENNA", readonly=True, ack=False)
    ant_names = list(ant_tab.getcol("NAME"))
    n_ant = len(ant_names)
    ant_tab.close()

    spw_tab = table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True, ack=False)
    spw_freqs = [spw_tab.getcol("CHAN_FREQ")[i] for i in range(spw_tab.nrows())]
    n_spw = len(spw_freqs)
    spw_tab.close()

    pol_tab = table(f"{ms_path}::POLARIZATION", readonly=True, ack=False)
    corr_types = list(pol_tab.getcol("CORR_TYPE")[0])
    n_corr = len(corr_types)
    pol_tab.close()

    field_tab = table(f"{ms_path}::FIELD", readonly=True, ack=False)
    field_names = list(field_tab.getcol("NAME"))
    field_ids   = list(range(len(field_names)))
    phase_dirs  = field_tab.getcol("PHASE_DIR")  # (n_field, 1, 2)
    field_ra    = [float(phase_dirs[i, 0, 0]) for i in range(len(field_names))]
    field_dec   = [float(phase_dirs[i, 0, 1]) for i in range(len(field_names))]
    field_tab.close()

    ms = table(ms_path, readonly=True, ack=False)
    ant1_col = ms.getcol("ANTENNA1")
    ant2_col = ms.getcol("ANTENNA2")
    times_col = ms.getcol("TIME")
    scans_col = ms.getcol("SCAN_NUMBER")
    ms.close()

    # Unique baselines (no autocorrelations)
    mask = ant1_col != ant2_col
    unique_bl = np.unique(np.stack([ant1_col[mask], ant2_col[mask]], axis=1), axis=0)
    n_baseline = len(unique_bl)

    unique_times = np.unique(times_col)

    return MSMetadata(
        ms_path=ms_path,
        n_ant=n_ant,
        ant_names=ant_names,
        n_spw=n_spw,
        spw_freqs=spw_freqs,
        n_corr=n_corr,
        corr_types=corr_types,
        field_names=field_names,
        field_ids=field_ids,
        field_ra=field_ra,
        field_dec=field_dec,
        n_baseline=n_baseline,
        ant1=unique_bl[:, 0].astype(np.int32),
        ant2=unique_bl[:, 1].astype(np.int32),
        times=unique_times,
        scans=scans_col,
    )


def read_data(
    ms_path: str,
    spw: int,
    fields: Optional[List[str]] = None,
    scans: Optional[List[int]] = None,
    data_col: str = "DATA",
    model_col: str = "MODEL_DATA",
    include_flags: bool = True,
) -> Dict[str, Any]:
    """Read visibility data from MS for given SPW / field / scan selection.

    Returns dict with:
      vis_obs:   (n_row, n_chan, 4) complex128
      vis_model: (n_row, n_chan, 4) complex128
      flags:     (n_row, n_chan, 4) bool
      ant1:      (n_row,) int32
      ant2:      (n_row,) int32
      times:     (n_row,) float64 MJD s
      scans:     (n_row,) int32
      field_ids: (n_row,) int32
      freqs:     (n_chan,) float64 Hz
    """
    from casacore.tables import table, taql

    ms = table(ms_path, readonly=True, ack=False)

    # Build TaQL selection
    conditions = [f"DATA_DESC_ID == {spw}", "ANTENNA1 != ANTENNA2"]

    if fields is not None:
        field_tab = table(f"{ms_path}::FIELD", readonly=True, ack=False)
        all_names = list(field_tab.getcol("NAME"))
        field_tab.close()
        ids = [i for i, n in enumerate(all_names) if n in fields]
        if ids:
            id_str = ",".join(str(i) for i in ids)
            conditions.append(f"FIELD_ID IN [{id_str}]")

    if scans is not None:
        scan_str = ",".join(str(s) for s in scans)
        conditions.append(f"SCAN_NUMBER IN [{scan_str}]")

    query = " AND ".join(conditions)
    sub = taql(f"SELECT * FROM $ms WHERE {query}")

    if sub.nrows() == 0:
        ms.close()
        sub.close()
        return {}

    vis_obs   = sub.getcol(data_col)   # (n_row, n_chan, n_corr)
    vis_model = sub.getcol(model_col) if model_col in sub.colnames() else np.ones_like(vis_obs)
    flags_raw = sub.getcol("FLAG") if include_flags else np.zeros(vis_obs.shape, dtype=bool)
    ant1_col  = sub.getcol("ANTENNA1").astype(np.int32)
    ant2_col  = sub.getcol("ANTENNA2").astype(np.int32)
    times_col = sub.getcol("TIME")
    scans_col = sub.getcol("SCAN_NUMBER").astype(np.int32)
    fids_col  = sub.getcol("FIELD_ID").astype(np.int32)

    sub.close()
    ms.close()

    # Frequencies for this SPW
    spw_tab = table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True, ack=False)
    freqs = spw_tab.getcol("CHAN_FREQ")[spw]
    spw_tab.close()

    # Convert (n_row, n_chan, 4) corr → (n_row, n_chan, 2, 2)
    def _to_2x2(v):
        out = np.zeros(v.shape[:2] + (2, 2), dtype=np.complex128)
        out[..., 0, 0] = v[..., 0]
        out[..., 0, 1] = v[..., 1] if v.shape[-1] > 2 else 0
        out[..., 1, 0] = v[..., 2] if v.shape[-1] > 2 else 0
        out[..., 1, 1] = v[..., -1]
        return out

    flags_2x2 = np.zeros(vis_obs.shape[:2] + (2, 2), dtype=bool)
    flags_2x2[..., 0, 0] = flags_raw[..., 0]
    flags_2x2[..., 0, 1] = flags_raw[..., 1] if flags_raw.shape[-1] > 2 else False
    flags_2x2[..., 1, 0] = flags_raw[..., 2] if flags_raw.shape[-1] > 2 else False
    flags_2x2[..., 1, 1] = flags_raw[..., -1]

    return {
        "vis_obs":   _to_2x2(vis_obs),
        "vis_model": _to_2x2(vis_model),
        "flags":     flags_2x2,
        "ant1":      ant1_col,
        "ant2":      ant2_col,
        "times":     times_col,
        "scans":     scans_col,
        "field_ids": fids_col,
        "freqs":     freqs,
    }


def write_corrected(
    ms_path: str,
    spw: int,
    row_indices: np.ndarray,
    corrected: np.ndarray,      # (n_row, n_chan, 2, 2) complex128
    output_col: str = "CORRECTED_DATA",
) -> None:
    """Write corrected visibilities back to MS."""
    from casacore.tables import table

    # Convert (n_row, n_chan, 2, 2) → (n_row, n_chan, n_corr)
    n_row, n_chan = corrected.shape[:2]
    if output_col == "CORRECTED_DATA":
        ms = table(ms_path, readonly=False, ack=False)
        if output_col not in ms.colnames():
            from casacore.tables import makearrcoldesc, maketabdesc
            col = ms.getcol("DATA")
            ms.close()
            ms = table(ms_path, readonly=False, ack=False)

        data_flat = np.zeros((n_row, n_chan, 4), dtype=np.complex128)
        data_flat[..., 0]  = corrected[..., 0, 0]
        data_flat[..., 1]  = corrected[..., 0, 1]
        data_flat[..., 2]  = corrected[..., 1, 0]
        data_flat[..., 3]  = corrected[..., 1, 1]

        ms.putcol(output_col, data_flat, row_indices[0], len(row_indices))
        ms.close()


def compute_solint_grid(
    times: np.ndarray,
    solint_s: float,
) -> List[np.ndarray]:
    """Partition unique timestamps into solint blocks.

    Returns list of arrays, each containing timestamps in one solint block.
    """
    if not np.isfinite(solint_s):
        return [times]

    blocks = []
    t_start = times[0]
    block = [times[0]]
    for t in times[1:]:
        if t - t_start >= solint_s:
            blocks.append(np.array(block))
            block = [t]
            t_start = t
        else:
            block.append(t)
    if block:
        blocks.append(np.array(block))
    return blocks


def parse_spw_selection(spw_str: Optional[str], n_spw: int) -> List[int]:
    """Parse SPW string to list of SPW indices."""
    if spw_str is None or spw_str.strip() in ("", "*"):
        return list(range(n_spw))
    result = []
    for part in spw_str.split(","):
        part = part.strip()
        if "~" in part:
            lo, hi = part.split("~")
            result.extend(range(int(lo), int(hi) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))
