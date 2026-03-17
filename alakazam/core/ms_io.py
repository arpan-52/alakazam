"""ALAKAZAM v1 MS I/O.

Reads raw data in native (n_row, n_chan, n_corr) format — no 2x2 conversion.
2x2 conversion happens later on tiny averaged data only.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations
import contextlib
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("alakazam")


@contextlib.contextmanager
def suppress_stderr():
    """Redirect stderr to /dev/null — kills casacore C++ ZENITH warnings."""
    try:
        stderr_fd = sys.stderr.fileno()
        saved_fd = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        try:
            yield
        finally:
            os.dup2(saved_fd, stderr_fd)
            os.close(saved_fd)
    except (AttributeError, OSError):
        yield  # fallback if no fileno (e.g. notebook)


# -------------------------------------------------------------------
# Metadata
# -------------------------------------------------------------------

@dataclass
class MSMetadata:
    ms_path: str
    n_ant: int
    ant_names: List[str]
    n_spw: int
    spw_freqs: List[np.ndarray]
    n_corr: int
    corr_types: List[int]
    field_names: List[str]
    field_ids: List[int]
    field_ra: List[float]
    field_dec: List[float]
    n_baseline: int
    ant1: np.ndarray
    ant2: np.ndarray
    times: np.ndarray
    scans: np.ndarray


def detect_metadata(ms_path: str) -> MSMetadata:
    from casacore.tables import table
    with suppress_stderr():
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
    field_ids = list(range(len(field_names)))
    phase_dirs = field_tab.getcol("PHASE_DIR")
    field_ra = [float(phase_dirs[i, 0, 0]) for i in range(len(field_names))]
    field_dec = [float(phase_dirs[i, 0, 1]) for i in range(len(field_names))]
    field_tab.close()

    ms = table(ms_path, readonly=True, ack=False)
    ant1_col = ms.getcol("ANTENNA1")
    ant2_col = ms.getcol("ANTENNA2")
    times_col = ms.getcol("TIME")
    scans_col = ms.getcol("SCAN_NUMBER")
    ms.close()

    mask = ant1_col != ant2_col
    unique_bl = np.unique(
        np.stack([ant1_col[mask], ant2_col[mask]], axis=1), axis=0)

    return MSMetadata(
        ms_path=ms_path, n_ant=n_ant, ant_names=ant_names,
        n_spw=n_spw, spw_freqs=spw_freqs,
        n_corr=n_corr, corr_types=corr_types,
        field_names=field_names, field_ids=field_ids,
        field_ra=field_ra, field_dec=field_dec,
        n_baseline=len(unique_bl),
        ant1=unique_bl[:, 0].astype(np.int32),
        ant2=unique_bl[:, 1].astype(np.int32),
        times=np.unique(times_col), scans=scans_col,
    )


def validate_selections(meta, field_names, scans=None, spw_ids=None):
    for f in field_names:
        if f and f not in meta.field_names:
            raise ValueError(f"Field '{f}' not found. Available: {meta.field_names}")
    if scans is not None:
        ms_scans = set(meta.scans)
        for s in scans:
            if s not in ms_scans:
                logger.warning(f"Scan {s} not in MS")
    if spw_ids is not None:
        for s in spw_ids:
            if s < 0 or s >= meta.n_spw:
                raise ValueError(f"SPW {s} out of range [0, {meta.n_spw})")


# -------------------------------------------------------------------
# Lightweight query — timestamps + scans only, no DATA
# -------------------------------------------------------------------

def query_times_scans(ms_path, spw, fields=None, scans=None):
    """Get TIME, SCAN_NUMBER, ANTENNA1, ANTENNA2 without loading DATA."""
    from casacore.tables import table, taql
    with suppress_stderr():
        ms = table(ms_path, readonly=True, ack=False)
    conds = [f"DATA_DESC_ID == {spw}", "ANTENNA1 != ANTENNA2"]
    if fields:
        ft = table(f"{ms_path}::FIELD", readonly=True, ack=False)
        names = list(ft.getcol("NAME")); ft.close()
        ids = [i for i, n in enumerate(names) if n in fields]
        if ids: conds.append(f"FIELD_ID IN [{','.join(str(i) for i in ids)}]")
    if scans:
        conds.append(f"SCAN_NUMBER IN [{','.join(str(s) for s in scans)}]")
    sub = taql(f"SELECT TIME, SCAN_NUMBER, ANTENNA1, ANTENNA2 FROM $ms WHERE {' AND '.join(conds)}")
    if sub.nrows() == 0:
        sub.close(); ms.close(); return {}
    r = {"times": sub.getcol("TIME"),
         "scans": sub.getcol("SCAN_NUMBER").astype(np.int32),
         "ant1": sub.getcol("ANTENNA1").astype(np.int32),
         "ant2": sub.getcol("ANTENNA2").astype(np.int32)}
    sub.close(); ms.close()
    return r


# -------------------------------------------------------------------
# Data reading — returns RAW (n_row, n_chan, n_corr), no 2x2
# -------------------------------------------------------------------

def _build_taql(ms_path, spw, fields=None, scans=None, time_range=None):
    conds = [f"DATA_DESC_ID == {spw}", "ANTENNA1 != ANTENNA2"]
    if fields:
        ft_tab = None
        try:
            from casacore.tables import table
            with suppress_stderr():
                ft_tab = table(f"{ms_path}::FIELD", readonly=True, ack=False)
            names = list(ft_tab.getcol("NAME"))
            ids = [i for i, n in enumerate(names) if n in fields]
            if ids: conds.append(f"FIELD_ID IN [{','.join(str(i) for i in ids)}]")
        finally:
            if ft_tab: ft_tab.close()
    if scans:
        conds.append(f"SCAN_NUMBER IN [{','.join(str(s) for s in scans)}]")
    if time_range:
        conds.append(f"TIME >= {time_range[0]} AND TIME <= {time_range[1]}")
    return " AND ".join(conds)


def read_data(ms_path, spw, fields=None, scans=None,
              data_col="DATA", model_col="MODEL_DATA",
              chan_slice=slice(None), time_range=None):
    """Read raw visibilities. Returns (n_row, n_chan, n_corr) arrays.
    No 2x2 conversion — that happens later on averaged data."""
    from casacore.tables import table, taql

    with suppress_stderr():
        ms = table(ms_path, readonly=True, ack=False)
        query = _build_taql(ms_path, spw, fields, scans, time_range)
        sub = taql(f"SELECT ROWID() AS RID, * FROM $ms WHERE {query}")

    if sub.nrows() == 0:
        sub.close(); ms.close(); return {}

    row_ids = sub.getcol("RID")
    vis_obs = sub.getcol(data_col)                    # (n_row, n_chan, n_corr)

    if model_col not in sub.colnames():
        sub.close(); ms.close()
        raise ValueError(f"Model column '{model_col}' not found in MS.")
    vis_model = sub.getcol(model_col)

    flags = sub.getcol("FLAG")                        # (n_row, n_chan, n_corr) bool
    if "FLAG_ROW" in sub.colnames():
        fr = sub.getcol("FLAG_ROW")
        flags = flags | fr[:, None, None]

    ant1 = sub.getcol("ANTENNA1").astype(np.int32)
    ant2 = sub.getcol("ANTENNA2").astype(np.int32)
    times = sub.getcol("TIME")
    scan_col = sub.getcol("SCAN_NUMBER").astype(np.int32)
    fids = sub.getcol("FIELD_ID").astype(np.int32)

    sub.close(); ms.close()

    spw_tab = table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True, ack=False)
    freqs_full = spw_tab.getcol("CHAN_FREQ")[spw]
    spw_tab.close()

    # Channel selection — slice numpy, no extra copy
    freqs = freqs_full[chan_slice]
    if chan_slice != slice(None):
        vis_obs = vis_obs[:, chan_slice, :]
        vis_model = vis_model[:, chan_slice, :]
        flags = flags[:, chan_slice, :]

    return {
        "vis_obs": vis_obs,       # (n_row, n_chan, n_corr) complex
        "vis_model": vis_model,   # (n_row, n_chan, n_corr) complex
        "flags": flags,           # (n_row, n_chan, n_corr) bool
        "ant1": ant1, "ant2": ant2,
        "times": times, "scans": scan_col, "field_ids": fids,
        "freqs": freqs, "row_ids": row_ids,
    }


# -------------------------------------------------------------------
# 2x2 conversion — only used on tiny averaged data
# -------------------------------------------------------------------

def raw_to_2x2(v):
    """Convert (*, n_corr) -> (*, 2, 2).  Works on any leading dims."""
    shape = v.shape[:-1]
    nc = v.shape[-1]
    out = np.zeros(shape + (2, 2), dtype=v.dtype)
    out[..., 0, 0] = v[..., 0]
    if nc > 2:
        out[..., 0, 1] = v[..., 1]
        out[..., 1, 0] = v[..., 2]
    out[..., 1, 1] = v[..., -1]
    return out


def flags_to_2x2(fl):
    """Convert flag (*, n_corr) -> (*, 2, 2)."""
    shape = fl.shape[:-1]
    nc = fl.shape[-1]
    out = np.zeros(shape + (2, 2), dtype=bool)
    out[..., 0, 0] = fl[..., 0]
    if nc > 2:
        out[..., 0, 1] = fl[..., 1]
        out[..., 1, 0] = fl[..., 2]
    out[..., 1, 1] = fl[..., -1]
    return out


# -------------------------------------------------------------------
# Write
# -------------------------------------------------------------------

def write_corrected(ms_path, row_ids, corrected, output_col="CORRECTED_DATA"):
    """Write corrected (n_row, n_chan, 2, 2) back to MS."""
    from casacore.tables import table, makearrcoldesc, maketabdesc

    nr, nc = corrected.shape[:2]
    flat = np.zeros((nr, nc, 4), dtype=np.complex128)
    flat[..., 0] = corrected[..., 0, 0]
    flat[..., 1] = corrected[..., 0, 1]
    flat[..., 2] = corrected[..., 1, 0]
    flat[..., 3] = corrected[..., 1, 1]

    ms = table(ms_path, readonly=False, ack=False)
    if output_col not in ms.colnames():
        desc = ms.getcoldesc("DATA")
        ms.addcols(maketabdesc(makearrcoldesc(
            output_col, 0+0j, ndim=desc.get("ndim", 2), shape=desc.get("shape"))))
    for i, rid in enumerate(row_ids):
        ms.putcell(output_col, int(rid), flat[i])
    ms.close()


def write_flags(ms_path, row_ids, flags):
    """Write flags (n_row, n_chan, 2, 2) back to MS FLAG column. OR with existing."""
    from casacore.tables import table

    nr, nc = flags.shape[:2]
    flat = np.zeros((nr, nc, 4), dtype=bool)
    flat[..., 0] = flags[..., 0, 0]
    flat[..., 1] = flags[..., 0, 1]
    flat[..., 2] = flags[..., 1, 0]
    flat[..., 3] = flags[..., 1, 1]

    ms = table(ms_path, readonly=False, ack=False)
    for i, rid in enumerate(row_ids):
        existing = ms.getcell("FLAG", int(rid))
        ms.putcell("FLAG", int(rid), existing | flat[i, :, :existing.shape[-1]])
    ms.close()


# -------------------------------------------------------------------
# Solint grid — scan-boundary aware
# -------------------------------------------------------------------

def compute_solint_grid(times, solint_s, scans=None):
    """Partition timestamps into solint blocks.
    solint_s: inf=all, -1.0=per-scan, >0=seconds."""
    if len(times) == 0: return []

    if scans is not None:
        utimes = np.unique(times)
        t2s = {}
        for t, s in zip(times, scans):
            if t not in t2s: t2s[t] = s
        uscans = np.array([t2s[t] for t in utimes])
    else:
        utimes = np.unique(times)
        uscans = None

    is_scan_mode = (solint_s == -1.0)

    if not np.isfinite(solint_s) and not is_scan_mode and uscans is None:
        return [utimes]

    blocks, block = [], [utimes[0]]
    t_start = utimes[0]
    cur_scan = uscans[0] if uscans is not None else None

    for i in range(1, len(utimes)):
        t = utimes[i]
        scan_break = (uscans is not None and uscans[i] != cur_scan)
        time_break = (not is_scan_mode and np.isfinite(solint_s)
                      and solint_s > 0 and t - t_start >= solint_s)

        if scan_break or time_break:
            blocks.append(np.array(block))
            block = [t]; t_start = t
            if uscans is not None: cur_scan = uscans[i]
        else:
            block.append(t)

    if block: blocks.append(np.array(block))
    return blocks
