"""ALAKAZAM v1 Apply.

Load Jones chain from HDF5, interpolate onto target grid, compose,
unapply to visibilities. Diagonal Jones optimized.

Optionally propagate solution flags to MS FLAG column.
Parallactic angle supported if apply_parang=True.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations
import json, logging, time as _time
from typing import Any, Dict, List, Optional
import numpy as np

from ..config import ApplyBlock, ApplyTerm, spw_ids_from_selection, chan_slice_for_spw
from ..core.ms_io import detect_metadata, read_data, write_corrected, write_flags, raw_to_2x2, query_times_scans
from ..core.interpolation import interpolate_jones_multifield
from ..core.memory import get_available_ram_gb
from ..io.hdf5 import load_all_fields
from ..jones.algebra import (compose_jones_chain, detect_feed_basis,
                             unapply_jones_to_rows)
from ..jones.parang import compute_parallactic_angles, parang_to_jones


logger = logging.getLogger("alakazam")

try:
    from rich.console import Console
    _con = Console()
except ImportError:
    _con = None

def _log(msg, style=""):
    logger.info(msg)
    if _con and style: _con.print(msg, style=style)
    elif _con: _con.print(msg)


def apply_calibration(ab: ApplyBlock) -> None:
    t0 = _time.time()
    meta = detect_metadata(ab.ms)

    spws = spw_ids_from_selection(ab.spw)
    if spws is None:
        spws = list(range(meta.n_spw))

    if ab.target_field:
        tgt_fids = [i for i, n in enumerate(meta.field_names) if n in ab.target_field]
    else:
        tgt_fids = list(meta.field_ids)

    if not tgt_fids:
        logger.warning(f"apply: no fields in {ab.ms}"); return

    feed_basis = detect_feed_basis(ab.ms) if ab.apply_parang else None

    # Active antennas — only those with data
    active_ants = sorted(set(meta.ant1.tolist() + meta.ant2.tolist()))
    n_active = len(active_ants)
    ant_remap = {orig: new for new, orig in enumerate(active_ants)}
    active_names = [meta.ant_names[a] for a in active_ants]

    jones_names = [t.jones for t in ab.terms]
    _log(f"  Jones chain: {' -> '.join(jones_names)}", "bold cyan")
    _log(f"  Active antennas: {n_active}/{meta.n_ant}", "dim")
    if ab.apply_parang:
        _log(f"  Parallactic angle correction: ON ({feed_basis.value if feed_basis else '?'})", "dim")
    _log(f"  Output column: {ab.output_col}", "dim")

    for spw in spws:
        freqs_full = meta.spw_freqs[spw]
        chan_sl = chan_slice_for_spw(ab.spw, spw, len(freqs_full))
        freqs = freqs_full[chan_sl]

        n_chan = len(freqs)
        # Peak memory during read_data:
        #   getcol(DATA)       -> obs array (c128, 16 B/el) allocated
        #   getcol(MODEL_DATA) -> casacore internal temp (c128) + model array allocated
        #                         obs still alive -> peak = obs + model + temp = 3 × c128
        #   After read + corrected copy: obs + model + corrected = 3 × c128
        # Use 3 × c128 + bool for flags as the peak per-element cost.
        bpr = n_chan * meta.n_corr * (3 * 16 + 1)  # 3×c128 + bool

        for fid in tgt_fids:
            fname = meta.field_names[fid]
            _log(f"  Applying: SPW {spw}  field={fname}", "bold")

            # Lightweight query — just timestamps, no vis data
            ts = query_times_scans(ab.ms, spw, fields=[fname],
                                   scans=ab.target_scans)
            if not ts:
                logger.warning(f"  no data for {fname} spw={spw}"); continue
            unique_times = np.unique(ts["times"])
            tgt_ra = meta.field_ra[fid] if meta.field_ra else None
            tgt_dec = meta.field_dec[fid] if meta.field_dec else None

            # Rows per timestamp from lightweight query
            row_times_all = ts["times"]
            rows_per_ts = int(np.median(
                np.bincount(np.searchsorted(unique_times, row_times_all))))
            rows_per_ts = max(rows_per_ts, 1)

            # Interpolate Jones for ALL unique timestamps upfront (small arrays)
            chain_jones = []
            for term in ab.terms:
                fdata = _load_term(term, spw, meta, fname, active_names)
                if not fdata: continue
                J = interpolate_jones_multifield(
                    fdata, unique_times, freqs, term.field_select,
                    term.time_interp, target_ra=tgt_ra, target_dec=tgt_dec,
                    pinned_fields=term.solution_field,
                    jones_label=term.jones, target_field=fname)
                chain_jones.append(J)

            # Parang for all unique_times
            parang = None
            if ab.apply_parang and feed_basis:
                pa_full = compute_parallactic_angles(ab.ms, unique_times, field=fname)
                parang = pa_full[:, active_ants]  # (n_time, n_active)

            # Measure available RAM AFTER Jones loaded, so chunk sizing
            # accounts for Jones memory already allocated.
            avail_bytes = int(get_available_ram_gb() * 0.4 * 1024**3)
            avail_bytes = max(avail_bytes, 100 * 1024**2)
            max_rows = max(1, avail_bytes // bpr)
            ts_per_chunk = max(1, max_rows // rows_per_ts)

            logger.info(f"    {len(unique_times)} timestamps, "
                        f"{bpr} B/row, chunk={ts_per_chunk} timestamps "
                        f"(budget {avail_bytes/1e9:.1f} GB after Jones loaded)")

            if not chain_jones:
                logger.warning(f"  no Jones for {fname} spw={spw}, writing uncorrected")
                for ci in range(0, len(unique_times), ts_per_chunk):
                    chunk_ts = unique_times[ci:ci + ts_per_chunk]
                    d = read_data(ab.ms, spw, fields=[fname], scans=ab.target_scans,
                                  data_col="DATA", model_col="MODEL_DATA",
                                  chan_slice=chan_sl,
                                  time_range=(float(chunk_ts[0]) - 0.01,
                                              float(chunk_ts[-1]) + 0.01))
                    if d:
                        write_corrected(ab.ms, d["row_ids"],
                                        raw_to_2x2(d["vis_obs"]), ab.output_col)
                        del d
                continue

            total_rows_written = 0

            # Process in time chunks
            for ci in range(0, len(unique_times), ts_per_chunk):
                chunk_ts = unique_times[ci:ci + ts_per_chunk]
                t_lo = float(chunk_ts[0]) - 0.01
                t_hi = float(chunk_ts[-1]) + 0.01

                d = read_data(ab.ms, spw, fields=[fname], scans=ab.target_scans,
                              data_col="DATA", model_col="MODEL_DATA",
                              chan_slice=chan_sl, time_range=(t_lo, t_hi))
                if not d: continue

                vis = raw_to_2x2(d["vis_obs"])
                ant1 = np.array([ant_remap[a] for a in d["ant1"]], dtype=np.int32)
                ant2 = np.array([ant_remap[a] for a in d["ant2"]], dtype=np.int32)
                row_times = d["times"]
                row_ids = d["row_ids"]
                del d

                corrected = vis.copy()
                flag_arr = np.zeros_like(vis, dtype=bool) if ab.propagate_flags else None

                for chunk_ti, t in enumerate(chunk_ts):
                    ti = ci + chunk_ti          # index into full unique_times
                    mask = row_times == t
                    if not mask.any(): continue

                    jones_at_t = []
                    if parang is not None and feed_basis:
                        P = parang_to_jones(parang[ti], feed_basis)
                        jones_at_t.append(
                            np.broadcast_to(P[:, np.newaxis, :, :],
                                            (n_active, n_chan, 2, 2)).copy())
                    for J_seq in chain_jones:
                        jones_at_t.append(J_seq[ti])

                    J_total = compose_jones_chain(jones_at_t)
                    if J_total is None: continue

                    corrected[mask] = unapply_jones_to_rows(
                        J_total, corrected[mask], ant1[mask], ant2[mask])

                    if flag_arr is not None:
                        row_indices = np.where(mask)[0]
                        for ri, (a1v, a2v) in zip(row_indices,
                                                   zip(ant1[mask], ant2[mask])):
                            if (np.any(~np.isfinite(J_total[a1v])) or
                                    np.any(~np.isfinite(J_total[a2v]))):
                                flag_arr[ri] = True

                write_corrected(ab.ms, row_ids, corrected, ab.output_col)
                total_rows_written += len(row_ids)

                if flag_arr is not None and np.any(flag_arr):
                    write_flags(ab.ms, row_ids, flag_arr)

                del vis, corrected, ant1, ant2, row_times, row_ids

            _log(f"    Written {ab.output_col} for {fname} SPW {spw} "
                 f"({total_rows_written} rows)", "green")

    _log(f"  Apply complete in {_time.time()-t0:.1f}s", "bold green")


def _remap_jones(jones, sol_ant_names, active_names):
    """Remap Jones from solution antenna order to active antenna order by name.

    jones: (n_sol_ant, ..., 2, 2)
    Returns: (n_active, ..., 2, 2). Unmatched active antennas get identity.
    """
    n_active = len(active_names)
    shape = (n_active,) + jones.shape[1:]
    J = np.zeros(shape, dtype=jones.dtype)
    J[..., 0, 0] = 1.0
    J[..., 1, 1] = 1.0
    sol_map = {name: i for i, name in enumerate(sol_ant_names)}
    for act_idx, name in enumerate(active_names):
        if name in sol_map:
            J[act_idx] = jones[sol_map[name]]
    return J


def _remap_delay(delay, sol_ant_names, active_names):
    """Remap delay from solution antenna order to active antenna order by name.

    delay: (n_sol_ant, n_freq, n_time, 2)
    Returns: (n_active, n_freq, n_time, 2). Unmatched antennas get zero delay.
    """
    n_active = len(active_names)
    shape = (n_active,) + delay.shape[1:]
    D = np.zeros(shape, dtype=delay.dtype)
    sol_map = {name: i for i, name in enumerate(sol_ant_names)}
    for act_idx, name in enumerate(active_names):
        if name in sol_map:
            D[act_idx] = delay[sol_map[name]]
    return D


def _load_term(term, spw, meta, target_fname, active_names):
    import json
    raw = load_all_fields(term.table, term.jones, spw)
    if not raw:
        logger.warning(f"    {term.jones}: no solutions found in {term.table} spw={spw}")
        return {}
    out = {}
    for fn, sol in raw.items():
        jones = sol["jones"]
        attrs = sol.get("attrs", {})
        if "ant_names" in attrs:
            sol_ant_names = json.loads(attrs["ant_names"])
            if sol_ant_names != active_names:
                jones = _remap_jones(jones, sol_ant_names, active_names)
        delay = sol.get("delay")
        if delay is not None and "ant_names" in attrs:
            sol_ant_names = json.loads(attrs["ant_names"])
            if sol_ant_names != active_names:
                delay = _remap_delay(delay, sol_ant_names, active_names)
        out[fn] = {
            "times": sol["time"], "freqs": sol.get("freq"),
            "jones": jones, "delay": delay,
            "ra_rad": sol.get("ra_rad", 0.0),
            "dec_rad": sol.get("dec_rad", 0.0),
        }
    fields_str = ", ".join(
        f"{fn}({len(sol['times'])}t)" for fn, sol in out.items())
    n_ant = next(iter(out.values()))["jones"].shape[0]
    logger.info(f"    {term.jones}: loaded [{fields_str}] ({n_ant} ants)")
    return out
