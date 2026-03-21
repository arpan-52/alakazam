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
from ..core.ms_io import detect_metadata, read_data, write_corrected, write_flags, raw_to_2x2
from ..core.interpolation import interpolate_jones_multifield
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

        for fid in tgt_fids:
            fname = meta.field_names[fid]
            _log(f"  Applying: SPW {spw}  field={fname}", "bold")

            d = read_data(ab.ms, spw, fields=[fname], scans=ab.target_scans,
                          data_col="DATA", model_col="MODEL_DATA",
                          chan_slice=chan_sl)
            if not d:
                logger.warning(f"  no data for {fname} spw={spw}"); continue

            vis = raw_to_2x2(d["vis_obs"])  # (n_row, n_chan, 4) -> (n_row, n_chan, 2, 2)
            # Remap ant indices to active set
            ant1 = np.array([ant_remap[a] for a in d["ant1"]], dtype=np.int32)
            ant2 = np.array([ant_remap[a] for a in d["ant2"]], dtype=np.int32)
            row_times = d["times"]
            row_ids = d["row_ids"]
            unique_times = np.unique(row_times)

            # Parang — only active antennas
            parang = None
            if ab.apply_parang and feed_basis:
                pa_full = compute_parallactic_angles(ab.ms, unique_times, field=fname)
                parang = pa_full[:, active_ants]  # (n_time, n_active)

            # Load and interpolate each Jones term
            chain_jones = []
            for term in ab.terms:
                fdata = _load_term(term, spw, meta, fname, active_names)
                if not fdata: continue
                tgt_ra = meta.field_ra[fid] if meta.field_ra else None
                tgt_dec = meta.field_dec[fid] if meta.field_dec else None
                J = interpolate_jones_multifield(
                    fdata, unique_times, freqs, term.field_select,
                    term.time_interp, target_ra=tgt_ra, target_dec=tgt_dec,
                    pinned_fields=term.solution_field,
                    jones_label=term.jones, target_field=fname)
                chain_jones.append(J)

            if not chain_jones:
                logger.warning(f"  no Jones available for {fname} spw={spw}, writing uncorrected")
                write_corrected(ab.ms, row_ids, vis, ab.output_col)
                continue

            # Apply per unique time
            corrected = vis.copy()
            flag_arr = np.zeros_like(vis, dtype=bool) if ab.propagate_flags else None

            n_chan = vis.shape[1]

            for ti, t in enumerate(unique_times):
                mask = row_times == t
                if not mask.any(): continue

                # All Jones are (n_active, n_freq, 2, 2) — always 4D
                jones_at_t = []
                if parang is not None and feed_basis:
                    P = parang_to_jones(parang[ti], feed_basis)  # (n_active, 2, 2)
                    jones_at_t.append(
                        np.broadcast_to(P[:, np.newaxis, :, :],
                                        (n_active, n_chan, 2, 2)).copy())

                for J_seq in chain_jones:
                    jones_at_t.append(J_seq[ti])  # (n_active, n_freq, 2, 2)

                J_total = compose_jones_chain(jones_at_t)
                if J_total is None: continue

                corrected[mask] = unapply_jones_to_rows(
                    J_total, corrected[mask], ant1[mask], ant2[mask])

                # Flag propagation: check for NaN/Inf in J_total
                if flag_arr is not None:
                    for ri, (a1v, a2v) in enumerate(zip(ant1[mask], ant2[mask])):
                        bad = (np.any(np.isnan(J_total[a1v])) or
                               np.any(np.isnan(J_total[a2v])) or
                               np.any(np.isinf(J_total[a1v])) or
                               np.any(np.isinf(J_total[a2v])))
                        if bad:
                            flag_arr[np.where(mask)[0][ri]] = True

            write_corrected(ab.ms, row_ids, corrected, ab.output_col)

            if flag_arr is not None and np.any(flag_arr):
                write_flags(ab.ms, row_ids, flag_arr)
                logger.info(f"    Propagated {int(flag_arr.sum())} flags to MS")

            _log(f"    Written {ab.output_col} for {fname} SPW {spw} "
                 f"({len(row_ids)} rows)", "green")

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
        out[fn] = {
            "times": sol["time"], "freqs": sol.get("freq"),
            "jones": jones, "ra_rad": sol.get("ra_rad", 0.0),
            "dec_rad": sol.get("dec_rad", 0.0),
        }
    fields_str = ", ".join(
        f"{fn}({len(sol['times'])}t)" for fn, sol in out.items())
    n_ant = next(iter(out.values()))["jones"].shape[0]
    logger.info(f"    {term.jones}: loaded [{fields_str}] ({n_ant} ants)")
    return out
