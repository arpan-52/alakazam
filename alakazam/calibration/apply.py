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
from ..core.ms_io import detect_metadata, read_data, write_corrected, write_flags
from ..core.interpolation import interpolate_jones_multifield
from ..io.hdf5 import load_all_fields
from ..jones.algebra import (compose_jones_chain, detect_feed_basis,
                             unapply_jones_to_rows)
from ..jones.parang import compute_parallactic_angles, parang_to_jones

logger = logging.getLogger("alakazam")


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

    for spw in spws:
        freqs_full = meta.spw_freqs[spw]
        chan_sl = chan_slice_for_spw(ab.spw, spw, len(freqs_full))
        freqs = freqs_full[chan_sl]

        for fid in tgt_fids:
            fname = meta.field_names[fid]
            logger.info(f"  apply: SPW {spw} field={fname}")

            d = read_data(ab.ms, spw, fields=[fname], scans=ab.target_scans,
                          data_col="DATA", model_col="MODEL_DATA",
                          chan_slice=chan_sl)
            if not d:
                logger.warning(f"  no data for {fname} spw={spw}"); continue

            vis = d["vis_obs"]
            ant1, ant2 = d["ant1"], d["ant2"]
            row_times = d["times"]
            row_ids = d["row_ids"]
            unique_times = np.unique(row_times)

            # Parang
            parang = None
            if ab.apply_parang and feed_basis:
                parang = compute_parallactic_angles(ab.ms, unique_times, field=fname)

            # Load and interpolate each Jones term
            chain_jones = []
            for term in ab.terms:
                fdata = _load_term(term, spw, meta, fname)
                if not fdata: continue
                tgt_ra = meta.field_ra[fid] if meta.field_ra else None
                tgt_dec = meta.field_dec[fid] if meta.field_dec else None
                J = interpolate_jones_multifield(
                    fdata, unique_times, freqs, term.field_select,
                    term.time_interp, target_ra=tgt_ra, target_dec=tgt_dec,
                    pinned_fields=term.solution_field)
                chain_jones.append(J)

            if not chain_jones:
                logger.error(f"  no Jones for {fname} spw={spw}"); continue

            # Apply per unique time
            corrected = vis.copy()
            flag_arr = np.zeros_like(vis, dtype=bool) if ab.propagate_flags else None

            for ti, t in enumerate(unique_times):
                mask = row_times == t
                if not mask.any(): continue

                jones_at_t = []
                if parang is not None and feed_basis:
                    jones_at_t.append(parang_to_jones(parang[ti], feed_basis))

                for J_seq in chain_jones:
                    if J_seq.ndim >= 4:
                        Jt = J_seq[ti]  # (n_ant, [n_f,] 2, 2)
                        # n_freq=1 means same solution for all channels → squeeze to 3D
                        if Jt.ndim == 4 and Jt.shape[1] == 1:
                            Jt = Jt[:, 0, :, :]  # (n_ant, 2, 2)
                        jones_at_t.append(Jt)
                    elif J_seq.ndim == 3:
                        jones_at_t.append(J_seq)

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
                logger.info(f"  propagated {flag_arr.sum()} flags")

            logger.info(f"  written {ab.output_col} SPW {spw} {fname}")

    logger.info(f"apply: done in {_time.time()-t0:.1f}s")


def _load_term(term, spw, meta, target_fname):
    raw = load_all_fields(term.table, term.jones, spw)
    if not raw: return {}
    out = {}
    for fn, sol in raw.items():
        native = sol.get("native_params") or {}
        if native and "type" not in native:
            native["type"] = term.jones
        out[fn] = {
            "times": sol["time"], "freqs": sol.get("freq"),
            "jones": sol["jones"], "ra_rad": sol.get("ra_rad", 0.0),
            "dec_rad": sol.get("dec_rad", 0.0),
            "native_params": native if native else None,
        }
    return out
