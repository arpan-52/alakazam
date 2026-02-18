"""ALAKAZAM Apply.

Apply a chain of Jones solutions to a target MS.

For each Jones term:
  1. Load all fields available for that jones_type in the table
  2. Select which field(s) to use based on field_select mode
  3. Interpolate (time + freq) onto the target time/freq grid
  4. Compose the full Jones chain
  5. Apply J_total^{-1} to visibilities → CORRECTED_DATA

Interpolation modes:
  nearest_time — nearest solution slot in time across all fields
  nearest_sky  — nearest field on sky, then nearest in time
  pinned       — use exactly the solution_field specified

Time/freq interpolation:
  exact        — stamp exact solint block; for K/Kcross recompute from delay at target freqs
  nearest_time — nearest slot (same as exact for time, nearest for freq)
  linear       — linear in time and freq
  cubic        — cubic spline in time and freq

Metadata provenance is stamped on every correction with full traceability.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations

import json
import logging
import time as _time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import ApplyBlock, ApplyTerm
from ..core.ms_io import detect_metadata, read_data
from ..core.interpolation import interpolate_jones_multifield
from ..io.hdf5 import load_all_fields, list_spws
from ..jones.algebra import compose_jones_chain, jones_unapply, jones_unapply_freq
from ..jones.parang import compute_parallactic_angles, parang_to_jones
from ..jones.algebra import FeedBasis

logger = logging.getLogger("alakazam")


def apply_calibration(ab: ApplyBlock) -> None:
    """Execute one apply: block."""
    t0 = _time.time()
    meta = detect_metadata(ab.ms)

    # Determine SPWs
    spws = ab.spw if ab.spw is not None else list(range(meta.n_spw))

    # Determine fields
    if ab.target_field is not None:
        target_fids = [
            i for i, n in enumerate(meta.field_names)
            if n in ab.target_field
        ]
    else:
        target_fids = list(meta.field_ids)

    if not target_fids:
        logger.warning(f"apply: no matching fields found in {ab.ms}")
        return

    logger.info(f"apply: {ab.ms}  fields={[meta.field_names[i] for i in target_fids]}"
                f"  spws={spws}  {len(ab.terms)} Jones terms")

    for spw in spws:
        freqs = meta.spw_freqs[spw]

        for fid in target_fids:
            fname = meta.field_names[fid]
            logger.info(f"  apply: SPW {spw}  field={fname}")

            # Read data
            d = read_data(
                ab.ms, spw,
                fields=[fname],
                scans=ab.target_scans,
                data_col="DATA",
                model_col="MODEL_DATA",
            )
            if not d:
                logger.warning(f"  apply: no data for field={fname} spw={spw}")
                continue

            vis      = d["vis_obs"]       # (n_row, n_chan, 2, 2)
            ant1     = d["ant1"]
            ant2     = d["ant2"]
            row_times = d["times"]        # (n_row,)
            unique_times = np.unique(row_times)

            # Parallactic angle correction
            if ab.apply_parang:
                feed_basis = FeedBasis.LINEAR  # detect_feed_basis could be called here
                parang = compute_parallactic_angles(ab.ms, unique_times, field=fname)
                # Will be applied per timestamp below

            # Build Jones chain per unique time, then broadcast to rows
            # For efficiency: interpolate all terms to unique_times grid,
            # then compose once per timestamp

            chain_jones = []  # list of (n_unique_t, n_ant, [n_freq,] 2, 2)

            provenance = []

            for term in ab.terms:
                logger.info(f"    loading {term.jones} from {term.table}")

                # Load all fields for this jones_type + spw
                fields_data = _load_fields_for_term(term, spw, meta, fname)

                if not fields_data:
                    logger.error(
                        f"    apply: no solutions found for {term.jones} in {term.table}"
                    )
                    continue

                # Determine target RA/Dec for sky-nearest selection
                target_ra  = meta.field_ra[fid]  if meta.field_ra  else None
                target_dec = meta.field_dec[fid] if meta.field_dec else None

                # Interpolate onto unique_times
                J_interp = interpolate_jones_multifield(
                    fields_data=fields_data,
                    target_times=unique_times,
                    target_freqs=freqs,
                    field_select=term.field_select,
                    time_interp=term.time_interp,
                    target_ra=target_ra,
                    target_dec=target_dec,
                    pinned_fields=term.solution_field,
                )
                # J_interp: (n_t, n_ant, [n_freq,] 2, 2)
                chain_jones.append(J_interp)

                provenance.append({
                    "jones":        term.jones,
                    "table":        term.table,
                    "field_select": term.field_select,
                    "time_interp":  term.time_interp,
                    "fields_used":  list(fields_data.keys()),
                })

            if not chain_jones:
                logger.error(f"  apply: no Jones loaded for field={fname} spw={spw}")
                continue

            # Apply per unique time slot
            corrected = vis.copy()

            t_idx_map = {t: i for i, t in enumerate(unique_times)}

            for t_slot_idx, t in enumerate(unique_times):
                row_mask = row_times == t
                if not row_mask.any():
                    continue

                # Compose chain for this time slot
                jones_at_t = []
                for J_seq in chain_jones:
                    if J_seq.ndim == 4:
                        # (n_t, n_ant, 2, 2) → pick slot
                        jones_at_t.append(J_seq[t_slot_idx])
                    elif J_seq.ndim == 5:
                        # (n_t, n_ant, n_freq, 2, 2) → pick slot
                        jones_at_t.append(J_seq[t_slot_idx])

                if ab.apply_parang:
                    P = parang_to_jones(parang[t_slot_idx], feed_basis)
                    jones_at_t.insert(0, P)

                J_total = compose_jones_chain(jones_at_t)

                vis_rows = corrected[row_mask]  # (n_row_t, n_chan, 2, 2)
                a1_rows  = ant1[row_mask]
                a2_rows  = ant2[row_mask]

                if J_total.ndim == 3:
                    # (n_ant, 2, 2) — freq-independent
                    corrected[row_mask] = jones_unapply(
                        J_total, vis_rows[:, 0], a1_rows, a2_rows
                    )[:, None, :, :].repeat(vis_rows.shape[1], axis=1)
                elif J_total.ndim == 4:
                    # (n_ant, n_freq, 2, 2)
                    corrected[row_mask] = jones_unapply_freq(
                        J_total, vis_rows, a1_rows, a2_rows
                    )

            # Write corrected data
            _write_corrected_rows(ab.ms, spw, d, corrected, ab.output_col)

            logger.info(
                f"  apply: written {ab.output_col}  SPW {spw}  field={fname}  "
                f"provenance={json.dumps(provenance)}"
            )

    logger.info(f"apply: done in {_time.time()-t0:.1f}s")


def _load_fields_for_term(
    term: ApplyTerm,
    spw: int,
    meta,
    target_fname: str,
) -> Dict[str, Dict]:
    """Load all available fields for a jones term into interpolation-ready dicts."""
    from ..io.hdf5 import load_all_fields

    raw = load_all_fields(term.table, term.jones, spw)
    if not raw:
        return {}

    fields_data = {}
    for fname, sol in raw.items():
        native_params = sol.get("native_params") or {}
        if native_params:
            native_params["type"] = term.jones

        fields_data[fname] = {
            "times":         sol["time"],
            "freqs":         sol.get("freq"),
            "jones":         sol["jones"],
            "ra_rad":        sol.get("ra_rad", 0.0),
            "dec_rad":       sol.get("dec_rad", 0.0),
            "native_params": native_params if native_params else None,
        }

    return fields_data


def _write_corrected_rows(
    ms_path: str,
    spw: int,
    data_dict: Dict,
    corrected: np.ndarray,  # (n_row, n_chan, 2, 2)
    output_col: str,
) -> None:
    """Write corrected visibilities back to MS."""
    from casacore.tables import table, taql

    n_row, n_chan = corrected.shape[:2]
    data_flat = np.zeros((n_row, n_chan, 4), dtype=np.complex128)
    data_flat[..., 0] = corrected[..., 0, 0]
    data_flat[..., 1] = corrected[..., 0, 1]
    data_flat[..., 2] = corrected[..., 1, 0]
    data_flat[..., 3] = corrected[..., 1, 1]

    ms = table(ms_path, readonly=False, ack=False)

    if output_col not in ms.colnames():
        from casacore.tables import makearrcoldesc, maketabdesc, tableutil
        desc = ms.getcoldesc("DATA")
        desc["name"] = output_col
        ms.addcols(maketabdesc(makearrcoldesc(
            output_col, 0+0j,
            ndim=desc.get("ndim", 2),
            shape=desc.get("shape"),
        )))

    # Use row indices from data_dict to write to correct rows
    # Build a TaQL to find the right rows
    ms.close()

    ms = table(ms_path, readonly=False, ack=False)
    # Write row by row using the stored row map
    # For simplicity: re-query to get row numbers
    from casacore.tables import taql
    sub = taql(f"SELECT ROWID() AS RID FROM $ms WHERE DATA_DESC_ID == {spw} AND ANTENNA1 != ANTENNA2")
    row_ids = sub.getcol("RID")
    sub.close()

    # Match by writing all rows for this SPW in order
    # (assumes data was read in same order — valid for taql with same selection)
    if len(row_ids) >= n_row:
        for i, rid in enumerate(row_ids[:n_row]):
            ms.putcell(output_col, int(rid), data_flat[i])

    ms.close()
