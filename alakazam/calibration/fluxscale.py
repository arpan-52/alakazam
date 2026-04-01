"""ALAKAZAM v1 Fluxscale.

Bootstrap absolute flux scale from reference field(s) to transfer field(s).

Algorithm (per SPW, per polarisation):
  For each reference field:
    R   = median( |g_ref[a]| / |g_trn[a]| )   over antennas (sigma-clipped)
    M   = mean( |MODEL_DATA| )                 for the reference field
    F   = R / sqrt(M)                          Jones multiplier
  Combine F from all reference fields (median).
  Apply:  g_scaled = g_transfer * F
  Report: S_trn = 1 / F**2                    derived flux of transfer cal

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations

import logging
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..io.hdf5 import (load_solutions, save_fluxscale, copy_solutions,
                        rescale_solutions, list_spws, _exists)
from ..config import FluxscaleBlock, spw_ids_from_selection

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


# ------------------------------------------------------------------
# Read model amplitude from the MS
# ------------------------------------------------------------------

def _read_model_amp(
    ms_path: str, model_col: str, field_name: str, spw: int,
) -> Tuple[float, float]:
    """Mean |MODEL_DATA| per pol for one field / spw.

    Reads a single baseline to stay RAM-cheap.
    Returns (model_amp_p, model_amp_q).
    """
    from casacore.tables import table, taql
    from ..core.ms_io import suppress_stderr

    with suppress_stderr():
        ms = table(ms_path, readonly=True, ack=False)

    # Resolve field name -> id
    ft = table(f"{ms_path}::FIELD", readonly=True, ack=False)
    names = list(ft.getcol("NAME")); ft.close()
    fid = next((i for i, n in enumerate(names) if n == field_name), None)
    if fid is None:
        ms.close()
        raise KeyError(f"Field {field_name!r} not found in {ms_path}")

    query = (f"DATA_DESC_ID == {spw} AND FIELD_ID == {fid} "
             f"AND ANTENNA1 != ANTENNA2")
    sub = taql(f"SELECT {model_col} FROM $ms WHERE {query}")

    if sub.nrows() == 0:
        sub.close(); ms.close()
        raise ValueError(f"No rows for field={field_name}, spw={spw} in {ms_path}")

    # Read one row — point-source model is constant across baselines
    model = sub.getcol(model_col, startrow=0, nrow=1)  # (1, n_chan, n_corr)
    sub.close(); ms.close()

    n_corr = model.shape[2]
    q_idx = 3 if n_corr == 4 else 1

    amp_p = float(np.mean(np.abs(model[0, :, 0])))
    amp_q = float(np.mean(np.abs(model[0, :, q_idx])))
    return amp_p, amp_q


# ------------------------------------------------------------------
# Compute amplitude ratio
# ------------------------------------------------------------------

def compute_scale(
    g_ref: np.ndarray,
    g_transfer: np.ndarray,
    ref_ant: Optional[int] = None,
    sigma_clip: float = 3.0,
    n_clip_iter: int = 3,
) -> Tuple[float, float, float, float, int]:
    """Amplitude ratio of reference to transfer gains, per polarisation.

    Returns: (ratio_p, ratio_q, scatter_p, scatter_q, n_ant_used)
    where ratio = median( |g_ref[a]| / |g_trn[a]| ) after sigma-clipping.
    """
    def _mean_amp_per_ant(g, pol):
        a = np.abs(g[..., pol, pol])
        if a.ndim == 1:
            return a
        return np.mean(a.reshape(a.shape[0], -1), axis=1)

    amp_ref_p = _mean_amp_per_ant(g_ref, 0)
    amp_trn_p = _mean_amp_per_ant(g_transfer, 0)
    amp_ref_q = _mean_amp_per_ant(g_ref, 1)
    amp_trn_q = _mean_amp_per_ant(g_transfer, 1)

    valid = (amp_trn_p > 1e-10) & (amp_ref_p > 1e-10)
    if ref_ant is not None and ref_ant < len(valid):
        valid[ref_ant] = False

    if valid.sum() < 1:
        logger.warning("fluxscale: no valid antennas, returning 1.0")
        return 1.0, 1.0, 0.0, 0.0, 0

    ratios_p = amp_ref_p / amp_trn_p
    ratios_q = amp_ref_q / amp_trn_q

    mask = valid.copy()
    for _ in range(n_clip_iter):
        if mask.sum() < 2:
            break
        med_p = np.median(ratios_p[mask])
        std_p = np.std(ratios_p[mask])
        med_q = np.median(ratios_q[mask])
        std_q = np.std(ratios_q[mask])
        outlier_p = (np.abs(ratios_p - med_p) > sigma_clip * std_p) if std_p > 0 else np.zeros(len(mask), bool)
        outlier_q = (np.abs(ratios_q - med_q) > sigma_clip * std_q) if std_q > 0 else np.zeros(len(mask), bool)
        new_mask = mask & ~outlier_p & ~outlier_q
        if new_mask.sum() == mask.sum():
            break
        logger.info(f"fluxscale: sigma-clip rejected {mask.sum() - new_mask.sum()} antennas")
        mask = new_mask

    if mask.sum() < 1:
        logger.warning("fluxscale: all antennas rejected by sigma-clip")
        return 1.0, 1.0, 0.0, 0.0, 0

    ratio_p = float(np.median(ratios_p[mask]))
    scatter_p = float(np.std(ratios_p[mask]))
    ratio_q = float(np.median(ratios_q[mask]))
    scatter_q = float(np.std(ratios_q[mask]))
    n_ant = int(mask.sum())

    return ratio_p, ratio_q, scatter_p, scatter_q, n_ant


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def run_fluxscale(fb: FluxscaleBlock) -> None:
    """Execute one fluxscale block."""
    output = fb.output
    jones_type = fb.jones_type

    # Determine SPWs
    ref_spws = list_spws(fb.reference_table, jones_type,
                         fb.reference_field[0])
    sel_spw_ids = spw_ids_from_selection(fb.spw)
    if sel_spw_ids is not None:
        spws = [s for s in sel_spw_ids if s in ref_spws]
    else:
        spws = ref_spws

    if not spws:
        logger.warning(f"fluxscale: no SPWs found in {fb.reference_table}")
        return

    # Copy transfer -> output
    if not _exists(output):
        logger.info(f"fluxscale: copying {fb.transfer_table} -> {output}")
        shutil.copy2(fb.transfer_table, output)
    else:
        copy_solutions(fb.transfer_table, output)

    _log(f"  Jones type: {jones_type}", "bold cyan")
    _log(f"  Reference: {fb.reference_field} from {fb.reference_table}", "dim")
    _log(f"  Transfer:  {fb.transfer_field} from {fb.transfer_table}", "dim")
    _log(f"  MS: {fb.ms}  model_col: {fb.model_col}", "dim")

    for trn_field in fb.transfer_field:
        for spw in spws:
            # Load transfer gains once
            try:
                trn_data = load_solutions(
                    fb.transfer_table, jones_type, trn_field, spw)
            except KeyError as e:
                logger.error(f"    SPW {spw} {trn_field}: {e}")
                continue

            # Gather Jones multiplier from each reference field
            factors_p, factors_q = [], []

            for ref_field in fb.reference_field:
                try:
                    ref_data = load_solutions(
                        fb.reference_table, jones_type, ref_field, spw)
                except KeyError as e:
                    logger.error(f"    SPW {spw} ref={ref_field}: {e}")
                    continue

                rp, rq, scp, scq, n_ant = compute_scale(
                    ref_data["jones"], trn_data["jones"])

                # Model amplitude for this reference field
                mp, mq = _read_model_amp(fb.ms, fb.model_col, ref_field, spw)

                fp = rp / np.sqrt(mp)
                fq = rq / np.sqrt(mq)
                factors_p.append(fp)
                factors_q.append(fq)

                # Derived transfer flux from this reference
                s_p = 1.0 / fp ** 2
                s_q = 1.0 / fq ** 2
                _log(f"    SPW {spw}: ref={ref_field}  "
                     f"model=({mp:.4f}, {mq:.4f})  "
                     f"R=({rp:.4f}, {rq:.4f})  "
                     f"S_trn=({s_p:.4f}, {s_q:.4f}) Jy  "
                     f"n_ant={n_ant}", "green")

            if not factors_p:
                logger.error(f"    SPW {spw} {trn_field}: "
                             f"no reference fields produced a scale")
                continue

            final_fp = float(np.median(factors_p))
            final_fq = float(np.median(factors_q))

            # Derived flux density of transfer calibrator
            s_trn_p = 1.0 / final_fp ** 2
            s_trn_q = 1.0 / final_fq ** 2
            s_trn_I = 0.5 * (s_trn_p + s_trn_q)

            _log(f"    SPW {spw}: {trn_field}  "
                 f"derived flux I={s_trn_I:.4f} Jy  "
                 f"(p={s_trn_p:.4f}, q={s_trn_q:.4f})", "bold green")

            rescale_solutions(output, jones_type, trn_field,
                              spw, final_fp, final_fq)

            save_fluxscale(
                output,
                transfer_field=trn_field, spw=spw,
                scale_p=final_fp, scale_q=final_fq,
                scatter_p=float(np.std(factors_p)) if len(factors_p) > 1 else 0.0,
                scatter_q=float(np.std(factors_q)) if len(factors_q) > 1 else 0.0,
                n_ant=n_ant,
                reference_field=",".join(fb.reference_field),
                reference_table=fb.reference_table,
                jones_type=jones_type,
            )

    _log(f"  Fluxscale saved to {output}", "bold green")
