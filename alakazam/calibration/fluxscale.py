"""ALAKAZAM v1 Fluxscale.

Bootstrap absolute flux scale from a known reference field to transfer fields.

Algorithm:
  For each SPW and polarisation:
    scale = median( |g_ref[a]| / |g_transfer[a]| )^2  over antennas
  Apply: g_scaled = g_transfer * sqrt(scale)

Output file contains:
  - All solutions from transfer_table copied
  - Rescaled G solutions for transfer_field(s)
  - fluxscale/ group with scale factors and provenance

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


def compute_scale(
    g_ref: np.ndarray,
    g_transfer: np.ndarray,
    ref_ant: Optional[int] = None,
    sigma_clip: float = 3.0,
    n_clip_iter: int = 3,
) -> Tuple[float, float, float, float, int]:
    """Compute fluxscale factors for both polarisations.

    Uses iterative sigma-clipping to reject outlier antennas.

    Returns: (scale_p, scale_q, scatter_p, scatter_q, n_ant_used)
    """
    amp_ref_p = np.abs(g_ref[..., 0, 0])
    amp_trn_p = np.abs(g_transfer[..., 0, 0])
    amp_ref_q = np.abs(g_ref[..., 1, 1])
    amp_trn_q = np.abs(g_transfer[..., 1, 1])

    # Median over time axis (axis=0)
    if amp_ref_p.ndim > 1:
        med_ref_p = np.median(amp_ref_p, axis=0)
        med_trn_p = np.median(amp_trn_p, axis=0)
        med_ref_q = np.median(amp_ref_q, axis=0)
        med_trn_q = np.median(amp_trn_q, axis=0)
    else:
        med_ref_p = amp_ref_p
        med_trn_p = amp_trn_p
        med_ref_q = amp_ref_q
        med_trn_q = amp_trn_q

    valid = (med_trn_p > 1e-10) & (med_ref_p > 1e-10)
    if ref_ant is not None and ref_ant < len(valid):
        valid[ref_ant] = False

    if valid.sum() < 1:
        logger.warning("fluxscale: no valid antennas, returning 1.0")
        return 1.0, 1.0, 0.0, 0.0, 0

    ratios_p = (med_ref_p / med_trn_p) ** 2
    ratios_q = (med_ref_q / med_trn_q) ** 2

    # Iterative sigma-clipping
    mask = valid.copy()
    for _ in range(n_clip_iter):
        if mask.sum() < 2:
            break
        med_p = np.median(ratios_p[mask])
        std_p = np.std(ratios_p[mask])
        med_q = np.median(ratios_q[mask])
        std_q = np.std(ratios_q[mask])
        if std_p > 0:
            outlier_p = np.abs(ratios_p - med_p) > sigma_clip * std_p
        else:
            outlier_p = np.zeros_like(mask)
        if std_q > 0:
            outlier_q = np.abs(ratios_q - med_q) > sigma_clip * std_q
        else:
            outlier_q = np.zeros_like(mask)
        new_mask = mask & (~outlier_p) & (~outlier_q)
        if new_mask.sum() == mask.sum():
            break  # converged
        n_rejected = mask.sum() - new_mask.sum()
        if n_rejected > 0:
            logger.info(f"fluxscale: sigma-clip iter rejected "
                        f"{n_rejected} antennas")
        mask = new_mask

    if mask.sum() < 1:
        logger.warning("fluxscale: all antennas rejected by sigma-clip")
        return 1.0, 1.0, 0.0, 0.0, 0

    scale_p = float(np.median(ratios_p[mask]))
    scatter_p = float(np.std(ratios_p[mask]))
    scale_q = float(np.median(ratios_q[mask]))
    scatter_q = float(np.std(ratios_q[mask]))
    n_ant = int(mask.sum())

    logger.info(
        f"fluxscale: scale_p={scale_p:.4f} (scatter={scatter_p:.4f})  "
        f"scale_q={scale_q:.4f} (scatter={scatter_q:.4f})  n_ant={n_ant}")
    return scale_p, scale_q, scatter_p, scatter_q, n_ant


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

    logger.info(f"  Jones type: {jones_type}")
    logger.info(f"  Reference: {fb.reference_field} from {fb.reference_table}")
    logger.info(f"  Transfer:  {fb.transfer_field} from {fb.transfer_table}")

    for ref_field in fb.reference_field:
        for trn_field in fb.transfer_field:
            logger.info(f"  Bootstrapping: {ref_field} -> {trn_field}")

            for spw in spws:
                try:
                    ref_data = load_solutions(
                        fb.reference_table, jones_type, ref_field, spw)
                    trn_data = load_solutions(
                        fb.transfer_table, jones_type, trn_field, spw)
                except KeyError as e:
                    logger.error(f"    SPW {spw}: {e}")
                    continue

                sp, sq, scp, scq, n_ant = compute_scale(
                    ref_data["jones"], trn_data["jones"])

                logger.info(f"    SPW {spw}: scale_p={sp:.4f} scale_q={sq:.4f} "
                            f"(scatter: {scp:.4f}, {scq:.4f}) "
                            f"using {n_ant} antennas")

                rescale_solutions(output, jones_type, trn_field,
                                  spw, sp, sq)

                save_fluxscale(
                    output,
                    transfer_field=trn_field, spw=spw,
                    scale_p=sp, scale_q=sq,
                    scatter_p=scp, scatter_q=scq,
                    n_ant=n_ant,
                    reference_field=ref_field,
                    reference_table=fb.reference_table,
                    jones_type=jones_type,
                )

    logger.info(f"  Fluxscale saved to {output}")
