"""ALAKAZAM Fluxscale.

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
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..io.hdf5 import (
    load_solutions, load_all_fields, save_fluxscale,
    copy_solutions, rescale_solutions, list_spws, _path_exists,
)
from ..config import FluxscaleBlock

logger = logging.getLogger("alakazam")


def compute_scale(
    g_ref: np.ndarray,       # (n_sol_t, n_ant, 2, 2) complex128
    g_transfer: np.ndarray,
    ref_ant: Optional[int] = None,
) -> Tuple[float, float, float, float, int]:
    """Compute fluxscale factors for both polarisations.

    Returns: (scale_p, scale_q, scatter_p, scatter_q, n_ant_used)
    """
    # Use diagonal elements (gains are diagonal)
    amp_ref = np.abs(g_ref[:, :, 0, 0])       # (n_t, n_ant)
    amp_trn = np.abs(g_transfer[:, :, 0, 0])

    amp_ref_q = np.abs(g_ref[:, :, 1, 1])
    amp_trn_q = np.abs(g_transfer[:, :, 1, 1])

    # Median over time then ratio per antenna
    med_ref_p = np.median(amp_ref, axis=0)   # (n_ant,)
    med_trn_p = np.median(amp_trn, axis=0)
    med_ref_q = np.median(amp_ref_q, axis=0)
    med_trn_q = np.median(amp_trn_q, axis=0)

    # Exclude ref antenna and flagged (zero) antennas
    valid = (med_trn_p > 1e-10) & (med_ref_p > 1e-10)
    if ref_ant is not None:
        valid[ref_ant] = False

    if valid.sum() < 1:
        logger.warning("fluxscale: no valid antennas for scale computation, returning 1.0")
        return 1.0, 1.0, 0.0, 0.0, 0

    ratios_p = (med_ref_p[valid] / med_trn_p[valid]) ** 2
    ratios_q = (med_ref_q[valid] / med_trn_q[valid]) ** 2

    scale_p   = float(np.median(ratios_p))
    scatter_p = float(np.std(ratios_p))
    scale_q   = float(np.median(ratios_q))
    scatter_q = float(np.std(ratios_q))
    n_ant     = int(valid.sum())

    logger.info(
        f"fluxscale: scale_p={scale_p:.4f} (scatter={scatter_p:.4f})  "
        f"scale_q={scale_q:.4f} (scatter={scatter_q:.4f})  n_ant={n_ant}"
    )
    return scale_p, scale_q, scatter_p, scatter_q, n_ant


def run_fluxscale(fb: FluxscaleBlock) -> None:
    """Execute one fluxscale block.

    Reads reference and transfer solutions, computes scale factors,
    writes rescaled solutions to output file.
    """
    output = fb.output
    jones_type = fb.jones_type

    # Determine SPWs
    ref_spws = list_spws(fb.reference_table, jones_type, fb.reference_field[0])
    if fb.spw is not None:
        spws = [s for s in fb.spw if s in ref_spws]
    else:
        spws = ref_spws

    if not spws:
        logger.warning(f"fluxscale: no SPWs found in {fb.reference_table}")
        return

    # Copy transfer_table → output (once, if not already done)
    if not _path_exists(output):
        logger.info(f"fluxscale: copying {fb.transfer_table} → {output}")
        shutil.copy2(fb.transfer_table, output)
    else:
        # Append — copy any missing Jones groups from transfer_table
        copy_solutions(fb.transfer_table, output)

    # Process each reference × transfer pair × SPW
    for ref_field in fb.reference_field:
        for trn_field in fb.transfer_field:
            logger.info(f"fluxscale: {ref_field} → {trn_field}")

            for spw in spws:
                try:
                    ref_data = load_solutions(fb.reference_table, jones_type, ref_field, spw)
                    trn_data = load_solutions(fb.transfer_table, jones_type, trn_field, spw)
                except KeyError as e:
                    logger.error(f"fluxscale: {e}")
                    continue

                scale_p, scale_q, scatter_p, scatter_q, n_ant = compute_scale(
                    ref_data["jones"], trn_data["jones"]
                )

                # Rescale transfer solutions in the output file
                rescale_solutions(output, jones_type, trn_field, spw, scale_p, scale_q)

                # Store scale factors
                save_fluxscale(
                    output,
                    transfer_field=trn_field,
                    spw=spw,
                    scale_p=scale_p,
                    scale_q=scale_q,
                    scatter_p=scatter_p,
                    scatter_q=scatter_q,
                    n_ant=n_ant,
                    reference_field=ref_field,
                    reference_table=fb.reference_table,
                    jones_type=jones_type,
                )

    logger.info(f"fluxscale: written to {output}")
