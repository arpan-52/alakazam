"""
Apply Calibration Solutions to Measurement Set.

Applies Jones solutions to MS data with scan/SPW/field selection support.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict
import logging

logger = logging.getLogger('jackal')


def apply_calibration(
    ms_path: Union[str, Path],
    jones_solutions: Dict[str, np.ndarray],  # {'K': jones_K, 'G': jones_G, ...}
    jones_metadata: Dict[str, 'MSMetadata'],  # Metadata for each Jones type
    input_col: str = 'DATA',
    output_col: str = 'CORRECTED_DATA',
    field: Optional[str] = None,
    spw: Optional[str] = None,
    scans: Optional[str] = None,
    inverse: bool = False
) -> Dict:
    """
    Apply calibration solutions to MS.

    Parameters
    ----------
    ms_path : str or Path
        Path to measurement set
    jones_solutions : dict
        Dictionary of Jones matrices by type: {'K': jones_K, 'G': jones_G, ...}
        Each Jones: (n_sol_time, n_sol_freq, n_ant, 2, 2)
    jones_metadata : dict
        Dictionary of MSMetadata for each Jones type
    input_col : str
        Input column name (default: 'DATA')
    output_col : str
        Output column name (default: 'CORRECTED_DATA')
    field : str, optional
        Field selection
    spw : str, optional
        SPW selection: '0', '0~2', '0,2,4'
    scans : str, optional
        Scan selection: '1', '1~10', '1,3,5'
    inverse : bool
        If True, apply inverse (unapply) of Jones matrices

    Returns
    -------
    stats : dict
        Application statistics
    """
    from casacore.tables import table, taql
    from .interpolation import interpolate_jones_time, interpolate_jones_freq

    ms_path = Path(ms_path)
    logger.info("="*80)
    logger.info("APPLYING CALIBRATION")
    logger.info("="*80)
    logger.info(f"  MS: {ms_path}")
    logger.info(f"  Input column: {input_col}")
    logger.info(f"  Output column: {output_col}")
    if field:
        logger.info(f"  Field: {field}")
    if spw:
        logger.info(f"  SPW: {spw}")
    if scans:
        logger.info(f"  Scans: {scans}")
    logger.info(f"  Jones types: {list(jones_solutions.keys())}")
    logger.info(f"  Mode: {'unapply' if inverse else 'apply'}")
    logger.info("="*80)

    # Open MS
    ms = table(str(ms_path), readonly=False, ack=False)

    # Build selection
    conditions = []
    if field is not None:
        field_tab = table(f"{ms_path}::FIELD", readonly=True, ack=False)
        field_names = list(field_tab.getcol("NAME"))
        field_tab.close()
        if field in field_names:
            field_id = field_names.index(field)
            conditions.append(f"FIELD_ID=={field_id}")

    if spw is not None:
        if "~" in spw:
            start, end = spw.split("~")
            spw_ids = list(range(int(start), int(end) + 1))
        elif "," in spw:
            spw_ids = [int(s) for s in spw.split(",")]
        else:
            spw_ids = [int(spw)]
        conditions.append(f"DATA_DESC_ID IN [{','.join(map(str, spw_ids))}]")

    if scans is not None:
        if "~" in scans:
            start, end = scans.split("~")
            scan_ids = list(range(int(start), int(end) + 1))
        elif "," in scans:
            scan_ids = [int(s) for s in scans.split(",")]
        else:
            scan_ids = [int(scans)]
        conditions.append(f"SCAN_NUMBER IN [{','.join(map(str, scan_ids))}]")

    # Select data
    if conditions:
        query = f"SELECT * FROM $ms WHERE {' AND '.join(conditions)}"
        sel = taql(query)
    else:
        sel = ms

    n_rows = sel.nrows()
    logger.info(f"  Processing {n_rows} rows")

    if n_rows == 0:
        logger.warning("  No rows selected, nothing to apply")
        sel.close()
        ms.close()
        return {'n_rows': 0, 'n_applied': 0}

    # Read data
    logger.info(f"  Reading {input_col}...")
    vis = sel.getcol(input_col)  # (n_row, n_freq, 2, 2)
    antenna1 = sel.getcol("ANTENNA1")
    antenna2 = sel.getcol("ANTENNA2")
    times = sel.getcol("TIME")

    # Get frequency info
    spw_tab = table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True, ack=False)
    freq = spw_tab.getcol("CHAN_FREQ")[0]  # Assuming single SPW for now
    spw_tab.close()

    # Apply each Jones type
    vis_corrected = vis.copy()

    for jones_type, jones_array in jones_solutions.items():
        logger.info(f"  Applying {jones_type}...")

        meta = jones_metadata[jones_type]
        n_ant = meta.n_ant

        # For each row, interpolate Jones to row's time/freq and apply
        for row in range(n_rows):
            if row % 10000 == 0 and row > 0:
                logger.info(f"    Processed {row}/{n_rows} rows")

            a1, a2 = antenna1[row], antenna2[row]
            t = times[row]

            # Interpolate Jones to this time/frequency
            # For simplicity, use nearest time and full frequency
            # (Full implementation would interpolate properly)

            # Get nearest solution time index
            t_idx = np.argmin(np.abs(meta.time_intervals - t))

            # Get Jones for this time (all frequencies, all antennas)
            if jones_array.ndim == 5:
                # (n_sol_time, n_sol_freq, n_ant, 2, 2)
                J = jones_array[t_idx, 0, :, :, :]  # Use first freq solution for now
            elif jones_array.ndim == 4:
                # (n_sol_time, n_ant, 2, 2)
                J = jones_array[t_idx, :, :, :]
            else:
                # (n_ant, 2, 2)
                J = jones_array

            # Apply: V_corrected = J_a1^{-1} @ V @ J_a2^{-H}
            J1 = J[a1]
            J2 = J[a2]

            if inverse:
                # Unapply: V_out = J_a1 @ V @ J_a2^H
                for f in range(vis.shape[1]):
                    vis_corrected[row, f] = J1 @ vis_corrected[row, f] @ np.conj(J2.T)
            else:
                # Apply: V_out = J_a1^{-1} @ V @ J_a2^{-H}
                J1_inv = np.linalg.inv(J1)
                J2_inv = np.linalg.inv(J2)
                for f in range(vis.shape[1]):
                    vis_corrected[row, f] = J1_inv @ vis_corrected[row, f] @ np.conj(J2_inv.T)

        logger.info(f"    {jones_type} applied to all rows")

    # Write corrected data
    logger.info(f"  Writing to {output_col}...")

    # Ensure output column exists
    try:
        sel.getcol(output_col, nrow=1)
    except:
        logger.info(f"    Creating column {output_col}...")
        # Column doesn't exist, create it
        from casacore.tables import maketabdesc, makearrcoldesc
        desc = makearrcoldesc(output_col, vis[0], ndim=3)
        ms.addcols(maketabdesc([desc]))

    sel.putcol(output_col, vis_corrected)

    sel.close()
    ms.close()

    logger.info("="*80)
    logger.info(f"  CALIBRATION APPLIED SUCCESSFULLY")
    logger.info(f"  Corrected {n_rows} rows")
    logger.info("="*80)

    return {
        'n_rows': n_rows,
        'n_applied': n_rows,
        'jones_types': list(jones_solutions.keys())
    }


__all__ = ['apply_calibration']
