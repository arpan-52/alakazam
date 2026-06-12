"""Synthetic MS builder for ALAKAZAM chain tests.

Creates a tiny but fully valid Measurement Set (6 antennas, 1 SPW,
32 channels, 4-corr linear feeds) with DATA corrupted by a known
ground-truth Jones chain:

    C[a, f] = CP @ KC[f] @ D[a] @ G[a] @ K[a, f]
    DATA_ij  = C_i @ M_true @ C_j^H

MODEL_DATA holds the model the solver is given (e.g. a 1 Jy point for
the unit-model fluxscale convention); MODEL_TRUE holds the true sky
model (read by fluxscale via model_col).

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import os
import shutil

import numpy as np
from casacore.tables import default_ms, maketabdesc, makearrcoldesc, table


def build_chain_jones(n_ant, freqs, tau_ns=None, g_amp=None, g_phase=None,
                      d_pq=None, d_qp=None, tau_kc_ns=0.0, phi_cp=0.0):
    """Ground-truth chain C[a,f] = CP @ KC @ D @ G @ K, (n_ant, n_chan, 2, 2)."""
    n_chan = len(freqs)
    if tau_ns is None:  tau_ns = np.zeros((n_ant, 2))
    if g_amp is None:   g_amp = np.ones((n_ant, 2))
    if g_phase is None: g_phase = np.zeros((n_ant, 2))
    if d_pq is None:    d_pq = np.zeros(n_ant, complex)
    if d_qp is None:    d_qp = np.zeros(n_ant, complex)

    C = np.zeros((n_ant, n_chan, 2, 2), np.complex128)
    for a in range(n_ant):
        G = np.diag(g_amp[a] * np.exp(1j * g_phase[a]))
        D = np.array([[1.0, d_pq[a]], [d_qp[a], 1.0]])
        DG = D @ G
        for f, nu in enumerate(freqs):
            K = np.diag(np.exp(-2j * np.pi * tau_ns[a] * 1e-9 * nu))
            kc = np.exp(-2j * np.pi * tau_kc_ns * 1e-9 * nu)
            CPKC = np.diag([kc, np.exp(1j * phi_cp)])  # CP @ KC (both diagonal)
            C[a, f] = CPKC @ (DG @ K)
    return C


def stokes_model(n_chan, I=1.0, Q=0.0, U=0.0):
    """Point-source coherency in the linear basis: XX=I+Q, XY=YX=U, YY=I-Q."""
    M = np.zeros((n_chan, 2, 2), np.complex128)
    M[:, 0, 0] = I + Q
    M[:, 1, 1] = I - Q
    M[:, 0, 1] = U
    M[:, 1, 0] = U
    return M


def make_ms(path, freqs, C, scan_fields, field_names, models_true,
            models_solve, n_ant=6, n_t_per_scan=4, dt=10.0):
    """Write the synthetic MS.

    scan_fields:  field id per scan (scan numbers are 1-based)
    models_true:  per-field (n_chan, 2, 2) true sky — goes into DATA + MODEL_TRUE
    models_solve: per-field (n_chan, 2, 2) model given to the solver (MODEL_DATA)
    """
    if os.path.exists(path):
        shutil.rmtree(path)

    n_chan = len(freqs)
    n_corr = 4
    bl = [(i, j) for i in range(n_ant) for j in range(i + 1, n_ant)]
    n_bl = len(bl)

    t0 = 4.92e9  # MJD seconds
    rows = []
    for si, fid in enumerate(scan_fields):
        scan_no = si + 1
        scan_start = t0 + si * (n_t_per_scan + 4) * dt
        for ti in range(n_t_per_scan):
            t = scan_start + ti * dt
            for (a1, a2) in bl:
                rows.append((t, a1, a2, fid, scan_no))
    n_row = len(rows)

    times = np.array([r[0] for r in rows])
    ant1 = np.array([r[1] for r in rows], np.int32)
    ant2 = np.array([r[2] for r in rows], np.int32)
    fids = np.array([r[3] for r in rows], np.int32)
    scans = np.array([r[4] for r in rows], np.int32)

    def _vis_for(model_per_field, corrupt):
        v = np.zeros((n_row, n_chan, n_corr), np.complex128)
        for r in range(n_row):
            M = model_per_field[fids[r]]
            if corrupt:
                out = np.einsum("fab,fbc,fdc->fad",
                                C[ant1[r]], M, C[ant2[r]].conj())
            else:
                out = M
            v[r] = out.reshape(n_chan, 4)
        return v

    data = _vis_for(models_true, corrupt=True)
    model_solve = _vis_for(models_solve, corrupt=False)
    model_true = _vis_for(models_true, corrupt=False)

    # ---- main table ----
    extra = maketabdesc([
        makearrcoldesc("DATA", 0.0 + 0j, ndim=2, valuetype="complex"),
        makearrcoldesc("MODEL_DATA", 0.0 + 0j, ndim=2, valuetype="complex"),
        makearrcoldesc("MODEL_TRUE", 0.0 + 0j, ndim=2, valuetype="complex"),
    ])
    ms = default_ms(path, extra)
    ms.addrows(n_row)
    ms.putcol("TIME", times)
    ms.putcol("TIME_CENTROID", times)
    ms.putcol("ANTENNA1", ant1)
    ms.putcol("ANTENNA2", ant2)
    ms.putcol("FIELD_ID", fids)
    ms.putcol("SCAN_NUMBER", scans)
    ms.putcol("DATA_DESC_ID", np.zeros(n_row, np.int32))
    ms.putcol("FEED1", np.zeros(n_row, np.int32))
    ms.putcol("FEED2", np.zeros(n_row, np.int32))
    ms.putcol("ARRAY_ID", np.zeros(n_row, np.int32))
    ms.putcol("OBSERVATION_ID", np.zeros(n_row, np.int32))
    ms.putcol("PROCESSOR_ID", np.zeros(n_row, np.int32))
    ms.putcol("STATE_ID", np.zeros(n_row, np.int32))
    ms.putcol("INTERVAL", np.full(n_row, dt))
    ms.putcol("EXPOSURE", np.full(n_row, dt))
    ms.putcol("UVW", np.zeros((n_row, 3)))
    ms.putcol("SIGMA", np.ones((n_row, n_corr), np.float32))
    ms.putcol("WEIGHT", np.ones((n_row, n_corr), np.float32))
    ms.putcol("FLAG", np.zeros((n_row, n_chan, n_corr), bool))
    ms.putcol("FLAG_ROW", np.zeros(n_row, bool))
    ms.putcol("DATA", data)
    ms.putcol("MODEL_DATA", model_solve)
    ms.putcol("MODEL_TRUE", model_true)
    ms.close()

    # ---- ANTENNA ----
    with table(f"{path}/ANTENNA", readonly=False, ack=False) as t:
        t.addrows(n_ant)
        centre = np.array([-1601185.4, -5041977.5, 3554875.9])  # VLA-ish ITRF
        rng = np.random.default_rng(7)
        pos = centre[None, :] + rng.uniform(-2000, 2000, (n_ant, 3))
        t.putcol("POSITION", pos)
        t.putcol("OFFSET", np.zeros((n_ant, 3)))
        t.putcol("NAME", [f"A{i:02d}" for i in range(n_ant)])
        t.putcol("STATION", [f"S{i:02d}" for i in range(n_ant)])
        t.putcol("MOUNT", ["ALT-AZ"] * n_ant)
        t.putcol("TYPE", ["GROUND-BASED"] * n_ant)
        t.putcol("DISH_DIAMETER", np.full(n_ant, 25.0))
        t.putcol("FLAG_ROW", np.zeros(n_ant, bool))

    # ---- SPECTRAL_WINDOW ----
    df = freqs[1] - freqs[0]
    with table(f"{path}/SPECTRAL_WINDOW", readonly=False, ack=False) as t:
        t.addrows(1)
        t.putcell("NUM_CHAN", 0, n_chan)
        t.putcell("CHAN_FREQ", 0, freqs)
        t.putcell("CHAN_WIDTH", 0, np.full(n_chan, df))
        t.putcell("EFFECTIVE_BW", 0, np.full(n_chan, df))
        t.putcell("RESOLUTION", 0, np.full(n_chan, df))
        t.putcell("REF_FREQUENCY", 0, freqs[0])
        t.putcell("TOTAL_BANDWIDTH", 0, n_chan * df)
        t.putcell("MEAS_FREQ_REF", 0, 5)  # TOPO
        t.putcell("NAME", 0, "SPW0")
        t.putcell("NET_SIDEBAND", 0, 1)
        t.putcell("IF_CONV_CHAIN", 0, 0)
        t.putcell("FREQ_GROUP", 0, 0)
        t.putcell("FREQ_GROUP_NAME", 0, "G0")
        t.putcell("FLAG_ROW", 0, False)

    # ---- POLARIZATION (linear XX, XY, YX, YY) ----
    with table(f"{path}/POLARIZATION", readonly=False, ack=False) as t:
        t.addrows(1)
        t.putcell("NUM_CORR", 0, 4)
        t.putcell("CORR_TYPE", 0, np.array([9, 10, 11, 12], np.int32))
        t.putcell("CORR_PRODUCT", 0,
                  np.array([[0, 0, 1, 1], [0, 1, 0, 1]], np.int32))
        t.putcell("FLAG_ROW", 0, False)

    # ---- DATA_DESCRIPTION ----
    with table(f"{path}/DATA_DESCRIPTION", readonly=False, ack=False) as t:
        t.addrows(1)
        t.putcell("SPECTRAL_WINDOW_ID", 0, 0)
        t.putcell("POLARIZATION_ID", 0, 0)
        t.putcell("FLAG_ROW", 0, False)

    # ---- FIELD ----
    with table(f"{path}/FIELD", readonly=False, ack=False) as t:
        n_f = len(field_names)
        t.addrows(n_f)
        for i, name in enumerate(field_names):
            direction = np.array([[0.5 + 0.05 * i, 0.1 + 0.02 * i]])
            t.putcell("NAME", i, name)
            t.putcell("CODE", i, "")
            t.putcell("PHASE_DIR", i, direction)
            t.putcell("DELAY_DIR", i, direction)
            t.putcell("REFERENCE_DIR", i, direction)
            t.putcell("NUM_POLY", i, 0)
            t.putcell("SOURCE_ID", i, i)
            t.putcell("TIME", i, t0)
            t.putcell("FLAG_ROW", i, False)

    return path
