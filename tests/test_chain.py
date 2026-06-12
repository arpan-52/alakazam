"""ALAKAZAM end-to-end chain tests on synthetic data.

Three runs through the real pipeline (solve -> fluxscale -> apply), each
against a known ground-truth Jones chain built by synth_ms:

  A. Diagonal chain (K, G amp-only) + unit-model fluxscale + apply.
     Everything is exactly solvable: corrected target == true sky to
     float32 storage precision, derived transfer flux == truth.
  B. Full polarization chain (K, G, KC, CP, D) with correct models.
     Per-term checks use degeneracy-free invariants (combined phase at
     band centre — constant phase trades against delay slope across a
     finite band, exactly as on real data).
  C. Leakage-only chain: D solver exact recovery.

Run:  python tests/test_chain.py
"""

import os
import sys
import tempfile

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from synth_ms import build_chain_jones, stokes_model, make_ms  # noqa: E402
from alakazam.config import load_config  # noqa: E402
from alakazam.flow import run_pipeline  # noqa: E402
from alakazam.io.hdf5 import load_solutions  # noqa: E402

N_ANT = 6
N_CHAN = 32
FREQS = 1.0e9 + 1.0e6 * np.arange(N_CHAN)
NU_MID = float(np.mean(FREQS))
TWO_PI = 2.0 * np.pi


def _write_cfg(tmp, name, cfg):
    path = os.path.join(tmp, name)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return load_config(path)


def _read_corrected(ms_path, field_id):
    from casacore.tables import table, taql
    t = table(ms_path, ack=False)
    sub = taql(f"SELECT CORRECTED_DATA FROM $t WHERE FIELD_ID == {field_id} "
               f"AND ANTENNA1 != ANTENNA2")
    corr = sub.getcol("CORRECTED_DATA")
    sub.close(); t.close()
    return corr


def _check(label, err, tol):
    status = "OK " if err < tol else "FAIL"
    print(f"  [{status}] {label}: err={err:.3e} (tol {tol:.0e})")
    assert err < tol, f"{label}: {err:.3e} >= {tol:.0e}"


def _wrap(p):
    return np.angle(np.exp(1j * p))


# ======================================================================
# Run A — diagonal chain + fluxscale
# ======================================================================

def run_a(tmp):
    print("\n=== Run A: K + G(amp) chain, unit-model fluxscale, apply ===")
    rng = np.random.default_rng(11)
    tau = np.zeros((N_ANT, 2))
    tau[1:] = rng.uniform(-3.0, 3.0, (N_ANT - 1, 2))      # ns, ref ant 0
    g_amp = rng.uniform(0.8, 1.2, (N_ANT, 2))             # ref amp free
    C = build_chain_jones(N_ANT, FREQS, tau_ns=tau, g_amp=g_amp)

    S_CAL1, S_CAL2, S_TGT = 1.6, 2.5, 0.8
    truth = [stokes_model(N_CHAN, I=S_CAL1), stokes_model(N_CHAN, I=S_CAL2),
             stokes_model(N_CHAN, I=S_TGT)]
    unit = [stokes_model(N_CHAN, I=1.0)] * 3               # solver sees 1 Jy

    ms = make_ms(os.path.join(tmp, "runA.ms"), FREQS, C,
                 scan_fields=[0, 1, 2, 0],
                 field_names=["CAL1", "CAL2", "TGT"],
                 models_true=truth, models_solve=unit, n_ant=N_ANT)
    h5 = os.path.join(tmp, "runA.h5")

    cfg = _write_cfg(tmp, "runA.yaml", {
        "solve": [{
            "ms": ms, "output": h5, "ref_ant": 0,
            # G1 is a phase-only step on the K0+G0-corrected residual: it
            # must come out identity (zero phases, amps pinned at 1).
            "jones": ["K", "G", "G"],
            "field": [["CAL1"], ["CAL1", "CAL2"], ["CAL1"]],
            "time_interval": ["inf", "scan", "inf"],
            "freq_interval": ["full", "full", "full"],
            "phase_only": [False, False, True],
        }],
        "fluxscale": [{
            "reference_table": h5, "reference_field": ["CAL1"],
            "transfer_table": h5, "transfer_field": ["CAL2"],
            "output": h5, "ms": ms, "model_col": "MODEL_TRUE",
            "jones_type": "G0",
        }],
        "apply": [{
            "ms": ms, "output_col": "CORRECTED_DATA",
            "target_field": ["TGT"],
            "jones": ["K0", "G0"], "tables": [h5, h5],
            "field_select": ["nearest_time", "pinned"],
            "solution_field": [None, ["CAL2"]],
            "time_interp": ["nearest", "nearest"],
        }],
    })
    run_pipeline(cfg)

    print("\n--- Run A assertions ---")
    k = load_solutions(h5, "K0", "CAL1", 0)
    _check("K delays vs truth (ns)",
           np.max(np.abs(k["delay"][:, 0, 0, :] - tau)), 1e-3)

    # Gains solved against 1 Jy model absorb sqrt(true flux)
    g1 = load_solutions(h5, "G0", "CAL1", 0)
    amp1 = np.abs(g1["jones"][:, 0, 0, [0, 1], [0, 1]])
    _check("G amps CAL1 = g*sqrt(1.6)",
           np.max(np.abs(amp1 / (g_amp * np.sqrt(S_CAL1)) - 1)), 1e-4)

    # Fluxscale: derived transfer flux and rescaled gains
    import h5py
    with h5py.File(h5) as f:
        a = f["fluxscale/field_CAL2/spw_0"].attrs
        s_derived = 0.5 * (1 / a["scale_p"] ** 2 + 1 / a["scale_q"] ** 2)
    _check("fluxscale derived flux = 2.5", abs(s_derived - S_CAL2), 1e-3)

    g2 = load_solutions(h5, "G0", "CAL2", 0)
    amp2 = np.abs(g2["jones"][:, 0, 0, [0, 1], [0, 1]])
    _check("rescaled CAL2 gains = true g", np.max(np.abs(amp2 / g_amp - 1)), 1e-4)

    # Phase-only step on fully-corrected residual: identity, amps exactly 1
    g3 = load_solutions(h5, "G1", "CAL1", 0)
    g3d = g3["jones"][:, 0, 0, [0, 1], [0, 1]]
    _check("phase-only G1 amps exactly 1", np.max(np.abs(np.abs(g3d) - 1)), 1e-12)
    _check("phase-only G1 phases ~ 0 (rad)", np.max(np.abs(np.angle(g3d))), 1e-5)

    # Rerunning fluxscale must refuse (in-place rescale would square the scale)
    from alakazam.calibration.fluxscale import run_fluxscale
    try:
        run_fluxscale(cfg.fluxscale_blocks[0])
        raise AssertionError("second fluxscale run did not raise")
    except ValueError as e:
        assert "already" in str(e)
        print("  [OK ] fluxscale rerun guard raised ValueError")

    # End to end: corrected target == true sky
    corr = _read_corrected(ms, 2)
    expect = truth[2].reshape(N_CHAN, 4)[None, :, :]
    _check("corrected TGT == true sky (rel)",
           np.max(np.abs(corr - expect)) / S_TGT, 1e-4)


# ======================================================================
# Run B — full polarization chain
# ======================================================================

def run_b(tmp):
    print("\n=== Run B: full chain K, G, KC, CP, D ===")
    rng = np.random.default_rng(23)
    tau = np.zeros((N_ANT, 2))
    base = rng.uniform(-3.0, 3.0, N_ANT - 1)
    tau[1:, 0] = base
    tau[1:, 1] = base + rng.uniform(-0.2, 0.2, N_ANT - 1)
    g_amp = rng.uniform(0.85, 1.15, (N_ANT, 2))
    g_phase = np.zeros((N_ANT, 2))
    g_phase[1:] = rng.uniform(-0.1, 0.1, (N_ANT - 1, 2))
    d_pq = (rng.uniform(-1, 1, N_ANT) + 1j * rng.uniform(-1, 1, N_ANT)) * 0.012
    d_qp = (rng.uniform(-1, 1, N_ANT) + 1j * rng.uniform(-1, 1, N_ANT)) * 0.012
    d_pq[0] = 0.0                                          # solver gauge
    tau_kc, phi_cp = 3.0, 0.4

    C = build_chain_jones(N_ANT, FREQS, tau_ns=tau, g_amp=g_amp,
                          g_phase=g_phase, d_pq=d_pq, d_qp=d_qp,
                          tau_kc_ns=tau_kc, phi_cp=phi_cp)
    cal = stokes_model(N_CHAN, I=1.0, Q=0.12, U=0.15)
    tgt = stokes_model(N_CHAN, I=0.8, Q=-0.05, U=0.06)

    ms = make_ms(os.path.join(tmp, "runB.ms"), FREQS, C,
                 scan_fields=[0, 1, 0],
                 field_names=["CAL1", "TGT"],
                 models_true=[cal, tgt], models_solve=[cal, tgt], n_ant=N_ANT)
    h5 = os.path.join(tmp, "runB.h5")

    cfg = _write_cfg(tmp, "runB.yaml", {
        "solve": [{
            "ms": ms, "output": h5, "ref_ant": 0,
            "jones": ["K", "G", "KC", "D", "CP"],
            "field": [["CAL1"]] * 5,
            "time_interval": ["inf", "scan", "inf", "inf", "inf"],
            # D per 4 MHz bin: the effective leakage rotates with frequency
            # (conjugation by the KC term it is solved under), so a
            # full-band D average decorrelates by ~|d|*sin(w*tau_c*B/2).
            "freq_interval": ["full", "full", "full", "4MHz", "full"],
            # K with constant gain phases in the data converges through a
            # slow Gauss-Newton tail (~140 iters) — give it room.
            "max_iter": 300,
        }],
        "apply": [{
            "ms": ms, "output_col": "CORRECTED_DATA",
            "target_field": ["TGT"],
            "jones": ["K0", "G0", "KC0", "D0", "CP0"],
            "tables": [h5] * 5,
            "field_select": "nearest_time",
            "time_interp": "nearest",
        }],
    })
    run_pipeline(cfg)

    print("\n--- Run B assertions ---")
    k = load_solutions(h5, "K0", "CAL1", 0)
    tau_hat = k["delay"][:, 0, 0, :]
    # Constant G phase trades against delay slope across the finite band
    # (phase/2*pi*nu_mid ~ 0.016 ns here) — allow for it.
    _check("K delays vs truth (ns)", np.max(np.abs(tau_hat - tau)), 0.05)

    g = load_solutions(h5, "G0", "CAL1", 0)
    gd = g["jones"][:, 0, 0, [0, 1], [0, 1]]
    _check("G amps vs truth (rel)", np.max(np.abs(np.abs(gd) / g_amp - 1)), 1e-2)

    # Degeneracy-free invariant: combined K+G phase at band centre
    theta_hat = -TWO_PI * tau_hat * 1e-9 * NU_MID + np.angle(gd)
    theta_tru = -TWO_PI * tau * 1e-9 * NU_MID + g_phase
    dtheta = _wrap((theta_hat - theta_tru)
                   - (theta_hat[0] - theta_tru[0]))       # ref-relative
    _check("combined K+G phase @ band centre (rad)", np.max(np.abs(dtheta)), 1e-2)

    # KC is solved first on cross-hands carrying both the delay slope and
    # the constant CP rotation. Over a narrow band at absolute frequency a
    # constant phase is degenerate with delay, so KC absorbs CP's constant:
    #   tau_hat = tau_c + phi_cp/(2*pi*nu_mid)
    # Leakage cross-talk adds only a small residual on top. The exact
    # invariant is the total cross-hand phase at band centre.
    kc = load_solutions(h5, "KC0", "CAL1", 0)
    tau_c_hat = float(kc["delay"][0, 0, 0, 0])
    cp = load_solutions(h5, "CP0", "CAL1", 0)
    phi_hat = float(np.angle(cp["jones"][0, 0, 0, 1, 1]))
    tau_c_expect = tau_kc + phi_cp / (TWO_PI * NU_MID) * 1e9
    _check("KC delay = tau_c + phi_cp/(2 pi nu_mid) (ns)",
           abs(tau_c_hat - tau_c_expect), 5e-3)
    _check("CP ~ 0 (constant absorbed by KC) (rad)", abs(phi_hat), 5e-2)
    xph_hat = -TWO_PI * tau_c_hat * 1e-9 * NU_MID - phi_hat
    xph_tru = -TWO_PI * tau_kc * 1e-9 * NU_MID - phi_cp
    _check("combined KC+CP cross phase @ band centre (rad)",
           abs(_wrap(xph_hat - xph_tru)), 1e-2)

    # D is solved after diagonal preapply -> gauge-conjugated leakage:
    # |d_hat_pq| = |d_pq|*|g_q/g_p|, |d_hat_qp| = |d_qp|/|g_q/g_p| (delay
    # and absorbed-CP rotations cancel in magnitude). Because KC already
    # took the constant rotation, the cross-hands reach D clean — no
    # offset lands in the free d_qp's.
    d = load_solutions(h5, "D0", "CAL1", 0)
    rho = g_amp[:, 1] / g_amp[:, 0]
    d_hat_pq = d["jones"][:, 0, 0, 0, 1]
    d_hat_qp = d["jones"][:, 0, 0, 1, 0]
    _check("D d_pq magnitudes vs gauge-conjugated truth",
           np.max(np.abs(np.abs(d_hat_pq) - np.abs(d_pq) * rho)), 5e-3)
    _check("D d_qp magnitudes vs gauge-conjugated truth",
           np.max(np.abs(np.abs(d_hat_qp) - np.abs(d_qp) / rho)), 5e-3)

    corr = _read_corrected(ms, 1)
    expect = tgt.reshape(N_CHAN, 4)[None, :, :]
    err = np.abs(corr - expect) / 0.8
    # Parallel hands invert tightly; cross-hands carry the residual
    # leakage <-> KC/CP cross-talk bias (chained solving, not a bug) plus
    # some LM-endpoint scatter on GPU builds — allow more there.
    _check("corrected TGT parallel hands (rel)",
           float(max(err[..., 0].max(), err[..., 3].max())), 1e-2)
    _check("corrected TGT cross hands (rel)",
           float(max(err[..., 1].max(), err[..., 2].max())), 3e-2)


# ======================================================================
# Run C — leakage only, exact
# ======================================================================

def run_c(tmp):
    print("\n=== Run C: D-only chain (exact) ===")
    rng = np.random.default_rng(37)
    d_pq = (rng.uniform(-1, 1, N_ANT) + 1j * rng.uniform(-1, 1, N_ANT)) * 0.04
    d_qp = (rng.uniform(-1, 1, N_ANT) + 1j * rng.uniform(-1, 1, N_ANT)) * 0.04
    d_pq[0] = 0.0
    C = build_chain_jones(N_ANT, FREQS, d_pq=d_pq, d_qp=d_qp)
    cal = stokes_model(N_CHAN, I=1.0, Q=0.1, U=0.08)
    tgt = stokes_model(N_CHAN, I=0.8, Q=-0.04, U=0.05)

    ms = make_ms(os.path.join(tmp, "runC.ms"), FREQS, C,
                 scan_fields=[0, 1],
                 field_names=["CAL1", "TGT"],
                 models_true=[cal, tgt], models_solve=[cal, tgt], n_ant=N_ANT)
    h5 = os.path.join(tmp, "runC.h5")

    cfg = _write_cfg(tmp, "runC.yaml", {
        "solve": [{
            "ms": ms, "output": h5, "ref_ant": 0,
            "jones": ["D"], "field": [["CAL1"]],
            "time_interval": ["inf"], "freq_interval": ["full"],
        }],
        "apply": [{
            "ms": ms, "output_col": "CORRECTED_DATA",
            "target_field": ["TGT"],
            "jones": ["D0"], "tables": [h5],
            "field_select": "nearest_time", "time_interp": "nearest",
        }],
    })
    run_pipeline(cfg)

    print("\n--- Run C assertions ---")
    d = load_solutions(h5, "D0", "CAL1", 0)
    _check("d_pq exact", np.max(np.abs(d["jones"][:, 0, 0, 0, 1] - d_pq)), 1e-5)
    _check("d_qp exact", np.max(np.abs(d["jones"][:, 0, 0, 1, 0] - d_qp)), 1e-5)

    corr = _read_corrected(ms, 1)
    expect = tgt.reshape(N_CHAN, 4)[None, :, :]
    _check("corrected TGT == true sky (rel)",
           np.max(np.abs(corr - expect)) / 0.8, 1e-4)


def main():
    tmp = tempfile.mkdtemp(prefix="alakazam_chain_")
    print(f"scratch dir: {tmp}")
    run_a(tmp)
    run_b(tmp)
    run_c(tmp)
    print("\nALL CHAIN TESTS PASSED")


if __name__ == "__main__":
    main()
