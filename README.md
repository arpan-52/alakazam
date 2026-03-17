# ALAKAZAM v1.0.0

**Radio Interferometric Calibration Pipeline**
Arpan Pal — NRAO / NCRA, 2026

---

## What it does

ALAKAZAM solves for Jones matrix calibration terms from radio interferometric
Measurement Sets (MS), applies flux scale bootstrapping, and applies corrected
solutions back to target data. Full RIME formalism with chained Jones terms,
multi-field solving, flexible interpolation, parallactic angle support, and
complete provenance tracking.

---

## Jones types

| Type | Name          | Matrix                                              | Constraint                        |
|------|---------------|-----------------------------------------------------|-----------------------------------|
| `K`  | Parallel delay | `diag(e^{-2πi τ_p ν}, e^{-2πi τ_q ν})`           | `τ[ref,:] = 0`                   |
| `G`  | Gains          | `diag(g_p e^{iφ_p}, g_q e^{iφ_q})`                | `φ[ref,:] = 0`, amp **free**     |
| `D`  | Leakage        | `[[1, d_pq], [d_qp, 1]]`                           | `d_pq[ref] = 0`, `d_qp[ref]` free |
| `KC` | Cross delay    | `diag(e^{-2πi τ_cross ν}, 1)` — single global param | none (1 parameter)               |
| `CP` | Cross phase    | `diag(1, e^{iφ_cross})` — single global param       | none (1 parameter)               |

**Feed bases**: LINEAR (XX/XY/YX/YY) and CIRCULAR (RR/RL/LR/LL) — auto-detected.

---

## Install

```bash
pip install .
```

Requires Python ≥ 3.9. Dependencies: numpy, scipy, numba, h5py, pyyaml, psutil, rich.
Optional: jax+jaxlib (default solver backend), torch, python-casacore.

---

## Quick start

```bash
alakazam run config.yaml          # run solve + fluxscale + apply
alakazam info cal.h5              # inspect a solution file
alakazam fluxscale-info cal.h5    # show stored scale factors
```

---

## Config format

A YAML file with three optional top-level blocks: `solve`, `fluxscale`, `apply`.
**Every per-step parameter can be a scalar (broadcast to all steps) or a list
(one entry per step).**

### `solve:`

```yaml
solve:
  - ms: all_cals.ms
    output: cal.h5
    ref_ant: 0
    data_col: DATA
    model_col: MODEL_DATA        # must exist in MS — error if missing
    apply_parang: false           # correct for parallactic angle before solving
    solver_backend: jax_scipy     # jax_scipy | torch_lbfgs
    memory_limit_gb: 0            # 0 = auto-detect
    rfi_threshold: 5.0
    max_iter: 100
    tol: 1.0e-10
    n_workers: 0                  # 0 = auto (cpu_count - 1)

    # Jones chain — one entry per solve step
    jones: [K, G, D, KC, CP]

    # Field groups — scalar or per-step lists
    # Scalar: same field for all steps
    # List: one group per step, each group can have multiple fields
    field: [[3C286], [3C286, 3C147], [3C286], [3C286], [3C286]]

    # Scan selection — absent = all scans
    # scans: [[1,2], [1,2,3,4], [5], [5], [5]]

    # SPW selection — CASA-native syntax
    # spw: "0:10~500,1"
    # Per-step SPW override:
    # step_spw: ["0,1", "0,1", "0:10~100", "0,1", "0,1"]

    # Per-step solve parameters (scalar = same for all steps)
    time_interval: [inf, 5min, inf, inf, inf]
    freq_interval: [full, full, full, full, full]
    phase_only: false             # scalar = all steps false
    preapply_time_interp: [exact, nearest, nearest, nearest, nearest]

    # External preapply — inject solutions from a previous run
    # external_preapply:
    #   tables: [prev.h5]
    #   jones: [K]
    #   field_select: [nearest_time]
    #   time_interp: [exact]
```

**Averaging per solver type:**
- K, KC: average in time only, keep frequency axis (they solve in frequency)
- G, D, CP: average in both time and frequency (frequency-independent Jones)

**Solint blocks never cross scan boundaries.**

**All solint blocks are solved in parallel.**

#### Time interval formats

| Value  | Meaning                |
|--------|------------------------|
| `inf`  | Whole observation      |
| `5min` | 5-minute slots         |
| `120s` | 120-second slots       |
| `30`   | 30-second slots        |

#### Frequency interval formats

| Value  | Meaning                |
|--------|------------------------|
| `full` | Whole SPW              |
| `4MHz` | 4 MHz bins             |
| `32`   | 32-channel bins        |

#### SPW selection (CASA-native)

| Value                    | Meaning                                    |
|--------------------------|--------------------------------------------|
| absent / `*`             | All SPWs, all channels                     |
| `0`                      | SPW 0, all channels                        |
| `0:127~156`              | SPW 0, channels 127-156                    |
| `0:127~156,1,2,5:10~20` | SPW 0 ch127-156, SPW 1 all, SPW 2 all, SPW 5 ch10-20 |

---

### `fluxscale:`

Bootstraps flux scale from a known calibrator. Uses iterative sigma-clipping
(3σ, 3 iterations) to reject outlier antennas.

```yaml
fluxscale:
  - reference_table: cal.h5
    reference_field: 3C147
    transfer_table: cal.h5
    transfer_field: [PKS1934, J0538]
    output: cal_scaled.h5
    jones_type: G
```

---

### `apply:`

```yaml
apply:
  - ms: science.ms
    output_col: CORRECTED_DATA
    target_field: NGC1234
    apply_parang: false
    propagate_flags: false     # write solution flags to MS FLAG column

    jones:  [K, G]
    tables: [cal.h5, cal_scaled.h5]
    field_select: [nearest_time, nearest_sky]
    time_interp: [exact, linear]
    # solution_field: [null, PKS1934]  # only for field_select=pinned
```

**Interpolation modes**: exact, nearest, linear, cubic
**Field selection**: nearest_time, nearest_sky, pinned

When `propagate_flags: true`, any antenna with a bad solution (NaN, Inf,
zero determinant) gets its rows flagged in the MS FLAG column. Existing
flags are never removed — only new flags added.

---

## Flagging

1. **MS flags respected**: FLAG column + FLAG_ROW read and merged
2. **RFI flagging**: MAD-based threshold on ALL 4 correlations independently.
   If any correlation at a (row, channel) is an outlier, all 4 are flagged.
3. **Solution flagging**: Per-antenna per-time-slot. Checks for NaN, Inf,
   zero determinant. Stored in HDF5 as `flags (n_t, n_ant)`.
4. **Flag propagation** (optional): Solution flags written to MS FLAG on apply.

---

## HDF5 solution format

Universal schema — **every Jones type uses the same shape**:

```
cal.h5
├── attrs: version, created_at, alakazam_version
├── K/
│   └── field_3C286/
│       └── spw_0/
│           ├── jones       (n_ant, n_freq, n_time, 2, 2) complex128
│           ├── flags       (n_ant, n_freq, n_time)         bool
│           ├── time        (n_time,) float64 MJD seconds
│           ├── freq        (n_freq,) float64 Hz
│           ├── solver_stats/
│           │   ├── converged  (n_freq, n_time) bool
│           │   ├── n_iter     (n_freq, n_time) int32
│           │   └── cost       (n_freq, n_time) float64
│           └── attrs:
│               jones_type, field_name, field_ra, field_dec,
│               spw, n_ant, n_freq, n_time,
│               matrix_form, ref_ant_constraint, ref_ant,
│               solint_s, freqint_hz, phase_only, solver_backend,
│               ms, scan_ids, preapply_chain, apply_parang,
│               n_cells_total, n_cells_converged
├── native_params/K/field_3C286/spw_0/
│   └── delay   (n_ant, n_freq, n_time, 2) float64 ns
└── fluxscale/field_PKS1934/spw_0/
    └── attrs: scale_p, scale_q, scatter_p, scatter_q,
               reference_field, reference_table, n_ant
```

The dimensions `n_freq` and `n_time` are determined entirely by the user's
`freq_interval` and `time_interval` settings. Every solver produces the same
shape — the difference is only in how the solver receives its input data
(K/KC get multi-channel data, G/D/CP get frequency-averaged data).

---

## Solve chain logic

```
raw data
  → [parang correction if apply_parang=true]
  → solve step 1 (K)               → stored
  → preapply step 1
  → solve step 2 (G)               → stored
  → preapply step 1 + step 2
  → solve step 3 (D)               → stored
  ...
```

Different solints across steps are handled transparently — each prior step's
solutions are interpolated onto the current data's time grid.

---

## Solver backends

| Backend       | Method                  | GPU support |
|---------------|-------------------------|-------------|
| `jax_scipy`   | JAX + scipy L-BFGS-B    | Auto        |
| `torch_lbfgs` | PyTorch LBFGS           | Auto        |

Falls back to scipy LM (Levenberg-Marquardt) if JAX/PyTorch not installed.
GPU/CPU auto-detected from available hardware.

---

## Architecture

```
alakazam/
├── __init__.py, __main__.py, cli.py
├── config.py          → YAML parser, CASA SPW syntax, validation
├── flow.py            → orchestrator: solve → fluxscale → apply
├── jones/
│   ├── algebra.py     → 2×2 ops, apply/unapply, residuals (numba)
│   ├── constructors.py → param → Jones (numba)
│   ├── constructors_ad.py → AD-compatible (pure numpy, for JAX/torch)
│   └── parang.py      → parallactic angle
├── solvers/
│   ├── __init__.py    → ABC, registry, backend detection
│   ├── parallel_delay.py (K), gains.py (G), leakage.py (D)
│   ├── cross_delay.py (KC), cross_phase.py (CP)
├── core/
│   ├── ms_io.py       → casacore I/O, FLAG+FLAG_ROW, scan-aware solint
│   ├── averaging.py   → baseline averaging (numba)
│   ├── interpolation.py → time+freq interp, field selection
│   ├── memory.py, rfi.py, quality.py
├── io/
│   └── hdf5.py        → solution read/write with full provenance
└── calibration/
    ├── apply.py       → load → interpolate → compose → unapply → write
    └── fluxscale.py   → sigma-clipped flux bootstrapping
```

---

Developed by Arpan Pal, NRAO / NCRA, 2026
