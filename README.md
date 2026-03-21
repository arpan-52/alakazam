# ALAKAZAM v1.0.0

Radio interferometric calibration pipeline.

Arpan Pal — NRAO / NCRA, 2026

## Install

```bash
pip install .

# With GPU (JAX + CUDA):
pip install .[all]
```

Requires: Python >= 3.9, python-casacore, numpy, scipy, numba, h5py, pyyaml, rich.
Optional: jax + jaxlib (GPU/CPU), pyceres.

## Quick start

```bash
alakazam run config.yaml         # run solve -> fluxscale -> apply
alakazam info cal.h5             # print solution summary
alakazam fluxscale-info cal.h5   # print fluxscale factors
```

## Jones types

| Key | Name | Matrix | Constraint | Freq-dependent |
|-----|------|--------|------------|----------------|
| K | Parallel delay | `diag(e^{-2piitau_p nu}, e^{-2piitau_q nu})` | `tau[ref,:]=0` | Fits across freq, produces 1 delay per pol |
| G | Complex gains | `diag(g_p e^{iphi_p}, g_q e^{iphi_q})` | `phi[ref,:]=0`, amp free | Per freq bin |
| D | Leakage | `[[1, d_pq], [d_qp, 1]]` | `d_pq[ref]=0`, `d_qp[ref]` free | Per freq bin |
| KC | Cross-hand delay | `diag(e^{-2piitau nu}, 1)` global | 1 parameter | Fits across freq, 1 delay |
| CP | Cross-hand phase | `diag(1, e^{iphi})` global | 1 parameter | Per freq bin |

Every solver receives averaged data for one cell and returns `(n_ant, 2, 2)` Jones matrices. The flow assembles these into the full `(n_ant, n_freq, n_time, 2, 2)` grid.

## Solver backends

| Backend | Method | Notes |
|---------|--------|-------|
| `ceres` | pyceres Levenberg-Marquardt | Default. Analytic Jacobians, SPARSE_NORMAL_CHOLESKY, multi-threaded. Fastest on CPU. |
| `jax` | K: jaxopt LM. G/D/KC/CP: BFGS | Auto CPU/GPU. GPU uses f32, CPU uses f64. JIT-compiled. |
| `scipy` | scipy `least_squares` LM | CPU only. Finite-difference Jacobians. |

No automatic fallbacks. If the requested backend is not installed, the pipeline crashes immediately.

```yaml
solver_backend: ceres   # or jax, scipy
```

## Initial guesses

| Jones | Method |
|-------|--------|
| K | FFT fringe-fitting with 8x zero-padding, BFS propagation from ref_ant |
| G | BFS gain-ratio extraction from parallel hands |
| D | Cross/parallel ratio on baselines to ref_ant |
| KC | Cross-hand phase slope via polyfit on unwrapped angle |
| CP | Mean cross-hand phase (RIME-corrected sign) |

## Solution naming

Each Jones step gets a unique key with a per-type counter starting from 0:

```yaml
jones: [K, G, D, G, G]
# HDF5 keys: K0, G0, D0, G1, G2
```

Use these keys in apply and fluxscale blocks:

```yaml
jones: [K0, G0, D0, G1, G2]
```

## Universal schema

Every solution has the same shape:

```
jones:  (n_ant, n_freq, n_time, 2, 2)  complex128
flags:  (n_ant, n_freq, n_time)         bool
```

`n_freq` and `n_time` are set by the user's `freq_interval` and `time_interval`.
If `freq_interval: full` then `n_freq=1`. If `time_interval: inf` then `n_time=1`.

Only active antennas (those with data in the MS) are stored and processed.
Antenna names are saved in HDF5 attrs for remapping at apply time.

## HDF5 layout

```
cal.h5
├── K0/field_3C286/scan_0/spw_0/
│   ├── jones        (n_ant, n_freq, n_time, 2, 2)  complex128
│   ├── flags        (n_ant, n_freq, n_time)          bool
│   ├── time         (n_time,)  MJD seconds
│   ├── freq         (n_freq,)  Hz
│   └── solver_stats/
│       ├── converged  (n_freq, n_time)  bool
│       ├── n_iter     (n_freq, n_time)  int32
│       └── cost       (n_freq, n_time)  float64
│   attrs: jones_type, field_name, scan_id, spw, n_ant, field_ra, field_dec,
│          matrix_form, ref_ant, ref_ant_name, ant_names, solint_s, freqint_hz,
│          phase_only, feed_basis, solver_backend, apply_parang, ms, preapply_chain
├── G0/field_3C286/scan_0/spw_0/
│   └── ...
└── fluxscale/field_PKS1934/spw_0/
    └── attrs: scale_p, scale_q, scatter_p, scatter_q
```

## Config format

YAML with three top-level blocks: `solve`, `fluxscale`, `apply`. All optional. Each is a list of blocks executed in order.

### Solve block

```yaml
solve:
  - ms: calibrators.ms
    output: cal.h5
    ref_ant: C04              # index or antenna name
    data_col: DATA
    model_col: MODEL_DATA
    solver_backend: ceres     # ceres | jax | scipy
    apply_parang: false       # parallactic angle correction
    rfi_threshold: 5.0        # MAD sigma threshold
    max_iter: 100
    tol: 1.0e-10
    memory_limit_gb: 0        # 0 = auto (40% available RAM)
    gpu: false                # force GPU for JAX

    jones: [K, G, G]

    field:
      - [3C286]              # K: solve on 3C286
      - [3C286, 3C147]       # G0: solve on both
      - [3C286, 3C147]       # G1: solve on both

    time_interval: [scan, inf, 2min]
    freq_interval: [full, 4MHz, full]
    phase_only:    [false, false, false]
    preapply_time_interp: [exact, nearest, nearest]

    # Optional: use solutions from a prior run as preapply
    external_preapply:
      tables:       [oldcal.h5, oldcal.h5]
      jones:        [K0, G0]
      field_select: [nearest_time, nearest_time]
      time_interp:  [nearest, nearest]
```

### Fluxscale block

```yaml
fluxscale:
  - reference_table: cal.h5
    reference_field: [3C286]
    transfer_table: cal.h5
    transfer_field: [3C147]
    output: cal_scaled.h5
    jones_type: G0            # which Jones to scale
    spw: "0"                  # optional SPW selection
```

Bootstrap absolute flux from a known calibrator. Uses iterative sigma-clipping (3sigma, 3 iterations) on per-antenna gain amplitude ratios.

### Apply block

```yaml
apply:
  - ms: science.ms
    output_col: CORRECTED_DATA
    target_field: [B0329+54, DA240]   # omit for all fields
    target_scans: ""                   # CASA scan selection
    apply_parang: true
    propagate_flags: true              # write solution flags to MS FLAG

    jones:        [K0,            G0,            G1           ]
    tables:       [cal.h5,        cal.h5,        cal_scaled.h5]
    field_select: [nearest_time,  nearest_time,  nearest_time ]
    time_interp:  [nearest,       nearest,       linear       ]
    solution_field: [~, ~, ~]    # required when field_select: pinned
```

## Time interval

| Value | Meaning |
|-------|---------|
| `inf` / `infinity` | Entire observation (n_time=1) |
| `scan` | One solution per scan |
| `5min` / `2m` | Minute bins |
| `120s` | Second bins |
| `1h` | Hour bins |

## Frequency interval

| Value | Meaning |
|-------|---------|
| `full` / `inf` | Entire SPW (n_freq=1) |
| `4MHz` | 4 MHz bins |
| `128kHz` | 128 kHz bins |

## SPW selection (CASA syntax)

```
0               # SPW 0, all channels
0:127~156       # SPW 0, channels 127-156
0:10~500,1,2    # SPW 0 ch10-500, SPW 1 all, SPW 2 all
```

## Solve chain

```
raw data
  -> [parallactic angle correction if apply_parang: true]
  -> solve K0 -> store to HDF5
  -> preapply K0
  -> solve G0 -> store to HDF5
  -> preapply K0 + G0
  -> solve G1 -> store to HDF5
  -> ...
```

Each subsequent step automatically preapplies all prior Jones terms. Solutions are interpolated to each cell's centre time. Different solints across steps are handled transparently.

## Apply: field selection and interpolation

Target fields (e.g. science targets) typically have no solutions in the HDF5 — only calibrator fields do. The apply stage interpolates calibrator solutions onto the target field's time/frequency grid.

### Field selection modes

| Mode | Behaviour |
|------|-----------|
| `nearest_time` | For each target timestep, pick the calibrator whose solution times are closest. Default. |
| `nearest_sky` | Pick the calibrator closest on the sky (angular distance). Falls back to `nearest_time` if RA/Dec unavailable. |
| `pinned` | Use exactly the field(s) listed in `solution_field`. No selection logic. |

### Time interpolation

| Mode | Behaviour |
|------|-----------|
| `exact` / `nearest` | Nearest-neighbour in time |
| `linear` | Amp/phase linear interpolation |
| `cubic` | Cubic spline on amp/phase (requires scipy, >= 4 time samples) |

### Frequency interpolation

Automatic based on solution vs target grids:

| Condition | Method |
|-----------|--------|
| n_sol_freq = 1 | Broadcast (same Jones for all channels) |
| Grids match exactly | Pass through |
| Different grids | Nearest-frequency mapping |

### Apply logging

Two log lines per Jones term per target field:

```
K0: loaded [3C286(3t)] (7 ants)
K0: 3C286 -> DA240  (nearest_time, time=nearest 3->45, freq=broadcast 1->2048)
```

First line: what solutions are available. Second line: what was chosen, why, and how the interpolation maps solution grids onto target grids.

### Diagonal optimization

Diagonal Jones (K, G, KC, CP) use optimized numba kernels that only touch `(0,0)` and `(1,1)`. Full 2x2 matrix inversion is used only for D (leakage).

### Active antenna remapping

The MS ANTENNA table may contain antennas with no data (e.g. 32 in header but only 7 with visibilities). Alakazam detects active antennas, remaps all indices, and only processes/stores the active set. At apply time, solutions are remapped by antenna name.

## Flagging

1. MS `FLAG` + `FLAG_ROW` read and merged at load time
2. RFI: MAD threshold on all 4 correlations independently (`rfi_threshold`)
3. Solution flags: NaN, Inf, zero determinant -> stored as `(n_ant, n_freq, n_time)` bool in HDF5
4. Optional flag propagation to MS on apply (`propagate_flags: true`)

## Memory management

Data is loaded per time_bin from the MS using TaQL time-range filtering. Only active antennas are processed.

Three tiers based on available RAM (default limit: 40% of free RAM):
- **Tier 1**: Entire dataset fits in memory -> single read
- **Tier 2**: One time_bin fits -> bin-by-bin reads
- **Tier 3**: One time_bin exceeds RAM -> chunked reads with running accumulator

Raw data stays in native `(n_row, n_chan, n_corr)` format through load -> flag -> average. The 2x2 matrix conversion only happens on the tiny averaged cell data.

## Architecture

```
alakazam/
├── cli.py                CLI entry point (run, info, fluxscale-info)
├── config.py             YAML parser, CASA SPW syntax, dataclasses
├── flow.py               Orchestrator: solve -> fluxscale -> apply
├── jones/
│   ├── algebra.py        2x2 ops, compose, diagonal-optimized apply (numba)
│   ├── constructors.py   Jones builders: gains, delay, leakage, etc. (numba)
│   ├── constructors_ad.py  AD-compatible builders (numpy, for JAX backend)
│   └── parang.py         Parallactic angle computation + Jones rotation
├── solvers/
│   ├── __init__.py       ABC, BFS, backend detection, registry
│   ├── parallel_delay.py K solver (FFT init, LM fit across frequency)
│   ├── gains.py          G solver (BFS init, amp+phase or phase-only)
│   ├── leakage.py        D solver (cross/parallel init, full 2x2)
│   ├── cross_delay.py    KC solver (phase slope init, 1 global param)
│   └── cross_phase.py    CP solver (mean cross-hand phase, 1 global param)
├── core/
│   ├── ms_io.py          Casacore I/O (TaQL queries, metadata, raw format)
│   ├── averaging.py      Baseline averaging (numba, chunked accumulator)
│   ├── interpolation.py  Time/freq interpolation, field selection
│   ├── memory.py         RAM/VRAM detection (psutil)
│   └── rfi.py            MAD flagging
├── io/
│   └── hdf5.py           HDF5 solution read/write, fuzzy key matching
└── calibration/
    ├── apply.py           Interpolate + compose + correct visibilities
    └── fluxscale.py       Sigma-clipped flux bootstrap
```

## Examples

See `examples/` for complete YAML configs:

| File | Description |
|------|-------------|
| `simple_kg.yaml` | Basic delay + gain calibration |
| `bandpass_timegain.yaml` | Delay + bandpass (freq-dependent G) + time gains with parang |
| `full_polarization.yaml` | Full polarization: K + G(bp) + G(time) + D + KC + CP |
| `fluxscale.yaml` | Flux bootstrapping from 3C286 to 3C147 |
| `external_preapply.yaml` | Re-solve gains using prior K+G as external preapply |
| `pinned_field.yaml` | Explicitly map calibrator -> target (pinned field selection) |
| `nearest_sky.yaml` | Pick calibrator closest on the sky for each target |
| `phase_only.yaml` | Phase-only gain calibration (amplitude fixed to 1) |
| `spw_channel_selection.yaml` | CASA-style SPW and channel selection |
| `multi_spw.yaml` | Multi-SPW calibration |
