# ALAKAZAM v1.0.0

Radio interferometric calibration pipeline.

Arpan Pal — NRAO / NCRA, 2026

## Install

```bash
pip install .

# With GPU (JAX + CUDA):
pip install .[all]
```

Requires: Python ≥ 3.9, python-casacore, numpy, scipy, numba, h5py, pyyaml, jax.

## Quick start

```bash
alakazam run config.yaml
alakazam info cal.h5
```

## Jones types

| Key | Name | Matrix | Constraint |
|-----|------|--------|------------|
| K | Parallel delay | `diag(e^{-2πiτ_p ν}, e^{-2πiτ_q ν})` | `τ[ref,:]=0` |
| G | Gains | `diag(g_p e^{iφ_p}, g_q e^{iφ_q})` | `φ[ref,:]=0`, amp free |
| D | Leakage | `[[1, d_pq], [d_qp, 1]]` | `d_pq[ref]=0`, `d_qp[ref]` free |
| KC | Cross delay | `diag(e^{-2πiτν}, 1)` global | 1 param |
| CP | Cross phase | `diag(1, e^{iφ})` global | 1 param |

## Solution naming

Each Jones step gets a unique key with a per-type counter starting from 0:

```yaml
jones: [K, G, D, G, G]
# HDF5 keys: K0, G0, D0, G1, G2
```

Use these keys in apply:

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

Only active antennas (those with data) are stored. Antenna names are in metadata.

## HDF5 layout

```
cal.h5
├── K0/field_3C286/spw_0/
│   ├── jones        (n_ant, n_freq, n_time, 2, 2)
│   ├── flags        (n_ant, n_freq, n_time)
│   ├── time         (n_time,)
│   ├── freq         (n_freq,)
│   ├── solver_stats/
│   │   ├── converged  (n_freq, n_time)
│   │   ├── n_iter     (n_freq, n_time)
│   │   └── cost       (n_freq, n_time)
│   └── attrs: jones_type, field_name, ref_ant, ref_ant_name,
│              ant_names, n_active, solint_s, freqint_hz,
│              solver_backend, ms, preapply_chain, ...
├── G0/field_3C286/spw_0/
│   └── ...
├── G1/field_3C286/spw_0/
│   └── ...
├── native_params/K0/field_3C286/spw_0/
│   └── delay  (n_ant, n_freq, n_time, 2)
└── fluxscale/field_PKS1934/spw_0/
    └── attrs: scale_p, scale_q, scatter_p, scatter_q
```

## Config format

YAML with three blocks: `solve`, `fluxscale`, `apply`.
Every per-step parameter can be a scalar (same for all steps) or a list (one per step).

### Time interval

| Value | Meaning |
|-------|---------|
| `inf` | Entire observation |
| `scan` | One solution per scan |
| `5min` | 5-minute bins |
| `120s` | 120-second bins |

### Frequency interval

| Value | Meaning |
|-------|---------|
| `full` | Entire SPW (n_freq=1) |
| `4MHz` | 4 MHz bins |

### SPW selection (CASA syntax)

```
0               # SPW 0, all channels
0:127~156       # SPW 0, channels 127-156
0:10~500,1,2    # SPW 0 ch10-500, SPW 1 all, SPW 2 all
```

### ref_ant

Accepts integer index or antenna name:

```yaml
ref_ant: 0        # by index
ref_ant: C04      # by name
```

## Solve chain

```
raw data
  → [parallactic angle correction if apply_parang: true]
  → solve K0 → store
  → preapply K0
  → solve G0 → store
  → preapply K0 + G0
  → solve G1 → store
```

Different solints across steps are handled transparently. Preapply Jones are
interpolated to each cell's centre time.

## Memory management

Data is loaded per time_bin from the MS using TaQL time-range filtering.
Only active antennas are processed.

Three tiers based on available RAM:
- **Tier 1/2**: One time_bin fits in RAM → load, flag, average, free, solve
- **Tier 3**: One time_bin exceeds RAM → chunked reads with running accumulator

Raw data stays in native `(n_row, n_chan, n_corr)` format through load → flag → average.
The 2×2 matrix conversion only happens on the tiny averaged cell data.

## Solver backends

| Backend | Method | GPU |
|---------|--------|-----|
| `jax_scipy` | JAX exact gradients + scipy L-BFGS-B | auto |
| `torch_lbfgs` | PyTorch LBFGS | auto |

Falls back to scipy Levenberg-Marquardt if JAX unavailable.

## Initial guesses

| Jones | Method |
|-------|--------|
| K | BFS phase-slope propagation from ref_ant |
| G | BFS gain-ratio extraction from parallel hands |
| D | Cross/parallel ratio on baselines to ref_ant |
| KC | Cross-hand phase slope across frequency |
| CP | Mean cross-hand phase (RIME-corrected sign) |

## Flagging

1. MS `FLAG` + `FLAG_ROW` read and merged
2. RFI: MAD threshold on all 4 correlations independently
3. Solution flags: NaN, Inf, zero determinant → `(n_ant, n_freq, n_time)` stored in HDF5
4. Optional flag propagation to MS on apply (`propagate_flags: true`)

## Apply

Diagonal Jones (K, G, KC, CP) use optimized kernels that only touch `(0,0)` and `(1,1)`.
Full 2×2 inversion for D (leakage).

If a stored solution has `n_freq=1`, the same Jones is applied to every channel.
If `n_freq > 1`, per-channel correction.

## Fluxscale

Bootstraps flux density from a known calibrator. Iterative sigma-clipping
(3σ, 3 iterations) to reject outlier antennas.

## Architecture

```
alakazam/
├── cli.py              CLI entry point
├── config.py           YAML parser, CASA SPW syntax
├── flow.py             Orchestrator: solve → fluxscale → apply
├── jones/
│   ├── algebra.py      2×2 ops, diagonal-optimized apply (numba)
│   ├── constructors.py Jones builders (numba)
│   ├── constructors_ad.py  AD-compatible (numpy, for JAX)
│   └── parang.py       Parallactic angle
├── solvers/
│   ├── __init__.py     ABC, BFS, backend wrappers, initial guesses
│   ├── parallel_delay.py  K solver
│   ├── gains.py           G solver
│   ├── leakage.py         D solver
│   ├── cross_delay.py     KC solver
│   └── cross_phase.py     CP solver
├── core/
│   ├── ms_io.py        Casacore I/O (raw format, no 2×2)
│   ├── averaging.py    Baseline averaging (numba)
│   ├── interpolation.py Time/freq interpolation
│   ├── memory.py       RAM/VRAM detection
│   └── rfi.py          MAD flagging
├── io/
│   └── hdf5.py         Solution read/write
└── calibration/
    ├── apply.py        Interpolate + compose + correct
    └── fluxscale.py    Sigma-clipped flux bootstrap
```
