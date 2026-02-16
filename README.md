# ALAKAZAM

**A Radio Interferometric Calibration Suite for Arrays**

*Developed by Arpan Pal 2026, NRAO / NCRA*

---

## Overview

ALAKAZAM is a fast, memory-safe radio interferometry calibration package. It solves for antenna-based Jones matrices using the Radio Interferometer Measurement Equation (RIME):

```
V_obs = J_i · V_model · J_j^H
```

ALAKAZAM supports 5 Jones solver types, both LINEAR and CIRCULAR feed bases, parallactic angle correction, multi-SPW processing, and sequential calibration chains — all through a single YAML configuration file.

## Features

- **5 Jones solvers**: K (delay), G (gain/bandpass), D (leakage), Xf (cross-phase), Kcross (cross-delay)
- **Unified G solver**: G serves as both gain and bandpass — behavior controlled by `freq_interval`
- **Chain initial guess** for all solvers using BFS propagation from reference antenna
- **Levenberg-Marquardt optimization** via `scipy.optimize.least_squares`
- **Both LINEAR and CIRCULAR feeds** with automatic detection
- **Parallactic angle correction** in both solving and applying
- **Multi-SPW support**: process all SPWs in a single run, stored in one HDF5 file
- **Memory-safe**: chunked MS loading, memory prediction before allocation
- **RFI flagging**: MAD-based per-baseline outlier detection (numba JIT)
- **Proper interpolation**: amplitude/phase (not real/imag) for diagonal Jones
- **Rich console output**: progress bars, tables, colored logging
- **HDF5 solutions**: always 5D `(n_time, n_freq, n_ant, 2, 2)` with full metadata
- **Error estimation** from Jacobian/Hessian after optimization
- **Quality metrics**: SNR, χ², RMSE, R² per solution cell
- **Flux scale bootstrapping**

## Installation

```bash
pip install .
```

### Dependencies

- Python ≥ 3.9
- numpy, scipy, numba, h5py, pyyaml
- python-casacore ≥ 3.5
- rich (for console output)
- psutil (optional, for RAM monitoring)

## Quick Start

### 1. Create a YAML config

```yaml
# calibration.yaml
ms: observation.ms
output: solutions.h5
field: 3C286
spw: "*"
ref_ant: 0

jones: [K, G, G]
time_interval: [inf, inf, 2min]
freq_interval: [full, 4MHz, full]
```

### 2. Run

```bash
alakazam run calibration.yaml
```

### 3. Apply

```bash
alakazam apply target.ms solutions.h5
```

### 4. Inspect

```bash
alakazam info solutions.h5
```

## Configuration Reference

### Top-level fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ms` | str | *required* | Path to Measurement Set |
| `output` | str | `calibration.h5` | Output HDF5 file |
| `jones` | list/str | *required* | Jones types to solve: K, G, D, Xf, Kcross |
| `field` | str | all | Field name selection |
| `spw` | str | `*` | SPW selection: `*`, `0`, `0~3`, `0,2,4` |
| `scans` | str | all | Scan selection |
| `ref_ant` | int/list | 0 | Reference antenna (per-Jones or global) |
| `data_col` | str | `DATA` | Data column |
| `model_col` | str | `MODEL_DATA` | Model column |
| `apply_parang` | bool | true | Apply parallactic angle correction |
| `memory_limit_gb` | float | 0 (auto) | Memory limit in GB (0 = 60% of available) |

### Per-Jones fields

These can be a single value (applied to all Jones) or a list matching the length of `jones`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `time_interval` | str/list | `inf` | Time solint: `inf`, `30s`, `2min`, `0.5h` |
| `freq_interval` | str/list | `full` | Freq solint: `full`, `spw`, `4MHz`, `128chan` |
| `phase_only` | bool/list | false | Phase-only mode (G solver) |
| `rfi_threshold` | float/list | 5.0 | MAD sigma for RFI flagging (0 = disable) |
| `max_iter` | int/list | 100 | Max LM iterations |
| `tol` | float/list | 1e-10 | Convergence tolerance |

### Solint behavior by Jones type

| Config | G behavior |
|--------|-----------|
| `freq_interval: full` | Traditional gain (one value, all freqs averaged) |
| `freq_interval: 4MHz` | Bandpass (one value per 4 MHz chunk) |
| `freq_interval: spw` | Per-SPW (one value per spectral window) |
| `freq_interval: 128chan` | Per 128 channels |

K and Kcross always keep the frequency axis (never averaged) for delay fitting.

## Jones Types

### K — Antenna Delay
```
K(ν) = diag(exp(-2πi τ_p ν), exp(-2πi τ_q ν))
```
Fits delay in nanoseconds from phase slope vs frequency. Uses parallel-hand correlations.

### G — Complex Diagonal Gain
```
G = diag(g_p, g_q) = diag(A_p exp(iφ_p), A_q exp(iφ_q))
```
Solves for complex gain per polarization. Also serves as bandpass via `freq_interval`. Uses parallel-hand correlations.

### D — Polarization Leakage
```
D = [[1, d_pq], [d_qp, 1]]
```
Solves for instrumental polarization leakage. Uses all four correlations. Chain initial guess uses cross-hand data.

### Xf — Cross-hand Phase
```
Xf = diag(1, exp(iφ_pq))
```
Standard diagonal form. Solves for constant phase offset between polarization paths. Uses cross-hand correlations.

### Kcross — Cross-hand Delay
```
Kcross(ν) = diag(1, exp(-2πi τ_pq ν))
```
Frequency-dependent cross-hand delay. Uses cross-hand correlations.

## HDF5 Solution Format

```
solution.h5
├── K/
│   ├── spw_0/
│   │   ├── jones      (n_t, n_f, n_ant, 2, 2)  complex128
│   │   ├── time       (n_t,)
│   │   ├── freq       (n_f,)
│   │   ├── freq_full  (n_chan,)
│   │   ├── flags      (n_t, n_f, n_ant)
│   │   ├── weights    (n_t, n_f)
│   │   ├── params/delay  (n_t, n_f, n_ant, 2)
│   │   └── quality/chi2_red, snr, rmse, ...
│   ├── spw_1/ ...
│   └── metadata/
├── G/ ...
├── observation/
│   ├── ms_path, antenna_names, working_antennas, feed_basis
├── config/
│   └── yaml_content (full config for reproducibility)
└── creation_time, alakazam_version
```

## Python API

```python
import alakazam

# Run pipeline
config = alakazam.AlakazamConfig(
    ms_path='obs.ms',
    output='cal.h5',
    steps=[
        alakazam.SolveStep(jones_type='K'),
        alakazam.SolveStep(jones_type='G', freq_interval='4MHz'),
        alakazam.SolveStep(jones_type='G', time_interval='2min'),
    ],
)
solutions = alakazam.run_pipeline(config)

# Load solutions
sols = alakazam.load_solutions('cal.h5')
k_jones = sols['K'][0]['jones']       # (1, 1, n_ant, 2, 2)
k_delays = sols['K'][0]['params']['delay']

# Apply
alakazam.apply_calibration('target.ms', 'cal.h5')

# Flux bootstrapping
from alakazam.fluxscale import compute_fluxscale
scale, info = compute_fluxscale(g_cal, g_target, working_ants, ref_ant)

# Jones algebra
from alakazam.jones import delay_to_jones, jones_multiply
K = delay_to_jones(delays, freq)
G = alakazam.gain_to_jones(amp, phase, n_ant)
J_total = alakazam.compose_jones_chain([K[:, freq_mid], G])
```

## Architecture

```
alakazam/
├── __init__.py          Public API
├── config.py            YAML parsing + validation
├── pipeline.py          Single pipeline orchestrator
├── jones.py             All 2×2 Jones algebra (numba)
├── apply.py             Vectorized calibration apply
├── fluxscale.py         Flux bootstrapping
├── cli.py               CLI: run/apply/info/version
├── core/
│   ├── ms_io.py         Chunked MS reading
│   ├── averaging.py     Flag-aware averaging (numba)
│   ├── rfi.py           MAD-based RFI flagging (numba)
│   ├── interpolation.py Amp/phase interpolation
│   ├── quality.py       χ², SNR, RMSE metrics
│   └── memory.py        Memory prediction + safe chunking
├── solvers/
│   ├── __init__.py      Base class + BFS utilities
│   ├── registry.py      Solver registry
│   ├── k_delay.py       K solver
│   ├── g_gain.py        G solver (also bandpass)
│   ├── d_leakage.py     D solver
│   ├── xf_crossphase.py Xf solver
│   └── kcross_delay.py  Kcross solver
└── io/
    └── hdf5.py          Unified HDF5 I/O
```

## License

MIT

## Citation

If you use ALAKAZAM in your research, please cite:
```
Pal, A. 2026, ALAKAZAM: A Radio Interferometric Calibration Suite for Arrays
```
