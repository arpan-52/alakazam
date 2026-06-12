# ALAKAZAM v1.0.0

Radio interferometric calibration pipeline for CASA Measurement Sets.

Arpan Pal — NRAO / NCRA, 2026

## Features

- **5 Jones solvers**: K (parallel delay), G (gains), D (leakage), KC (cross delay), CP (cross phase)
- **Kokkos backend**: High-performance Levenberg-Marquardt via `boa` (CPU/OpenMP or GPU/CUDA)
- **Feed-basis aware**: LINEAR (XX/XY/YX/YY) and CIRCULAR (RR/RL/LR/LL) feeds
- **Interpolation**: Time modes (exact/nearest/linear/cubic), field selection (nearest_time/nearest_sky/pinned)
- **Memory-efficient**: Scan-level processing with configurable batching
- **HDF5 output**: Structured solution tables with provenance metadata

## Installation

### Option 1: pixi (CPU/OpenMP — recommended for most users)

This uses conda-forge packages and builds boa with OpenMP parallelization.

```bash
# Install pixi if you don't have it
curl -fsSL https://pixi.sh/install.sh | bash

# Clone and setup
git clone https://github.com/arpan-pal/alakazam.git
cd alakazam
pixi run setup
```

This builds the boa Kokkos solver and installs alakazam in one command.

### Option 2: Custom Kokkos (GPU/CUDA or advanced users)

For GPU support or custom Kokkos builds:

```bash
# 1. Build Kokkos and KokkosKernels with your desired backend
#    Example for CUDA:
#    cmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON ...
#    cmake -DKokkosKernels_ENABLE_TPL_CUBLAS=ON ...

# 2. Build the boa solver extension
cd alakazam/boa
cmake -B build -DKokkos_ROOT=/path/to/kokkos -DKokkosKernels_ROOT=/path/to/kokkos-kernels
cmake --build build -j

# 3. Install the Python package
cd ../..
pip install -e .
```

**Note:** install must be editable (`pip install -e .`) — the boa extension
(`_boa.so`) is loaded from the source tree (`alakazam/boa/build/bindings/`).

### Dependencies

- Python >= 3.9
- numpy, scipy, h5py, pyyaml, numba, psutil, rich
- python-casacore >= 3.5
- Kokkos + KokkosKernels (built with `BUILD_SHARED_LIBS=ON`)

## Quick start

```bash
alakazam run config.yaml         # run solve -> fluxscale -> apply
alakazam info cal.h5             # print solution summary
alakazam fluxscale-info cal.h5   # print fluxscale factors
```

## Solver backend

All solvers use the `boa` Kokkos LM backend:

- **CPU**: OpenMP parallelization (default with pixi/conda-forge)
- **GPU**: CUDA/HIP support (requires custom Kokkos build)

Query the execution space:

```python
from alakazam.solvers.boa_backend import import_boa
boa = import_boa()
print(boa.execution_space())  # e.g., "OpenMP", "Cuda", "Serial"
```

## Jones types

| Key | Name | Matrix | Constraint | Freq-dependent |
|-----|------|--------|------------|----------------|
| K | Parallel delay | `diag(e^{-2πiτ_p ν}, e^{-2πiτ_q ν})` | `τ[ref,:]=0` | Fits across freq |
| G | Complex gains | `diag(g_p e^{iφ_p}, g_q e^{iφ_q})` | `φ[ref,:]=0`, all amps free (incl. ref) | Per freq bin |

G has two modes: the default solves amplitudes and phases (ref amp free —
gains can absorb a global flux factor for the unit-model fluxscale
convention); `phase_only: true` solves only phases, with all amplitudes
fixed at 1 inside the solver (they are not parameters).

| D | Leakage | `[[1, d_pq], [d_qp, 1]]` | `d_pq[ref]=0`, `d_qp[ref]` free | Per freq bin |
| KC | Cross-hand delay | `diag(e^{-2πiτ ν}, 1)` global | 1 parameter | Fits across freq |
| CP | Cross-hand phase | `diag(1, e^{iφ})` global | 1 parameter | Per freq bin |

## Initial guesses

| Jones | Method |
|-------|--------|
| K | FFT fringe-fitting with 8x zero-padding, BFS propagation from ref_ant |
| G | BFS gain-ratio extraction from parallel hands |
| D | Cross/parallel ratio on baselines to ref_ant |
| KC | Cross-hand phase slope via polyfit on unwrapped angle |
| CP | Mean cross-hand phase (RIME-corrected sign) |

## Config format

YAML with three blocks: `solve`, `fluxscale`, `apply`. See `examples/` for complete configs.

### Solve block

```yaml
solve:
  - ms: calibrators.ms
    output: cal.h5
    ref_ant: C04              # index or antenna name
    data_col: DATA
    model_col: MODEL_DATA
    apply_parang: false
    rfi_threshold: 5.0
    max_iter: 100
    tol: 1.0e-10
    solver_backend: boa       # only boa is supported

    jones: [K, G, D]
    field: [[3C286], [3C286, 3C147], [3C286]]
    time_interval: [scan, inf, inf]
    freq_interval: [full, 4MHz, full]
    phase_only: [false, false, false]
```

### Fluxscale block

```yaml
fluxscale:
  - reference_table: cal.h5
    reference_field: [3C286]
    transfer_table: cal.h5
    transfer_field: [3C147]
    output: cal.h5
    jones_type: G0
```

### Apply block

```yaml
apply:
  - ms: science.ms
    output_col: CORRECTED_DATA
    target_field: [B0329+54]
    apply_parang: true
    propagate_flags: true

    jones: [K0, G0, D0]
    tables: [cal.h5, cal.h5, cal.h5]
    field_select: [nearest_time, nearest_time, nearest_time]
    time_interp: [nearest, linear, nearest]
```

**Order matters**: list `jones` terms in the order they were solved
(first-solved first). Each solution describes the residual after the terms
before it were removed, so the applied chain is composed in the same order.
For purely diagonal chains (K, G, KC, CP) order is immaterial; once D or
parang is in the chain it is not.

## Time interval

| Value | Meaning |
|-------|---------|
| `inf` | Entire observation (n_time=1) |
| `scan` | One solution per scan |
| `5min` / `120s` | Time bins |

## Frequency interval

| Value | Meaning |
|-------|---------|
| `full` | Entire SPW (n_freq=1) |
| `4MHz` | Frequency bins |

When KC (cross-hand delay) is in the chain, solve D with a finite
`freq_interval` (e.g. `4MHz`) rather than `full`: the effective leakage
rotates with frequency under the cross-hand delay, so a full-band D
average decorrelates by ~`|d|·sin(π τ_c B)`.

## Solution naming

Each Jones step gets a unique key with a per-type counter starting from 0:

```yaml
jones: [K, G, D, G, G]
# HDF5 keys: K0, G0, D0, G1, G2
```

## HDF5 layout

```
cal.h5
├── K0/field_3C286/scan_0/spw_0/
│   ├── jones, delay, flags, time, freq
│   └── solver_stats/{converged, n_iter, cost}
├── G0/field_3C286/scan_0/spw_0/
│   └── ...
└── fluxscale/field_PKS1934/spw_0/
    └── attrs: scale_p, scale_q, scatter_p, scatter_q
```

## Universal schema

```
jones:  (n_ant, n_freq, n_time, 2, 2)  complex128
flags:  (n_ant, n_freq, n_time)         bool
delay:  (n_ant, n_freq, n_time, 2)      float64 ns  [K/KC only]
```

## Architecture

```
alakazam/
├── cli.py           # Command-line interface
├── config.py        # YAML config parser
├── flow.py          # Pipeline orchestrator
├── solvers/         # Jones solvers (boa backend)
│   ├── gains.py, parallel_delay.py, leakage.py
│   ├── cross_delay.py, cross_phase.py
│   └── boa_backend.py     # Kokkos interface
├── jones/           # Jones matrix algebra
│   ├── algebra.py, constructors.py, parang.py
├── calibration/     # Apply and fluxscale
├── core/            # MS I/O, averaging, interpolation
├── boa/             # Kokkos C++ solver (CMake project)
└── io/              # HDF5 solution I/O
```

## Tests

```bash
# boa C++ unit tests (after building)
ctest --test-dir alakazam/boa/build

# End-to-end chain tests on synthetic data: builds a small MS with a known
# Jones chain, runs solve -> fluxscale -> apply, asserts recovery vs truth.
python tests/test_chain.py
```

## Examples

See the `examples/` directory for complete config files:

- `simple_kg.yaml` — Basic delay + gain calibration
- `full_polarization.yaml` — Full polarization (K, G, D, KC, CP)
- `fluxscale.yaml` — Fluxscale transfer example
- `meerkat_fullpol.yaml` — MeerKAT full polarization workflow
- `bandpass_timegain.yaml` — Bandpass + time-dependent gains
- `phase_only.yaml` — Phase-only calibration
- `nearest_sky.yaml` — Nearest-sky field selection
- `pinned_field.yaml` — Pinned field selection

## License

MIT
