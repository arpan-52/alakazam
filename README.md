# ALAKAZAM

**A Radio Interferometry Calibrator**
Developed by Arpan Pal, 2026

ALAKAZAM is a direction-independent calibration solver for radio interferometry data using Jones matrix formalism.

## Mathematical Formalism

### Visibility Equation

The measured visibility for baseline between antennas *i* and *j* is modeled as:

```
V_ij = J_i V_ij^true J_j^H
```

where:
- `V_ij` is the observed 2×2 visibility matrix
- `J_i`, `J_j` are 2×2 Jones matrices for antennas *i*, *j*
- `V_ij^true` is the true sky visibility
- `^H` denotes conjugate transpose

### Jones Matrix Decomposition

The total Jones matrix is decomposed as a product:

```
J = K · B · G · D · Xf · Kcross
```

Each term represents a specific instrumental effect. ALAKAZAM solves these terms sequentially.

### Calibration Objective

For each Jones type, solve the nonlinear least-squares problem:

```
min_J  Σ_ij |w_ij (V_ij^obs - J_i V_ij^model J_j^H)|²
```

where `w_ij` are visibility weights and the sum is over all baselines.

### Delay (K)

Parameterization:
```
K_i = [ e^(2πiντ_X)      0          ]
      [      0       e^(2πiντ_Y)    ]
```

where ν is frequency and τ_X, τ_Y are delays for each polarization.

**Solver**: Frequency-domain solver using FFT. Finds delays by peak-finding in delay spectrum.

### Bandpass (B)

Parameterization:
```
B_i(ν) = [ g_X(ν)    0      ]
         [   0     g_Y(ν)   ]
```

where g_X(ν), g_Y(ν) are complex frequency-dependent gains.

**Solver**: Nonlinear least-squares per frequency channel, time-averaged across observation.

### Gain (G)

Parameterization:
```
G_i = [ g_X    0   ]
      [  0    g_Y  ]
```

where g_X, g_Y are complex scalar gains (frequency-averaged).

**Solver**: Nonlinear least-squares, frequency-averaged across band.

### Leakage (D)

Parameterization:
```
D_i = [   1      d_xy ]
      [ d_yx      1   ]
```

where d_xy, d_yx are small complex leakage terms (|d| ≪ 1).

**Solver**: Nonlinear least-squares using all 4 correlation products.

### Crosshand Phase (Xf)

Parameterization:
```
Xf_i = [ 1         0        ]
       [ 0    e^(iϕ_XY)    ]
```

where ϕ_XY is a global phase offset between X and Y.

**Solver**: Nonlinear least-squares using XY and YX correlations. No reference antenna constraint.

### Crosshand Delay (Kcross)

Parameterization:
```
Kcross_i = [ 1              0           ]
           [ 0    e^(2πiντ_XY)          ]
```

where τ_XY is the delay between X and Y polarizations.

**Solver**: Frequency-domain solver using cross-correlation products.

## Solution Intervals (solint)

Solution intervals control time and frequency resolution of calibration solutions.

### Time Intervals

Specify as:
- `"10s"` - 10 seconds
- `"2min"` - 2 minutes
- `"inf"` - entire observation (single solution)

### Frequency Intervals

Specify as:
- `"2MHz"` - 2 MHz chunks
- `"128chan"` - 128 channel chunks
- `"full"` - entire band (single solution)

### Multi-dimensional Solutions

When both time and frequency intervals are specified, ALAKAZAM produces multi-dimensional Jones matrices with shape `(n_time, n_freq, n_ant, 2, 2)`.

Example:
```yaml
solint:
  time_interval: 2min
  freq_interval: 2MHz
```

This produces solutions on a 2D grid: one per 2-minute time block and per 2-MHz frequency chunk.

## Configuration Reference

### Basic Configuration

```yaml
ms_files: "path/to/data.ms"
output_h5: "calibration.h5"

field: "0"              # Field selection (optional)
spw: "0"                # SPW selection (optional)

jones: [K, B, G]        # Jones terms to solve, in order

solint:
  time_interval: inf    # or "10s", "2min", etc.
  freq_interval: full   # or "2MHz", "128chan", etc.

model_col: "MODEL_DATA"
data_col: "DATA"

ref_ant: 0              # Reference antenna index

max_iter: 200           # Maximum iterations for nonlinear solver
tol: 1e-8               # Convergence tolerance

rfi:
  enable: true
  threshold: 5.0        # MAD threshold for flagging
  per_baseline: true    # Flag per baseline (recommended)
```

### Multi-Jones Configuration with Different solint

Each Jones term can have its own solution interval:

```yaml
jones: [K, B, G]

solint:
  time_interval: [inf, inf, 2min]
  freq_interval: [full, 2MHz, full]
```

This solves:
- **K**: Single solution (time=inf, freq=full)
- **B**: Per 2MHz chunk (time=inf, freq=2MHz)
- **G**: Per 2-minute (time=2min, freq=full)

### Jones Chaining

When solving multiple Jones terms, previously solved terms are applied before solving the next:

```yaml
jones: [G, B]
```

**Solving process:**
1. Solve G: `min_G Σ |V^obs - G_i V^model G_j^H|²`
2. Solve B with G applied:
   - Corrected visibility: `V' = G_i^{-1} V^obs G_j^{-H}`
   - Solve: `min_B Σ |V' - B_i V^model B_j^H|²`

## Usage Examples

### Example 1: Basic Gain Calibration

Solve for frequency-averaged complex gains:

```yaml
# config_gain.yaml
ms_files: "calibrator.ms"
output_h5: "gain.h5"

field: "0"
jones: [G]

solint:
  time_interval: 2min
  freq_interval: full

ref_ant: 0
max_iter: 200
```

Run:
```bash
alakazam solve config_gain.yaml
```

Output: `gain.h5` containing Jones matrix with shape `(n_time, n_ant, 2, 2)`.

### Example 2: Delay + Bandpass Calibration

Solve delay and bandpass sequentially:

```yaml
# config_kb.yaml
ms_files: "calibrator.ms"
output_h5: "kb_solutions.h5"

jones: [K, B]

solint:
  time_interval: inf
  freq_interval: [full, 2MHz]

ref_ant: 0
```

Run:
```bash
alakazam solve config_kb.yaml
```

**Solving process:**
1. Solve K on full band → shape `(n_freq, n_ant, 2, 2)`
2. Apply K, then solve B per 2MHz → shape `(n_freq_chunks, n_ant, 2, 2)`

### Example 3: Full Polarization Calibration

Solve all polarization terms:

```yaml
# config_full_pol.yaml
ms_files: "calibrator.ms"
output_h5: "full_pol.h5"

jones: [Kcross, Xf, D, G]

solint:
  time_interval: [inf, inf, inf, 2min]
  freq_interval: full

ref_ant: 0
```

**Solving order:**
1. Kcross: Crosshand delay
2. Xf: Crosshand phase
3. D: Leakage
4. G: Parallel-hand gains

### Example 4: Multi-SPW Calibration

ALAKAZAM automatically detects and processes multiple spectral windows:

```yaml
# config_multi_spw.yaml
ms_files: "multi_spw.ms"
output_h5: "solutions_spw{spw}.h5"

spw: "*"                # Process all SPWs
jones: [K, B, G]

solint:
  time_interval: inf
  freq_interval: [full, 1MHz, full]

ref_ant: 0
```

Output: Separate H5 files per SPW: `solutions_spw0.h5`, `solutions_spw1.h5`, etc.

### Example 5: Applying Calibration

After solving, apply corrections to target field:

```yaml
# config_apply.yaml
ms_files: "target.ms"

apply:
  tables: ["gain.h5", "bandpass.h5"]
  jones: [G, B]
  output_col: "CORRECTED_DATA"
```

Run:
```bash
alakazam apply config_apply.yaml
```

**Application process:**
1. Load G and B from H5 files
2. Interpolate to native MS time and frequency grids
3. Composite: `J_total = B @ G`
4. Apply: `V_corrected = J_total^{-1} @ V_obs @ J_total^{-H}`
5. Write to `CORRECTED_DATA` column

### Example 6: Interpolation During Application

Jones solutions are interpolated to match target MS grids:

**Scenario**: Solved G with `time_interval: 10min`, applying to MS with 10s integrations.

```yaml
# Solve on calibrator
# config_solve.yaml
ms_files: "calibrator.ms"
output_h5: "gain_10min.h5"
jones: [G]
solint:
  time_interval: 10min
  freq_interval: full
```

After solving, `gain_10min.h5` contains shape `(6, n_ant, 2, 2)` for 1-hour observation.

```yaml
# Apply to target
# config_apply_target.yaml
ms_files: "target.ms"  # 10s integrations
apply:
  tables: ["gain_10min.h5"]
  jones: [G]
  output_col: "CORRECTED_DATA"
```

**Interpolation:**
- Time: Linear interpolation from 6 samples (10min) → 360 samples (10s)
- Frequency: Broadcast single solution → all channels

Result: Smooth gain corrections applied at 10s resolution.

## Advanced Configuration Options

### RFI Flagging

```yaml
rfi:
  enable: true
  threshold: 5.0          # MAD threshold
  per_baseline: true      # Flag per baseline (recommended)
  algorithm: "MAD"        # Median Absolute Deviation
```

**Algorithm**: For each baseline and chunk, compute:
```
MAD = median(|V - median(V)|)
σ = 1.4826 × MAD
```
Flag visibilities where `|V - median(V)| > threshold × σ`.

### Solver Convergence

```yaml
max_iter: 200            # Maximum iterations
tol: 1e-8                # Convergence tolerance on cost function
ftol: 1e-10              # Function tolerance (optional)
gtol: 1e-8               # Gradient tolerance (optional)
```

Solver converges when:
```
|Δcost| / cost < tol
```

### Phase-only Gain Solving

For stable amplitude calibrators:

```yaml
jones: [G]
phase_only: true
```

This constrains `|g_X| = |g_Y| = 1`, solving only for phases.

### Parallel Processing

ALAKAZAM automatically parallelizes across:
- Time chunks (when `time_interval` is specified)
- Frequency chunks (when `freq_interval` is chunked)
- SPWs (when multiple spectral windows present)

### Logging and Output

```yaml
verbose: true            # Rich progress bars and statistics
log_file: "alakazam.log" # Log file path (optional)
```

**Output includes:**
- Progress bars for multi-chunk solving
- Fit quality statistics (cost reduction)
- Gain statistics (median amplitude, phase scatter)
- Reference antenna gains
- Convergence information
- Jones chain visualization

## Output Format

Solutions are stored in HDF5 files with the following structure:

```
/
├── jones               # Jones matrix dataset (n_time, n_freq, n_ant, 2, 2)
├── time                # Time samples (MJD seconds)
├── freq                # Frequency samples (Hz)
├── antenna_names       # Antenna names
├── feed_basis          # Feed basis ('linear' or 'circular')
└── metadata/
    ├── jones_type      # 'K', 'B', 'G', etc.
    ├── ref_ant         # Reference antenna index
    ├── time_interval   # Solution time interval
    └── freq_interval   # Solution frequency interval
```

## Installation

```bash
pip install -e .
```

Requirements:
- Python ≥ 3.8
- NumPy, SciPy
- python-casacore (MS access)
- h5py (HDF5 I/O)
- PyYAML (config parsing)
- numba (JIT compilation)
- rich (terminal output, optional)

## Command-line Interface

### Solve

```bash
alakazam solve config.yaml
```

### Apply

```bash
alakazam apply config.yaml
```

### List Solutions

```bash
alakazam list solutions.h5
```

Output:
```
Jones terms in solutions.h5:
  - G: shape (155, 64, 2, 2), time_interval=2min, freq_interval=full
  - B: shape (32, 64, 2, 2), time_interval=inf, freq_interval=2MHz
```

## Best Practices

### Calibration Strategy

1. **Delay calibration first**: Solve K on strong calibrator to remove group delays
2. **Bandpass**: Solve B on bright source with good frequency coverage
3. **Gain calibration**: Solve G with short intervals on phase calibrator
4. **Polarization**: Solve D, Xf, Kcross on polarized calibrator

### Solution Interval Selection

- **Delay (K)**: `time_interval: inf` (stable over observation)
- **Bandpass (B)**: `time_interval: inf`, `freq_interval: 1-4MHz` (stable in time, varies with frequency)
- **Gain (G)**: `time_interval: 1-5min`, `freq_interval: full` (varies with time, achromatic)
- **Leakage (D)**: `time_interval: inf`, `freq_interval: full` (stable)

### Reference Antenna Selection

Choose reference antenna that:
- Has low noise
- Is centrally located
- Has good coverage to other antennas
- Is not flagged

### Convergence Issues

If solver fails to converge:
- Increase `max_iter`
- Loosen `tol`
- Check for bad data (high RFI)
- Verify model quality
- Try different reference antenna

## Citation

If you use ALAKAZAM, please cite:

```
ALAKAZAM: A Radio Interferometry Calibrator
Arpan Pal, 2026
```

## License

[Specify license]
