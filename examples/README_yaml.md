# JACKAL YAML Configuration

JACKAL supports multi-Jones sequential calibration via YAML configuration files.

## Key Features

- **Multi-Jones Sequential Solving**: Solve `[K, G, B]` in one block with automatic pre-apply chaining
- **Smart Parameter Expansion**: Single value broadcasts to all Jones, or provide list per-Jones
- **Automatic Pre-apply Chaining**: G automatically gets K, B gets K+G
- **External Pre-apply**: Apply existing calibration before starting the chain
- **Field Comma Syntax**: `field: "3C147,3C286"` solves for both fields
- **Solver-specific Parameters**: `phase_only` for G/B, `d_constraint` for D
- **Command-Line Interface**: Run calibration with `jackal run config.yaml`

## Installation

Install JACKAL to get the `jackal` command:

```bash
# Install from source (development mode)
pip install -e .

# Or install from PyPI (when available)
pip install jackal
```

## Quick Start

### 1. Create a YAML config file

```yaml
# sequential_cal.yaml
solve:
  - jones: [K, G, B]              # Sequential solving
    ms: observation.ms
    output: calibration.h5

    # Antenna & Feed
    ref_ant: 4
    feed_basis: linear

    # MS Selection
    field: 3C286
    spw: '0'
    scans: 1~10

    # Solution Intervals (per-Jones)
    time_interval: [inf, 60s, 60s]     # K: inf, G: 60s, B: 60s
    freq_interval: [4MHz, full, 1MHz]   # K: 4MHz, G: full, B: 1MHz

    # Columns
    data_col: DATA
    model_col: MODEL_DATA

    # RFI Flagging
    rfi_enable: true
    rfi_threshold: 5.0

    # Solver Parameters
    max_iter: 100
    tol: 1.0e-10

    # Solver-specific
    phase_only: [false, false, false]

    # Pre-apply (empty for first calibration)
    pre_apply_file: []
    pre_apply_jones: []
```

### 2. Run calibration

**Command-Line (recommended):**

```bash
# Run calibration from YAML config
jackal run sequential_cal.yaml

# Show solution information
jackal info calibration.h5

# Show version
jackal version
```

**Python API:**

```python
from jackal import solve_from_yaml

# Solve from YAML (automatic sequential solving with pre-apply chain)
results = solve_from_yaml('sequential_cal.yaml')
# Output written to calibration.h5 (contains K, G, and B solutions)
```

## YAML Format Reference

### Required Parameters

```yaml
solve:
  - jones: [K, G, B]              # Jones types to solve (sequential)
    ms: observation.ms            # Measurement set path
    output: calibration.h5        # Output HDF5 file
```

### Solution Intervals

```yaml
time_interval: inf                # Single value -> all Jones
# OR
time_interval: [inf, 60s, 60s]    # Per-Jones list

freq_interval: full               # Single value -> all Jones
# OR
freq_interval: [4MHz, full, 1MHz] # Per-Jones list
```

Time formats: `'60s'`, `'5min'`, `'inf'` (entire observation)
Frequency formats: `'1MHz'`, `'4MHz'`, `'full'` (all channels)

### Antenna & Feed

```yaml
ref_ant: 4                        # Reference antenna index
feed_basis: linear                # 'linear', 'circular', or 'auto'
```

### MS Selection (all optional)

```yaml
field: 3C286                      # Field name
# OR
field: 3C147,3C286                # Comma syntax: solve for both

# SPW selection (all formats supported)
spw: '0'                          # Single SPW
spw: '0~2'                        # SPW range (0, 1, 2)
spw: '0,2,4'                      # Multiple SPWs (0, 2, 4)
spw: '0:250~1800'                 # SPW 0, channels 250-1800
spw: '1:300~1200'                 # SPW 1, channels 300-1200
spw: 0                            # Integer (auto-converted to string)

# Scan selection (all formats supported)
scans: 10                         # Single scan (integer)
scans: '10'                       # Single scan (string)
scans: '10~20'                    # Scan range (10 through 20)
scans: '10,15,20'                 # Multiple scans (10, 15, 20)
```

**SPW:channel syntax** is useful for delay calibration when you want to exclude edge channels:
```yaml
jones: [K]
spw: '0:250~1800'                 # Exclude first 250 and last ~200 channels
```

### Column Selection

```yaml
data_col: DATA                    # Input data column
model_col: MODEL_DATA             # Model column to solve against
```

### RFI Flagging

```yaml
rfi_enable: true                  # Enable RFI flagging
rfi_threshold: 5.0                # MAD threshold in sigma
```

### Solver Parameters

```yaml
max_iter: 100                     # Max iterations
tol: 1.0e-10                      # Convergence tolerance
```

### Solver-Specific Options

#### G/B Gain (phase_only)

```yaml
jones: [G, B]
phase_only: false                 # Single value -> all Jones
# OR
phase_only: [true, false]         # G: phase-only, B: amp+phase
```

#### D Leakage (d_constraint)

```yaml
jones: [D]
d_constraint: XY                  # 'XY', 'YX', or 'both'
```

### External Pre-apply

Load and apply existing calibration before starting the solve chain:

```yaml
pre_apply_file: [a_calibration.h5]   # HDF5 file(s) to load
pre_apply_jones: [A]                  # Jones type(s) to apply
```

## Sequential Solving Logic

### Example 1: Basic Sequential (K → G → B)

```yaml
jones: [K, G, B]
pre_apply_file: []
pre_apply_jones: []
```

**Execution:**
1. Solve K (no pre-apply)
2. Solve G with K pre-applied
3. Solve B with K+G pre-applied

### Example 2: With External Pre-apply (A → K → G)

```yaml
jones: [K, G]
pre_apply_file: [a_cal.h5]
pre_apply_jones: [A]
```

**Execution:**
1. Load external A from `a_cal.h5`
2. Solve K with A pre-applied
3. Solve G with A+K pre-applied (K from step 2)

**Key Rule:** External pre-apply is loaded FIRST, then the chain builds in the order specified in `jones` list.

## Smart Parameter Expansion

Single values broadcast to all Jones types:

```yaml
jones: [K, G, B]
ref_ant: 4                        # All three use ref_ant=4
time_interval: 60s                # All three use 60s
```

Lists must match the number of Jones types:

```yaml
jones: [K, G, B]
time_interval: [inf, 60s, 30s]   # K: inf, G: 60s, B: 30s
freq_interval: [4MHz, full, 1MHz] # K: 4MHz, G: full, B: 1MHz
```

Mixing is allowed:

```yaml
jones: [K, G, B]
ref_ant: 4                        # Single -> broadcast
time_interval: [inf, 60s, 30s]   # List -> per-Jones
```

### Per-Jones SPW Selection (CASA-style)

Different Jones types can use different SPWs and channel ranges - this is standard CASA calibration practice:

```yaml
jones: [K, G, B]

# Different SPW+channel selection per Jones
spw: ['0:250~1800', '1:120~1450', '0,1']
# K: SPW 0, channels 250-1800 (wideband delay, exclude edges)
# G: SPW 1, channels 120-1450 (different SPW, different bad channels)
# B: SPWs 0,1, all channels (full spectral coverage for bandpass)
```

**Common use cases:**
- Different SPWs have different edge channel issues
- Different RFI per SPW (e.g., L-band: SPW 0 has low-freq RFI, SPW 1 has high-freq RFI)
- Wideband delay (K) on one SPW, narrowband gain (G) on another
- Bandpass (B) needs full multi-SPW coverage
- VLA/ALMA: Different subbands have different passband shapes

**More examples:**
```yaml
# Mix channel selection and full SPWs
spw: ['0:100~1900', '1', '0,1,2']

# Same channel range across multiple SPWs
spw: ['0:200~1800', '1:200~1800', '0~1:200~1800']

# Single SPW with different channel ranges per Jones
spw: ['0:500~1500', '0:300~1700', '0']
```

## Examples Provided

- **`k_delay.yaml`** - Single K delay calibration
- **`k_delay_chansel.yaml`** - K delay with channel selection
- **`g_gain.yaml`** - Single G gain calibration
- **`sequential_kgb.yaml`** - Multi-Jones K→G→B sequential solving
- **`sequential_per_spw.yaml`** - Per-Jones SPW selection (K on SPW 0, G on SPW 1, B on both)
- **`with_external_preapply.yaml`** - External pre-apply with K→G chain

## Multi-Block Calibration

You can have multiple solve blocks for different observations:

```yaml
solve:
  # First observation (flux calibrator)
  - jones: [K, G]
    ms: flux_cal.ms
    field: 3C147
    output: flux_cal.h5

  # Second observation (phase calibrator)
  - jones: [G]
    ms: phase_cal.ms
    field: J1234+5678
    output: phase_cal.h5
    # Use flux cal results as pre-apply
    pre_apply_file: [flux_cal.h5]
    pre_apply_jones: [K]
```

## Command-Line Interface

### Available Commands

```bash
# Run calibration from YAML config
jackal run config.yaml

# Show solution information
jackal info solution.h5

# Show JACKAL version
jackal version

# Get help
jackal --help
jackal run --help
```

### jackal run

Run calibration from YAML configuration file:

```bash
jackal run sequential_kgb.yaml
```

**Output:**
- Displays progress with timestamps
- Shows sequential solving: K → G (with K) → B (with K+G)
- Reports pre-apply chain for each Jones
- Prints summary with output files

**Options:**
- `--verbose, -v`: Show full tracebacks on errors

### jackal info

Display information about calibration solution:

```bash
jackal info calibration.h5
```

**Shows:**
- Jones type and metadata (antennas, channels, times)
- Jones array shape and dimensions
- Time and frequency coverage
- Convergence information
- Flagging statistics
- Multi-Jones file contents

### Examples

```bash
# Run K delay calibration
jackal run k_delay.yaml

# Run sequential K→G→B calibration
jackal run sequential_kgb.yaml

# Inspect the output
jackal info k_delay.h5

# Run with verbose error output
jackal run config.yaml -v
```

## Python API

### Load and run config

```python
from jackal import solve_from_yaml

# Run entire config (all solve blocks)
results = solve_from_yaml('config.yaml')

# Results dict contains all solve blocks
# Output files written as specified in YAML
```

### Save Python config to YAML

```python
from jackal import config_to_yaml

config_to_yaml(
    jones_list=['K', 'G'],
    ms_path='obs.ms',
    output_path='cal.h5',
    yaml_path='config.yaml',
    field='3C286',
    ref_ant=4,
    freq_interval=['4MHz', 'full'],
    time_interval=['inf', '60s'],
    phase_only=[False, True]  # G: phase-only
)
```

## Advantages Over Single-Jones Configs

**Old Approach (multiple files):**
```bash
# k_config.yaml, g_config.yaml, b_config.yaml
# Manual pre-apply handling in Python
```

**New Approach (one file):**
```yaml
solve:
  - jones: [K, G, B]
    # Automatic pre-apply chaining
    # One output file with all solutions
```

**Benefits:**
- Single YAML file for entire calibration pipeline
- Automatic pre-apply chaining (no manual intervention)
- Smart parameter expansion (less repetition)
- Clear sequential logic in one place
- Combined output for easy reuse

## Best Practices

1. **Start with K delay** for wideband data to correct for delay offsets
2. **Follow with G gains** to get time-variable amplitude/phase corrections
3. **Finish with B bandpass** for frequency-dependent corrections
4. **Use inf for K** (one solution per scan), short intervals for G (60s typical)
5. **Enable RFI flagging** with threshold 5.0 for robust solutions
6. **Set feed_basis='auto'** to automatically detect from MS

## Notes

- All Jones solutions are saved to a single HDF5 file specified by `output`
- Pre-apply chaining happens automatically in the order specified
- Field comma syntax is parsed but currently uses the first field (multi-field solving can be enhanced)
- Solver-specific parameters are optional and only apply to relevant Jones types
