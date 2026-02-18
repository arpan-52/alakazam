# ALAKAZAM

**Radio Interferometric Calibration Pipeline**
Arpan Pal — NRAO / NCRA, 2026

---

## What it does

ALAKAZAM solves for Jones matrix calibration terms from radio interferometric Measurement Sets (MS), applies flux scale bootstrapping, and applies corrected solutions back to target data. It supports the full RIME (Radio Interferometer Measurement Equation) formalism with chained Jones terms, multi-field solving, flexible interpolation, and complete provenance tracking.

**Supported Jones types**

| Type    | Description                         | Native param      |
|---------|-------------------------------------|-------------------|
| `K`     | Delay (frequency-dependent phase)   | delay (ns)        |
| `G`     | Complex gain (amp + phase)          | amp, phase        |
| `D`     | Leakage (off-diagonal)              | d_pq, d_qp        |
| `Xf`    | Cross-hand phase                    | phi_pq (rad)      |
| `Kcross`| Cross-hand delay                    | delay_pq (ns)     |

**Feed bases**: LINEAR (XX/XY/YX/YY) and CIRCULAR (RR/RL/LR/LL) — auto-detected from MS.

---

## Install

```bash
pip install .
```

Requires Python ≥ 3.9. Dependencies: numpy, scipy, numba, h5py, pyyaml, psutil, rich, python-casacore.

---

## Quick start

```bash
alakazam run config.yaml          # run solve + fluxscale + apply
alakazam info cal.h5              # inspect a solution file
alakazam fluxscale-info cal.h5    # show stored scale factors
```

---

## Config format

A config file has three top-level blocks: `solve`, `fluxscale`, `apply`. All blocks are optional — run any combination.

### `solve:`

```yaml
solve:
  - ms: all_cals.ms
    output: cal.h5
    ref_ant: 0
    data_col: DATA
    model_col: MODEL_DATA
    apply_parang: false
    memory_limit_gb: 0      # 0 = auto-detect from available RAM
    rfi_threshold: 5.0
    max_iter: 100
    tol: 1.0e-10

    # Jones chain — one entry per solve step
    jones:         [K,        G,                  G                ]

    # Field groups — [[fields for step 1], [fields for step 2], ...]
    # Scalar expands to all steps. Single field needs no brackets.
    # Multi-field per step: solve independently on each, store all.
    field:         [[3C147],  [3C147, 3C286],     [PKS1934, J0538] ]

    # Scan selection per step per field — matches field structure.
    # Absent = all scans for that field.
    # scans: [[1,2,3], [[1,2], [3,4]], [[5,6], [7,8]]]

    # SPW selection per step — absent = all SPWs.
    # Supports: 0,1,2  or  0~3  or absent (all)
    # step_spw: [0~3, 0~3, 0~3]

    # Per-step solve parameters
    time_interval: [inf,      inf,                5min             ]
    freq_interval: [full,     full,               full             ]
    phase_only:    [false,    false,              false            ]

    # Interpolation used when pre-applying previous steps before solving next
    # exact        = stamp whole solint block as-is (no interpolation)
    # nearest      = nearest solution slot in time
    # linear       = linear interpolation in time AND frequency
    # cubic        = cubic spline in time AND frequency
    preapply_time_interp: [exact, linear, linear]

    # External preapply — ONLY needed when injecting solutions from a
    # previous run before the chain starts. Absent = pure auto-chain.
    # Auto-chain always applies: before step N, preapply steps 1..N-1.
    # With external_preapply A: before step 1 apply A, before step 2
    # apply A+1, before step 3 apply A+1+2, etc.
    external_preapply:
      tables:       [prev.h5,       prev.h5      ]
      jones:        [K,             G            ]
      field_select: [nearest_time,  nearest_sky  ]
      time_interp:  [exact,         linear       ]
```

#### Time interval formats

| Value    | Meaning                          |
|----------|----------------------------------|
| `inf`    | Whole observation — one solution |
| `5min`   | 5-minute slots                   |
| `120s`   | 120-second slots                 |
| `30`     | 30-second slots (bare number)    |

#### Frequency interval formats

| Value    | Meaning                          |
|----------|----------------------------------|
| `full`   | Whole SPW — one solution         |
| `4MHz`   | 4 MHz channels                   |
| `32`     | 32-channel slots (bare number)   |

---

### `fluxscale:`

Bootstraps flux scale from a known calibrator to transfer fields. Reads G solutions, computes amplitude ratio, writes rescaled solutions to a new file. The original file is never modified.

```yaml
fluxscale:
  - reference_table: cal.h5        # H5 with G solutions for known flux source
    reference_field: 3C147         # field name inside that table

    transfer_table: cal.h5         # H5 with G solutions for gain calibrator
    transfer_field: [PKS1934, J0538]  # one or more fields to rescale

    output: cal_scaled.h5          # NEW file — append if already exists
    jones_type: G                  # which Jones type to use for ratio
    # spw absent = all SPWs
```

**What `output` contains:**
- Everything from `transfer_table` copied verbatim
- G solutions for each `transfer_field` rescaled: `g_scaled = g × √scale`
- `fluxscale/` group with scale factors, scatter, provenance per field per SPW

Multiple fluxscale blocks can write to the same output — they append, never overwrite.

---

### `apply:`

Applies any combination of Jones solutions to a target MS.

```yaml
apply:
  - ms: science.ms
    output_col: CORRECTED_DATA
    target_field: NGC1234          # field to correct in target MS
    apply_parang: false
    # spw absent = all SPWs
    # scans absent = all scans

    # Parallel lists — one entry per Jones term, applied in order
    jones:  [K,      G,             G            ]
    tables: [cal.h5, cal_scaled.h5, cal_scaled.h5]

    # How to pick which field's solutions to use from each table:
    #   nearest_time = closest solution slot in time across all fields
    #   nearest_sky  = closest field on sky to target_field RA/Dec
    #   pinned       = use solution_field exactly
    field_select: [nearest_time, nearest_sky, pinned]

    # How to interpolate once field is selected:
    #   exact   = stamp whole solint block as-is (no time/freq interpolation)
    #   nearest = nearest solution slot
    #   linear  = linear in time AND frequency
    #   cubic   = cubic spline in time AND frequency
    time_interp: [exact, linear, cubic]

    # Required only when field_select is pinned
    solution_field: [null, null, PKS1934]
```

---

## Interpolation — precise meaning

**`exact`** — The solution covering a given solint block is applied identically to every time/frequency sample in that block. No interpolation. The solint grid is preserved as-is.

**`nearest`** — Find the single closest solution slot in time. Apply it to all samples without interpolation.

**`linear`** — Linearly interpolate between bracketing solution slots in **both time and frequency**. For diagonal terms: amplitude and phase interpolated separately (unwrapped phase). For off-diagonal terms: complex linear interpolation.

**`cubic`** — Cubic spline in **both time and frequency**. Smoother than linear, better for slowly-varying terms like G.

**`nearest_sky`** (field select) — Among all fields stored in the table, find the one with the smallest angular separation from the target field's phase centre. Uses RA/Dec stored in solution metadata at solve time.

---

## Solve chain — how preapply works

**No external_preapply:**
```
raw data
  → solve step 1               → stored in cal.h5 / field_A
  → preapply step 1
  → solve step 2               → stored in cal.h5 / field_B, field_C
  → preapply step 1 + step 2
  → solve step 3               → stored in cal.h5 / field_D, field_E
```

**With external_preapply (solutions A from prev.h5):**
```
raw data
  → preapply A (external)
  → solve step 1               → stored
  → preapply A + step 1
  → solve step 2               → stored
  → preapply A + step 1 + step 2
  → solve step 3               → stored
```

A stays in the stack throughout. Every subsequent solve sees it.

---

## HDF5 solution format

```
cal.h5
├── attrs:  alakazam_version, created_utc, ms, ref_ant
├── G/
│   ├── field_3C147/
│   │   └── spw_0/
│   │       ├── jones         (n_t, n_f, n_ant, 2, 2) complex128
│   │       ├── time          (n_t,) MJD seconds
│   │       ├── freq          (n_f,) Hz
│   │       ├── flags         (n_t, n_f, n_ant) bool
│   │       └── attrs:        jones_type, field_name, field_ra_rad,
│   │                         field_dec_rad, scan_ids, ref_ant,
│   │                         time_interval, freq_interval, phase_only,
│   │                         preapply_chain (JSON — full provenance)
│   └── field_3C286/
│       └── spw_0/ ...
└── K/
    └── field_3C147/
        └── spw_0/
            ├── jones         (n_t, n_f, n_ant, 2, 2) complex128
            ├── delay         (n_t, n_ant, 2) float64  ← native params
            └── attrs: ...

cal_scaled.h5
├── G/ ...                     (everything from transfer_table)
└── fluxscale/
    ├── field_PKS1934/
    │   └── spw_0/
    │       └── attrs:  scale_p, scale_q, scatter_p, scatter_q,
    │                   n_ant, reference_field, reference_table,
    │                   jones_type, created_utc
    └── field_J0538/
        └── spw_0/ ...
```

**Provenance** — every solution group stores `preapply_chain` as a JSON array listing exactly what was applied before this step was solved: source (internal/external), table, jones type, field, field_select mode, time_interp mode. Full A-to-Z traceability.

---

## Architecture

```
alakazam/
├── __init__.py
├── __main__.py          → calls cli.main()
├── cli.py               → argparse: run / info / fluxscale-info
├── config.py            → dataclasses + YAML parser
├── flow.py              → orchestrator: solve → fluxscale → apply
│                           3-tier memory management, fork COW workers
├── jones/
│   ├── algebra.py       → 2×2 ops, apply/unapply, residuals (numba)
│   ├── constructors.py  → delay/gain/leakage/crossphase → Jones (numba)
│   └── parang.py        → parallactic angle computation and Jones
├── solvers/
│   ├── __init__.py      → JonesSolver ABC, registry, get_solver()
│   ├── k_delay.py
│   ├── g_gain.py
│   ├── d_leakage.py
│   ├── xf_crossphase.py
│   └── kcross_delay.py
├── core/
│   ├── ms_io.py         → casacore wrappers, metadata, solint grid
│   ├── averaging.py     → baseline averaging (numba)
│   ├── interpolation.py → nearest/linear/cubic in time+freq, sky select
│   ├── memory.py        → 3-tier RAM management
│   ├── rfi.py           → MAD-based flagging
│   └── quality.py       → solution quality metrics
├── io/
│   └── hdf5.py          → read/write solutions, fluxscale, provenance
└── calibration/
    ├── apply.py         → load → interpolate → compose chain → write
    └── fluxscale.py     → compute scale factors, rescale, write output
```

### Memory management (3-tier)

The solver decides how to load data based on available RAM:

| Tier | Condition | Strategy |
|------|-----------|----------|
| T1 | Full SPW fits in RAM | Single load, all time slots parallel |
| T2 | N slots fit | Batched: load N slots at a time |
| T3 | One slot too big | Pseudo-chunk: progressive accumulation within a slot |

Workers (multiprocessing) **never touch disk**. All data is shared read-only via fork copy-on-write.

### Multiprocessing

Each solint×freqint cell is an independent least-squares problem. Cells are dispatched to a process pool. Workers receive only indices; data is in the parent process's memory via CoW fork. Results (solutions) are collected back in the main process and written to HDF5 once per SPW.

---

## Examples

### Simplest: one MS, one field, KG chain

```yaml
solve:
  - ms: cal.ms
    output: cal.h5
    ref_ant: 0
    jones: [K, G]
    field: 3C286
    time_interval: [inf, 5min]
    freq_interval: [full, full]

apply:
  - ms: science.ms
    target_field: target
    jones: [K, G]
    tables: [cal.h5, cal.h5]
    field_select: [nearest_time, nearest_time]
    time_interp: [exact, linear]
    output_col: CORRECTED_DATA
```

### Multi-field solve with fluxscale

```yaml
solve:
  - ms: all_cals.ms
    output: cal.h5
    ref_ant: 0
    jones:         [K,       G,                 G               ]
    field:         [[3C147], [3C147, 3C286],    [PKS1934, J0538]]
    time_interval: [inf,     inf,               5min            ]
    freq_interval: [full,    full,              full            ]
    preapply_time_interp: [exact, linear, linear]

fluxscale:
  - reference_table: cal.h5
    reference_field: 3C147
    transfer_table: cal.h5
    transfer_field: [PKS1934, J0538]
    output: cal_scaled.h5

apply:
  - ms: science.ms
    target_field: NGC1234
    jones:  [K,      G            ]
    tables: [cal.h5, cal_scaled.h5]
    field_select: [nearest_time, nearest_sky]
    time_interp:  [exact,        cubic      ]
    output_col: CORRECTED_DATA
```

### Full polarisation calibration

```yaml
solve:
  - ms: polcal.ms
    output: polcal.h5
    ref_ant: 0
    jones:         [K,      G,    D,    Xf   ]
    field:         [3C286,  3C286, 3C286, 3C286]
    time_interval: [inf,    5min, inf,  inf  ]
    freq_interval: [full,   full, full, full ]
    phase_only:    [false,  false, false, false]
    preapply_time_interp: [exact, linear, linear, linear]

apply:
  - ms: science.ms
    target_field: target
    jones:  [K,         G,         D,         Xf        ]
    tables: [polcal.h5, polcal.h5, polcal.h5, polcal.h5 ]
    field_select: [nearest_time, nearest_time, nearest_time, nearest_time]
    time_interp:  [exact,        linear,       linear,       exact       ]
    output_col: CORRECTED_DATA
    apply_parang: true
```

### With external preapply (pre-existing K solutions)

```yaml
solve:
  - ms: gaincal.ms
    output: gaincal.h5
    ref_ant: 0
    jones: [G]
    field: PKS1934
    time_interval: [5min]
    freq_interval: [full]

    external_preapply:
      tables:       [prev_run.h5]
      jones:        [K]
      field_select: [nearest_time]
      time_interp:  [exact]

apply:
  - ms: science.ms
    target_field: NGC1234
    jones:  [K,            G          ]
    tables: [prev_run.h5,  gaincal.h5 ]
    field_select: [nearest_time, nearest_sky]
    time_interp:  [exact,        cubic      ]
    output_col: CORRECTED_DATA
```

---

## CLI reference

```
alakazam run config.yaml              Run all blocks in config
alakazam info solution.h5             Print solution file summary
alakazam fluxscale-info solution.h5   Print stored scale factors
```

---

## Developed by
Arpan Pal, NRAO / NCRA, 2026
