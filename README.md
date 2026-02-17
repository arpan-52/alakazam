# ALAKAZAM

**A Radio Interferometric Calibration Suite**

Developed by Arpan Pal 2026, NRAO / NCRA

## Install

```bash
pip install .
```

## Usage

```bash
# Run calibration
alakazam run config.yaml

# Inspect solutions
alakazam info cal.h5
```

## Config Format

```yaml
solve:
  - jones: [K, G, G]
    ms: calibrator.ms
    field: 3C286
    spw: "*"
    ref_ant: 0
    output: cal.h5
    time_interval: [inf, inf, 2min]
    freq_interval: [full, 4MHz, full]
    phase_only: [false, false, false]
    rfi_threshold: 5.0

apply:
  - ms: target.ms
    jones: [K, G, G]
    tables: [cal.h5]
    output_col: CORRECTED_DATA
```

## Jones Types

| Type   | Solves for              | Freq handling     |
|--------|-------------------------|-------------------|
| K      | Delay (ns per antenna)  | Needs all channels|
| G      | Complex gain per ant    | Can avg freq      |
| D      | Instrumental leakage    | Can avg freq      |
| Xf     | Cross-hand phase        | Can avg freq      |
| Kcross | Cross-hand delay        | Needs all channels|

## Architecture

```
alakazam/
├── cli.py            CLI (run + info)
├── config.py         YAML parsing
├── pipeline.py       Orchestrator (solve + apply)
├── jones/            Jones algebra
│   ├── algebra.py    2×2 mat ops
│   ├── constructors.py  param → Jones
│   └── parang.py     Parallactic angle
├── solvers/          One class per Jones type
├── core/             MS I/O, averaging, RFI, memory
└── io/               HDF5 save/load
```

## License

MIT
