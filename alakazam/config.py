"""
ALAKAZAM Configuration.

Parses YAML config with solve: and apply: blocks.
Crashes early with clear error messages on bad input.

Config format:
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
        max_iter: 100
        tol: 1e-10
        apply_parang: false

    apply:
      - ms: target.ms
        jones: [K, G, G]
        tables: [cal.h5]
        output_col: CORRECTED_DATA
        spw: "*"
        field: target

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import sys
import re
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import logging

logger = logging.getLogger("alakazam")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SolveStep:
    """One Jones term within a solve block."""
    jones_type: str
    time_interval: str = "inf"
    freq_interval: str = "full"
    phase_only: bool = False
    ref_ant: int = 0
    rfi_threshold: float = 5.0
    max_iter: int = 100
    tol: float = 1e-10


@dataclass
class SolveBlock:
    """A single solve: entry."""
    ms: str
    output: str
    ref_ant: int
    steps: List[SolveStep]
    field: Optional[str] = None
    spw: str = "*"
    scans: Optional[str] = None
    data_col: str = "DATA"
    model_col: str = "MODEL_DATA"
    apply_parang: bool = False
    memory_limit_gb: float = 0.0


@dataclass
class ApplyBlock:
    """A single apply: entry."""
    ms: str
    jones: List[str]
    tables: List[str]
    output_col: str = "CORRECTED_DATA"
    spw: str = "*"
    field: Optional[str] = None


@dataclass
class AlakazamConfig:
    """Full parsed config."""
    solve_blocks: List[SolveBlock] = field(default_factory=list)
    apply_blocks: List[ApplyBlock] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

JONES_ALIASES = {
    "K": "K", "DELAY": "K",
    "G": "G", "GAIN": "G",
    "B": "G", "BANDPASS": "G",
    "D": "D", "LEAKAGE": "D",
    "XF": "XF", "CROSSPHASE": "XF",
    "KCROSS": "KCROSS", "CROSSDELAY": "KCROSS",
}


def load_config(path: str) -> AlakazamConfig:
    """Load and validate YAML config. Crashes on bad input."""
    import yaml

    path = Path(path)
    if not path.exists():
        _die(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        _die("Config must be a YAML mapping")

    if "solve" not in raw and "apply" not in raw:
        _die("Config must have at least a 'solve:' or 'apply:' block")

    config = AlakazamConfig()

    # Parse solve blocks
    for idx, entry in enumerate(raw.get("solve", []) or []):
        config.solve_blocks.append(_parse_solve_block(entry, idx + 1))

    # Parse apply blocks
    for idx, entry in enumerate(raw.get("apply", []) or []):
        config.apply_blocks.append(_parse_apply_block(entry, idx + 1))

    return config


def _parse_solve_block(entry: dict, num: int) -> SolveBlock:
    """Parse one solve: entry."""
    required = ["jones", "ms", "output", "ref_ant"]
    for r in required:
        if r not in entry or entry[r] is None:
            _die(f"solve entry #{num}: missing required field '{r}'")

    jones_types = _ensure_list(entry["jones"])
    n = len(jones_types)
    if n == 0:
        _die(f"solve entry #{num}: 'jones' must be non-empty")

    # Normalize Jones types
    jones_types = [_normalize_jones(j, num) for j in jones_types]

    # Per-Jones lists
    time_int = _expand(entry.get("time_interval", "inf"), n)
    freq_int = _expand(entry.get("freq_interval", "full"), n)
    phase_only = _expand(entry.get("phase_only", False), n)
    rfi = entry.get("rfi_threshold", 5.0)
    max_iter = entry.get("max_iter", 100)
    tol = entry.get("tol", 1e-10)
    ref_ant_list = _expand(entry.get("ref_ant", 0), n)

    # Build steps
    steps = []
    for i in range(n):
        ra = int(ref_ant_list[i]) if isinstance(ref_ant_list[i], (int, float)) else int(entry["ref_ant"])
        steps.append(SolveStep(
            jones_type=jones_types[i],
            time_interval=str(time_int[i]),
            freq_interval=str(freq_int[i]),
            phase_only=bool(phase_only[i]),
            ref_ant=ra,
            rfi_threshold=float(rfi) if not isinstance(rfi, list) else float(rfi[i] if i < len(rfi) else rfi[0]),
            max_iter=int(max_iter) if not isinstance(max_iter, list) else int(max_iter[i] if i < len(max_iter) else max_iter[0]),
            tol=float(tol) if not isinstance(tol, list) else float(tol[i] if i < len(tol) else tol[0]),
        ))

    # Validate MS exists
    ms_path = str(entry["ms"])
    if not Path(ms_path).exists():
        _die(f"solve entry #{num}: MS not found: {ms_path}")

    return SolveBlock(
        ms=ms_path,
        output=str(entry["output"]),
        ref_ant=int(entry["ref_ant"]),
        steps=steps,
        field=entry.get("field"),
        spw=str(entry.get("spw", "*")),
        scans=str(entry["scans"]) if entry.get("scans") else None,
        data_col=entry.get("data_col", "DATA"),
        model_col=entry.get("model_col", "MODEL_DATA"),
        apply_parang=bool(entry.get("apply_parang", False)),
        memory_limit_gb=float(entry.get("memory_limit_gb", 0)),
    )


def _parse_apply_block(entry: dict, num: int) -> ApplyBlock:
    """Parse one apply: entry."""
    for r in ["ms", "jones", "tables"]:
        if r not in entry or entry[r] is None:
            _die(f"apply entry #{num}: missing required field '{r}'")

    jones = _ensure_list(entry["jones"])
    tables = _ensure_list(entry["tables"])

    # Expand single table to all Jones
    if len(tables) == 1 and len(jones) > 1:
        tables = tables * len(jones)
    if len(tables) != len(jones):
        _die(f"apply entry #{num}: tables count ({len(tables)}) must match jones count ({len(jones)})")

    return ApplyBlock(
        ms=str(entry["ms"]),
        jones=[_normalize_jones(j, num) for j in jones],
        tables=[str(t) for t in tables],
        output_col=entry.get("output_col", "CORRECTED_DATA"),
        spw=str(entry.get("spw", "*")),
        field=entry.get("field"),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_jones(name: str, block_num: int) -> str:
    key = str(name).upper().strip()
    if key not in JONES_ALIASES:
        _die(f"Block #{block_num}: invalid Jones type '{name}'. "
             f"Valid: K, G, B, D, Xf, Kcross")
    return JONES_ALIASES[key]


def _ensure_list(val):
    if isinstance(val, list):
        return val
    return [val]


def _expand(val, n: int) -> list:
    lst = _ensure_list(val)
    if len(lst) == 1:
        return lst * n
    if len(lst) != n:
        _die(f"List length {len(lst)} doesn't match jones count {n}")
    return lst


def _die(msg: str):
    print(f"\n  ALAKAZAM CONFIG ERROR: {msg}\n", file=sys.stderr)
    sys.exit(1)


def config_to_yaml(config: AlakazamConfig) -> str:
    """Serialize config back to YAML string for reproducibility."""
    import yaml
    d = {"solve": [], "apply": []}
    for sb in config.solve_blocks:
        d["solve"].append({
            "ms": sb.ms, "output": sb.output, "ref_ant": sb.ref_ant,
            "field": sb.field, "spw": sb.spw,
            "jones": [s.jones_type for s in sb.steps],
            "time_interval": [s.time_interval for s in sb.steps],
            "freq_interval": [s.freq_interval for s in sb.steps],
            "phase_only": [s.phase_only for s in sb.steps],
        })
    for ab in config.apply_blocks:
        d["apply"].append({
            "ms": ab.ms, "jones": ab.jones, "tables": ab.tables,
            "output_col": ab.output_col,
        })
    return yaml.dump(d, default_flow_style=False)
