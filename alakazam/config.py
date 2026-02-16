"""
ALAKAZAM Configuration.

Parses YAML config, validates all fields, expands per-Jones lists.
Crashes early with clear error messages on invalid input.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import yaml
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger("alakazam")


@dataclass
class SolveStep:
    """Configuration for one solve step (one Jones type)."""
    jones_type: str
    time_interval: str = "inf"
    freq_interval: str = "full"
    phase_only: bool = False
    ref_ant: int = 0
    rfi_threshold: float = 5.0
    max_iter: int = 100
    tol: float = 1e-10


@dataclass
class AlakazamConfig:
    """Complete pipeline configuration."""
    ms_path: str
    output: str = "calibration.h5"
    field: Optional[str] = None
    spw: str = "*"
    scans: Optional[str] = None
    data_col: str = "DATA"
    model_col: str = "MODEL_DATA"
    ref_ant: int = 0
    apply_parang: bool = True
    rfi_threshold: float = 5.0
    max_iter: int = 100
    tol: float = 1e-10
    memory_limit_gb: float = 0.0  # 0 = auto (60% of available)
    steps: List[SolveStep] = field(default_factory=list)


def _ensure_list(val, n: int, name: str):
    """Ensure val is a list of length n. If scalar, repeat it."""
    if isinstance(val, list):
        if len(val) == 1:
            return val * n
        if len(val) != n:
            raise ValueError(
                f"'{name}' has {len(val)} entries but 'jones' has {n}. "
                f"Must be a single value or a list matching the number of Jones types."
            )
        return val
    return [val] * n


def _validate_jones_type(jt: str) -> str:
    """Validate and normalize Jones type string."""
    valid = {"K", "G", "D", "XF", "KCROSS"}
    jt_upper = jt.upper()
    # Allow common aliases
    aliases = {"B": "G", "BANDPASS": "G", "GAIN": "G", "DELAY": "K",
               "LEAKAGE": "D", "CROSSPHASE": "XF", "CROSSDELAY": "KCROSS"}
    if jt_upper in aliases:
        jt_upper = aliases[jt_upper]
    if jt_upper not in valid:
        raise ValueError(
            f"Unknown Jones type '{jt}'. Valid types: K, G, D, Xf, Kcross. "
            f"(B is an alias for G — use freq_interval to control bandpass behavior.)"
        )
    return jt_upper


def _validate_time_interval(ti: str) -> str:
    """Validate time interval string."""
    ti = str(ti).strip().lower()
    if ti in ("inf", "full", "infinite"):
        return "inf"
    import re
    match = re.match(r'^([\d.]+)\s*(s|sec|min|m|h|hour)$', ti)
    if match:
        return ti
    # Try bare number (assume seconds)
    try:
        float(ti)
        return f"{ti}s"
    except ValueError:
        raise ValueError(
            f"Cannot parse time_interval '{ti}'. "
            f"Examples: 'inf', '30s', '2min', '0.5h'"
        )


def _validate_freq_interval(fi: str) -> str:
    """Validate frequency interval string."""
    fi = str(fi).strip().lower()
    if fi in ("full", "inf", "infinite"):
        return "full"
    if fi == "spw":
        return "spw"
    import re
    match = re.match(r'^([\d.]+)\s*(hz|khz|mhz|ghz|chan|channels?)$', fi)
    if match:
        return fi
    try:
        int(fi)
        return f"{fi}chan"
    except ValueError:
        raise ValueError(
            f"Cannot parse freq_interval '{fi}'. "
            f"Examples: 'full', 'spw', '4MHz', '128chan'"
        )


def load_config(config_path: str) -> AlakazamConfig:
    """Load and validate YAML configuration.

    Parameters
    ----------
    config_path : str
        Path to YAML config file.

    Returns
    -------
    config : AlakazamConfig
        Validated configuration.

    Raises
    ------
    SystemExit
        On validation errors (crashes with clear message).
    """
    config_path = Path(config_path)
    if not config_path.exists():
        _die(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        _die(f"Config file is empty: {config_path}")
    if not isinstance(raw, dict):
        _die(f"Config must be a YAML dictionary, got {type(raw).__name__}")

    # Required field
    if "ms" not in raw and "ms_path" not in raw:
        _die("Config must have 'ms' or 'ms_path' field specifying the Measurement Set.")

    ms_path = str(raw.get("ms", raw.get("ms_path", "")))
    if not ms_path:
        _die("'ms' cannot be empty.")

    # Jones types (required)
    if "jones" not in raw:
        _die("Config must have 'jones' field. Example: jones: [K, G]")
    jones_raw = raw["jones"]
    if isinstance(jones_raw, str):
        jones_list = [s.strip() for s in jones_raw.split(",")]
    elif isinstance(jones_raw, list):
        jones_list = [str(j).strip() for j in jones_raw]
    else:
        _die(f"'jones' must be a string or list, got {type(jones_raw).__name__}")

    n_jones = len(jones_list)
    if n_jones == 0:
        _die("'jones' list is empty. Need at least one Jones type.")

    # Validate Jones types
    try:
        jones_types = [_validate_jones_type(j) for j in jones_list]
    except ValueError as e:
        _die(str(e))

    # Per-Jones fields (can be single or list)
    try:
        time_intervals = _ensure_list(
            raw.get("time_interval", "inf"), n_jones, "time_interval")
        freq_intervals = _ensure_list(
            raw.get("freq_interval", "full"), n_jones, "freq_interval")
        phase_only_list = _ensure_list(
            raw.get("phase_only", False), n_jones, "phase_only")
    except ValueError as e:
        _die(str(e))

    # Validate intervals
    try:
        time_intervals = [_validate_time_interval(str(t)) for t in time_intervals]
        freq_intervals = [_validate_freq_interval(str(f)) for f in freq_intervals]
    except ValueError as e:
        _die(str(e))

    # Global fields
    ref_ant = int(raw.get("ref_ant", 0))
    output = str(raw.get("output", "calibration.h5"))
    field_name = raw.get("field", None)
    if field_name is not None:
        field_name = str(field_name)
    spw = str(raw.get("spw", "*"))
    scans = raw.get("scans", None)
    if scans is not None:
        scans = str(scans)
    data_col = str(raw.get("data_col", raw.get("data_column", "DATA")))
    model_col = str(raw.get("model_col", raw.get("model_column", "MODEL_DATA")))
    apply_parang = bool(raw.get("apply_parang", raw.get("parang", True)))
    rfi_threshold = float(raw.get("rfi_threshold", raw.get("rfi_sigma", 5.0)))
    max_iter = int(raw.get("max_iter", 100))
    tol = float(raw.get("tol", raw.get("tolerance", 1e-10)))
    memory_limit = float(raw.get("memory_limit_gb", raw.get("memory_limit", 0.0)))

    # Per-Jones ref_ant (can be overridden)
    ref_ant_list = _ensure_list(raw.get("ref_ant", ref_ant), n_jones, "ref_ant")
    ref_ant_list = [int(r) for r in ref_ant_list]

    # Per-Jones rfi_threshold
    rfi_list = _ensure_list(
        raw.get("rfi_threshold", raw.get("rfi_sigma", rfi_threshold)),
        n_jones, "rfi_threshold")
    rfi_list = [float(r) for r in rfi_list]

    # Per-Jones max_iter
    maxiter_list = _ensure_list(raw.get("max_iter", max_iter), n_jones, "max_iter")
    maxiter_list = [int(m) for m in maxiter_list]

    # Per-Jones tol
    tol_list = _ensure_list(raw.get("tol", raw.get("tolerance", tol)), n_jones, "tol")
    tol_list = [float(t) for t in tol_list]

    # Build solve steps
    steps = []
    for i in range(n_jones):
        steps.append(SolveStep(
            jones_type=jones_types[i],
            time_interval=time_intervals[i],
            freq_interval=freq_intervals[i],
            phase_only=bool(phase_only_list[i]),
            ref_ant=ref_ant_list[i],
            rfi_threshold=rfi_list[i],
            max_iter=maxiter_list[i],
            tol=tol_list[i],
        ))

    config = AlakazamConfig(
        ms_path=ms_path,
        output=output,
        field=field_name,
        spw=spw,
        scans=scans,
        data_col=data_col,
        model_col=model_col,
        ref_ant=ref_ant,
        apply_parang=apply_parang,
        rfi_threshold=rfi_threshold,
        max_iter=max_iter,
        tol=tol,
        memory_limit_gb=memory_limit,
        steps=steps,
    )

    _validate_config(config)
    return config


def _validate_config(config: AlakazamConfig):
    """Final validation — crash on errors."""
    # Check MS exists
    ms = Path(config.ms_path)
    if not ms.exists():
        _die(f"Measurement Set not found: {config.ms_path}")

    # Check output directory writable
    out = Path(config.output)
    out_dir = out.parent
    if not out_dir.exists():
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            _die(f"Cannot create output directory {out_dir}: {e}")

    # K and Kcross need freq
    for step in config.steps:
        if step.jones_type in ("K", "KCROSS"):
            if step.phase_only:
                _die(f"phase_only is not supported for {step.jones_type} solver.")

    # ref_ant must be non-negative
    for step in config.steps:
        if step.ref_ant < 0:
            _die(f"ref_ant must be >= 0, got {step.ref_ant}")

    logger.info("Configuration validated successfully.")


def config_to_yaml(config: AlakazamConfig) -> str:
    """Serialize config back to YAML string."""
    d = {
        "ms": config.ms_path,
        "output": config.output,
        "jones": [s.jones_type for s in config.steps],
        "time_interval": [s.time_interval for s in config.steps],
        "freq_interval": [s.freq_interval for s in config.steps],
        "ref_ant": config.ref_ant,
        "field": config.field,
        "spw": config.spw,
        "scans": config.scans,
        "data_col": config.data_col,
        "model_col": config.model_col,
        "apply_parang": config.apply_parang,
        "rfi_threshold": [s.rfi_threshold for s in config.steps],
        "phase_only": [s.phase_only for s in config.steps],
        "max_iter": [s.max_iter for s in config.steps],
        "tol": [s.tol for s in config.steps],
        "memory_limit_gb": config.memory_limit_gb,
    }
    return yaml.dump(d, default_flow_style=False, sort_keys=False)


def _die(message: str):
    """Print error and exit."""
    print(f"\n  ALAKAZAM CONFIG ERROR: {message}\n", file=sys.stderr)
    sys.exit(1)
