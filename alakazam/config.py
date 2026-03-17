"""ALAKAZAM v1 Config Parser.

Three top-level blocks: solve, fluxscale, apply.  All optional.

Jones names: K, G, D, KC, CP.

SPW syntax (CASA-native):
  0              -> SPW 0 all channels
  0:127~156      -> SPW 0 channels 127-156 inclusive
  0:127~156,1,2,5:10~20  -> mixed

Solver backends: jax_scipy (default), torch_lbfgs

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_JONES = {"K", "G", "D", "KC", "CP"}
VALID_BACKENDS = {"jax_scipy", "torch_lbfgs"}
VALID_TIME_INTERP = {"exact", "nearest", "linear", "cubic"}
VALID_FIELD_SELECT = {"nearest_time", "nearest_sky", "pinned"}

# ---------------------------------------------------------------------------
# SPW + channel selection (CASA-native syntax)
# ---------------------------------------------------------------------------

@dataclass
class SPWSelection:
    """One SPW with optional channel range."""
    spw: int
    chan_start: Optional[int] = None
    chan_end: Optional[int] = None


def parse_spw_selection(raw: Optional[str]) -> Optional[List[SPWSelection]]:
    """Parse CASA-native SPW:channel string.

    Examples
    --------
    None / "" / "*"           -> None  (all SPWs all channels)
    "0"                       -> [SPWSelection(0)]
    "0:127~156"               -> [SPWSelection(0, 127, 156)]
    "0:127~156,1,2,5:10~20"  -> [SPWSelection(0,127,156), SPWSelection(1),
                                   SPWSelection(2), SPWSelection(5,10,20)]
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if s in ("", "*"):
        return None

    result: List[SPWSelection] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            spw_str, chan_str = part.split(":", 1)
            spw_id = int(spw_str.strip())
            chan_str = chan_str.strip()
            if "~" in chan_str:
                lo, hi = chan_str.split("~", 1)
                result.append(SPWSelection(spw_id, int(lo.strip()),
                                           int(hi.strip())))
            else:
                ch = int(chan_str)
                result.append(SPWSelection(spw_id, ch, ch))
        elif "~" in part:
            lo, hi = part.split("~", 1)
            for spw_id in range(int(lo.strip()), int(hi.strip()) + 1):
                result.append(SPWSelection(spw_id))
        else:
            result.append(SPWSelection(int(part)))
    return result if result else None


def spw_ids_from_selection(sel: Optional[List[SPWSelection]]) -> Optional[List[int]]:
    """Extract unique sorted SPW indices from selection."""
    if sel is None:
        return None
    return sorted(set(s.spw for s in sel))


def chan_slice_for_spw(sel: Optional[List[SPWSelection]], spw: int,
                       n_chan: int) -> slice:
    """Return channel slice for a given SPW.  None selection -> slice(None)."""
    if sel is None:
        return slice(None)
    for s in sel:
        if s.spw == spw and s.chan_start is not None:
            start = max(0, s.chan_start)
            end = min(n_chan, s.chan_end + 1)
            return slice(start, end)
    return slice(None)


# ---------------------------------------------------------------------------
# Scan selection
# ---------------------------------------------------------------------------

def parse_scan_string(s: Optional[Union[str, int, list]]) -> Optional[List[int]]:
    """Parse scan selection -> sorted list of ints or None (all)."""
    if s is None:
        return None
    if isinstance(s, int):
        return [s]
    if isinstance(s, list):
        out: List[int] = []
        for item in s:
            if isinstance(item, int):
                out.append(item)
            else:
                parsed = parse_scan_string(item)
                if parsed:
                    out.extend(parsed)
        return sorted(set(out)) if out else None
    s = str(s).strip()
    if s in ("", "*"):
        return None
    result: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if "~" in part:
            lo, hi = part.split("~", 1)
            result.extend(range(int(lo.strip()), int(hi.strip()) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


# ---------------------------------------------------------------------------
# Ref antenna parsing
# ---------------------------------------------------------------------------

def _parse_ref_ant(val) -> Any:
    """Accept int index or string antenna name.
    Returns int if numeric, str if name — resolved to index in flow.py."""
    if isinstance(val, int):
        return val
    s = str(val).strip()
    try:
        return int(s)
    except ValueError:
        return s  # antenna name, resolved later against ANTENNA table


def resolve_ref_ant(ref_ant, ant_names: List[str]) -> int:
    """Resolve ref_ant (int or str name) to integer index.
    Raises ValueError if name not found."""
    if isinstance(ref_ant, int):
        if ref_ant < 0 or ref_ant >= len(ant_names):
            raise ValueError(f"ref_ant={ref_ant} out of range [0, {len(ant_names)})")
        return ref_ant
    # String name
    for i, name in enumerate(ant_names):
        if name.strip() == ref_ant.strip():
            return i
    raise ValueError(
        f"ref_ant={ref_ant!r} not found in ANTENNA table. "
        f"Available: {ant_names}")


# ---------------------------------------------------------------------------
# Time / freq interval parsing
# ---------------------------------------------------------------------------

def parse_time_interval(s: str) -> float:
    """Parse '2min', '30s', 'inf', 'scan', '120' -> seconds.

    'scan' returns -1.0 as sentinel — flow.py uses scan boundaries.
    """
    s = str(s).strip().lower()
    if s in ("inf", "infinity", ""):
        return float("inf")
    if s == "scan":
        return -1.0  # sentinel: means "one solution per scan"
    m = re.match(r"^([0-9.]+)\s*(min|m|s|sec|h|hr)?$", s)
    if not m:
        raise ValueError(f"Cannot parse time interval: {s!r}")
    val = float(m.group(1))
    unit = m.group(2) or "s"
    if unit in ("min", "m"):
        return val * 60.0
    if unit in ("h", "hr"):
        return val * 3600.0
    return val


def parse_freq_interval(s: str) -> Optional[float]:
    """Parse '4MHz', '128kHz', 'full', '32' -> Hz or None (full spw)."""
    s = str(s).strip().lower()
    if s in ("full", "inf", ""):
        return None
    m = re.match(r"^([0-9.]+)\s*(ghz|mhz|khz|hz)?$", s)
    if not m:
        raise ValueError(f"Cannot parse freq interval: {s!r}")
    val = float(m.group(1))
    unit = m.group(2) or "hz"
    mult = {"ghz": 1e9, "mhz": 1e6, "khz": 1e3, "hz": 1.0}
    return val * mult[unit]


# ---------------------------------------------------------------------------
# Field group parsing
# ---------------------------------------------------------------------------

def _parse_field_list(raw) -> List[List[str]]:
    """Parse field spec into list of groups (one group per step)."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [[f.strip() for f in raw.split(",")]]
    if isinstance(raw, list):
        if raw and isinstance(raw[0], list):
            return [[str(f).strip() for f in grp] for grp in raw]
        groups: List[List[str]] = []
        for item in raw:
            if isinstance(item, list):
                groups.append([str(f).strip() for f in item])
            else:
                groups.append([str(item).strip()])
        return groups
    return [[str(raw).strip()]]


def _expand_field_groups(groups: List[List[str]],
                         n_steps: int) -> List[List[str]]:
    """Expand to exactly n_steps entries."""
    if not groups:
        return [[""] for _ in range(n_steps)]
    if len(groups) == 1:
        return [groups[0][:] for _ in range(n_steps)]
    if len(groups) != n_steps:
        raise ValueError(
            f"field has {len(groups)} groups but jones has {n_steps} steps")
    return groups


def _parse_scans_per_step(
    raw, field_groups: List[List[str]]
) -> List[List[Optional[List[int]]]]:
    """Parse scans aligned to field_groups structure."""
    n_steps = len(field_groups)
    if raw is None:
        return [[None] * len(fg) for fg in field_groups]
    if isinstance(raw, (str, int)):
        parsed = parse_scan_string(raw)
        return [[parsed] * len(fg) for fg in field_groups]
    if isinstance(raw, list):
        if not raw:
            return [[None] * len(fg) for fg in field_groups]
        if isinstance(raw[0], list):
            result = []
            for step_idx, step_scans in enumerate(raw):
                fg = field_groups[step_idx]
                if (isinstance(step_scans, list) and step_scans
                        and isinstance(step_scans[0], list)):
                    result.append([parse_scan_string(sc) for sc in step_scans])
                else:
                    parsed = parse_scan_string(step_scans)
                    result.append([parsed] * len(fg))
            return result
        else:
            parsed = parse_scan_string(raw)
            return [[parsed] * len(fg) for fg in field_groups]
    return [[None] * len(fg) for fg in field_groups]


# ---------------------------------------------------------------------------
# ExternalPreapply
# ---------------------------------------------------------------------------

@dataclass
class ExternalPreapply:
    """External solutions applied before internal chain starts."""
    tables: List[str]
    jones: List[str]
    fields: List[Optional[List[str]]]
    field_select: List[str]
    time_interp: List[str]


# ---------------------------------------------------------------------------
# SolveBlock
# ---------------------------------------------------------------------------

@dataclass
class SolveBlock:
    ms: str
    output: str
    ref_ant: Any  # int index or str antenna name — resolved to int in flow

    jones: List[str]
    field_groups: List[List[str]]
    scans_per_step: List[List[Optional[List[int]]]]

    # Global SPW selection (applies to all steps unless step_spw overrides)
    spw: Optional[List[SPWSelection]] = None
    # Per-step SPW selection (overrides global when present)
    step_spw: Optional[List[Optional[List[SPWSelection]]]] = None

    time_interval: List[float] = field(default_factory=list)
    freq_interval: List[Optional[float]] = field(default_factory=list)
    phase_only: List[bool] = field(default_factory=list)
    preapply_time_interp: List[str] = field(default_factory=list)
    external_preapply: Optional[ExternalPreapply] = None

    data_col: str = "DATA"
    model_col: str = "MODEL_DATA"
    apply_parang: bool = False
    rfi_threshold: float = 5.0
    max_iter: int = 100
    tol: float = 1e-10
    memory_limit_gb: float = 0.0

    solver_backend: str = "jax_scipy"
    n_workers: int = 0          # 0 = auto
    gpu: bool = False           # auto-detect; True = force GPU


# ---------------------------------------------------------------------------
# FluxscaleBlock
# ---------------------------------------------------------------------------

@dataclass
class FluxscaleBlock:
    reference_table: str
    reference_field: List[str]
    transfer_table: str
    transfer_field: List[str]
    output: str
    jones_type: str = "G"
    spw: Optional[List[SPWSelection]] = None


# ---------------------------------------------------------------------------
# ApplyTerm
# ---------------------------------------------------------------------------

@dataclass
class ApplyTerm:
    jones: str
    table: str
    field_select: str
    time_interp: str
    solution_field: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# ApplyBlock
# ---------------------------------------------------------------------------

@dataclass
class ApplyBlock:
    ms: str
    output_col: str
    target_field: Optional[List[str]]
    target_scans: Optional[List[int]]
    spw: Optional[List[SPWSelection]]
    terms: List[ApplyTerm]
    apply_parang: bool = False
    propagate_flags: bool = False  # write solution flags to MS FLAG column


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class AlakazamConfig:
    solve_blocks: List[SolveBlock]
    fluxscale_blocks: List[FluxscaleBlock]
    apply_blocks: List[ApplyBlock]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_list(x, n: int) -> list:
    if not isinstance(x, list):
        return [x] * n
    if len(x) != n:
        raise ValueError(f"Expected list of length {n}, got {len(x)}")
    return x


def _validate_jones(jones_list: List[str]) -> None:
    for j in jones_list:
        if j not in VALID_JONES:
            raise ValueError(
                f"Unknown Jones type: {j!r}. Valid: {sorted(VALID_JONES)}")


def _validate_backend(backend: str) -> None:
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"Unknown solver_backend: {backend!r}. "
            f"Valid: {sorted(VALID_BACKENDS)}")


# ---------------------------------------------------------------------------
# Block parsers
# ---------------------------------------------------------------------------

def _parse_solve_block(d: Dict[str, Any]) -> SolveBlock:
    jones_list = d["jones"]
    _validate_jones(jones_list)
    n_steps = len(jones_list)

    # Field groups
    groups = _parse_field_list(d.get("field"))
    field_groups = _expand_field_groups(groups, n_steps)

    # Scans per step per field
    scans_per_step = _parse_scans_per_step(d.get("scans"), field_groups)

    # Global SPW
    spw = parse_spw_selection(d.get("spw"))

    # Per-step SPW
    raw_step_spw = d.get("step_spw")
    step_spw: Optional[List[Optional[List[SPWSelection]]]] = None
    if raw_step_spw is not None:
        step_spw_list = _ensure_list(raw_step_spw, n_steps)
        step_spw = [parse_spw_selection(s) for s in step_spw_list]

    # Per-step scalars
    def _step_list(key, default, parser=None):
        raw = d.get(key, default)
        vals = _ensure_list(raw, n_steps)
        if parser:
            vals = [parser(v) for v in vals]
        return vals

    time_interval = _step_list("time_interval", "inf", parse_time_interval)
    freq_interval = _step_list("freq_interval", "full", parse_freq_interval)
    phase_only = _step_list("phase_only", False)
    preapply_interp = _step_list("preapply_time_interp", "nearest")

    for m in preapply_interp:
        if m not in VALID_TIME_INTERP:
            raise ValueError(
                f"Invalid preapply_time_interp: {m!r}. "
                f"Valid: {sorted(VALID_TIME_INTERP)}")

    # Solver backend
    backend = d.get("solver_backend", "jax_scipy")
    _validate_backend(backend)

    # External preapply
    ext = None
    ep = d.get("external_preapply")
    if ep:
        ep_tables = ep["tables"]
        ep_jones = ep["jones"]
        _validate_jones(ep_jones)
        ep_n = len(ep_tables)
        ep_field_select = _ensure_list(
            ep.get("field_select", "nearest_time"), ep_n)
        ep_time_interp = _ensure_list(
            ep.get("time_interp", "nearest"), ep_n)
        raw_ep_fields = ep.get("fields")
        if raw_ep_fields is not None:
            ep_fields_parsed: List[Optional[List[str]]] = []
            for item in raw_ep_fields:
                if item is None:
                    ep_fields_parsed.append(None)
                elif isinstance(item, list):
                    ep_fields_parsed.append(
                        [str(f).strip() for f in item])
                else:
                    ep_fields_parsed.append([str(item).strip()])
        else:
            ep_fields_parsed = [None] * ep_n
        ext = ExternalPreapply(
            tables=ep_tables,
            jones=ep_jones,
            fields=ep_fields_parsed,
            field_select=ep_field_select,
            time_interp=ep_time_interp,
        )

    return SolveBlock(
        ms=d["ms"],
        output=d["output"],
        ref_ant=_parse_ref_ant(d.get("ref_ant", 0)),
        jones=jones_list,
        field_groups=field_groups,
        scans_per_step=scans_per_step,
        spw=spw,
        step_spw=step_spw,
        time_interval=time_interval,
        freq_interval=freq_interval,
        phase_only=phase_only,
        preapply_time_interp=preapply_interp,
        external_preapply=ext,
        data_col=d.get("data_col", "DATA"),
        model_col=d.get("model_col", "MODEL_DATA"),
        apply_parang=bool(d.get("apply_parang", False)),
        rfi_threshold=float(d.get("rfi_threshold", 5.0)),
        max_iter=int(d.get("max_iter", 100)),
        tol=float(d.get("tol", 1e-10)),
        memory_limit_gb=float(d.get("memory_limit_gb", 0.0)),
        solver_backend=backend,
        n_workers=int(d.get("n_workers", 0)),
        gpu=bool(d.get("gpu", False)),
    )


def _parse_fluxscale_block(d: Dict[str, Any]) -> FluxscaleBlock:
    def _to_list(x):
        if isinstance(x, list):
            return [str(f).strip() for f in x]
        return [str(x).strip()]

    jtype = d.get("jones_type", "G")
    if jtype not in VALID_JONES:
        raise ValueError(f"fluxscale jones_type {jtype!r} not valid")

    return FluxscaleBlock(
        reference_table=d["reference_table"],
        reference_field=_to_list(d["reference_field"]),
        transfer_table=d["transfer_table"],
        transfer_field=_to_list(d["transfer_field"]),
        output=d["output"],
        jones_type=jtype,
        spw=parse_spw_selection(d.get("spw")),
    )


def _parse_apply_block(d: Dict[str, Any]) -> ApplyBlock:
    jones_list = d["jones"]
    _validate_jones(jones_list)
    tables = d["tables"]
    n = len(jones_list)

    if len(tables) != n:
        raise ValueError(
            f"apply: jones has {n} entries but tables has {len(tables)}")

    raw_fs = d.get("field_select", "nearest_time")
    raw_ti = d.get("time_interp", "nearest")
    raw_sf = d.get("solution_field")

    field_select = _ensure_list(raw_fs, n)
    time_interp = _ensure_list(raw_ti, n)
    sol_field_raw = (_ensure_list(raw_sf, n) if raw_sf is not None
                     else [None] * n)

    terms: List[ApplyTerm] = []
    for i in range(n):
        fs = field_select[i]
        ti = time_interp[i]
        if fs not in VALID_FIELD_SELECT:
            raise ValueError(
                f"Invalid field_select: {fs!r}. "
                f"Valid: {sorted(VALID_FIELD_SELECT)}")
        if ti not in VALID_TIME_INTERP:
            raise ValueError(
                f"Invalid time_interp: {ti!r}. "
                f"Valid: {sorted(VALID_TIME_INTERP)}")
        sf_raw = sol_field_raw[i]
        sf: Optional[List[str]] = None
        if sf_raw is not None:
            if isinstance(sf_raw, list):
                sf = [str(x).strip() for x in sf_raw]
            else:
                sf = [str(sf_raw).strip()]
        if fs == "pinned" and sf is None:
            raise ValueError(
                f"apply term {i} ({jones_list[i]}): "
                f"field_select=pinned requires solution_field")
        terms.append(ApplyTerm(
            jones=jones_list[i], table=tables[i],
            field_select=fs, time_interp=ti, solution_field=sf))

    # Target field
    raw_tf = d.get("target_field")
    if raw_tf is None:
        target_field = None
    elif isinstance(raw_tf, list):
        target_field = [str(f).strip() for f in raw_tf]
    else:
        target_field = [f.strip() for f in str(raw_tf).split(",")]

    return ApplyBlock(
        ms=d["ms"],
        output_col=d.get("output_col", "CORRECTED_DATA"),
        target_field=target_field,
        target_scans=parse_scan_string(d.get("target_scans")),
        spw=parse_spw_selection(d.get("spw")),
        terms=terms,
        apply_parang=bool(d.get("apply_parang", False)),
        propagate_flags=bool(d.get("propagate_flags", False)),
    )


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> AlakazamConfig:
    """Load and parse a YAML config file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    solve_blocks = [_parse_solve_block(b)
                    for b in (raw.get("solve") or [])]
    fluxscale_blocks = [_parse_fluxscale_block(b)
                        for b in (raw.get("fluxscale") or [])]
    apply_blocks = [_parse_apply_block(b)
                    for b in (raw.get("apply") or [])]

    return AlakazamConfig(
        solve_blocks=solve_blocks,
        fluxscale_blocks=fluxscale_blocks,
        apply_blocks=apply_blocks,
    )
