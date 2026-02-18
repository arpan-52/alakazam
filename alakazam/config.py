"""ALAKAZAM Config Parser.

Three top-level blocks: solve, fluxscale, apply.

Parsing rules:
  - spw absent        → all SPWs
  - scans absent      → all scans for that field
  - field scalar      → same field for all steps
  - field [[...], [...]] → per-step groups
  - solution_field absent in apply → interpolation mode picks automatically

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import yaml


# ---------------------------------------------------------------------------
# Interpolation modes
# ---------------------------------------------------------------------------

VALID_INTERP = {"exact", "nearest_time", "nearest_sky", "linear", "cubic"}
VALID_FIELD_SELECT = {"nearest_time", "nearest_sky", "pinned"}


# ---------------------------------------------------------------------------
# Parsed SPW / scan helpers
# ---------------------------------------------------------------------------

def parse_spw_string(s: Optional[str]) -> Optional[List[int]]:
    """Parse '0,1,2' or '0~3' or None → list of ints or None (=all)."""
    if s is None:
        return None
    s = str(s).strip()
    if s in ("", "*"):
        return None
    result = []
    for part in s.split(","):
        part = part.strip()
        if "~" in part:
            lo, hi = part.split("~")
            result.extend(range(int(lo), int(hi) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def parse_scan_string(s: Optional[Union[str, int]]) -> Optional[List[int]]:
    """Parse '1,2,3' or '1~5' or int or None → list of ints or None (=all)."""
    if s is None:
        return None
    if isinstance(s, int):
        return [s]
    s = str(s).strip()
    if s in ("", "*"):
        return None
    result = []
    for part in s.split(","):
        part = part.strip()
        if "~" in part:
            lo, hi = part.split("~")
            result.extend(range(int(lo), int(hi) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def parse_time_interval(s: str) -> float:
    """Parse '2min', '30s', 'inf', '120' → seconds as float."""
    s = str(s).strip().lower()
    if s in ("inf", "infinity", ""):
        return float("inf")
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
    """Parse '4MHz', '128kHz', 'full' → Hz as float or None (=full)."""
    s = str(s).strip().lower()
    if s in ("full", "inf", ""):
        return None
    m = re.match(r"^([0-9.]+)\s*(mhz|khz|ghz|hz)?$", s)
    if not m:
        raise ValueError(f"Cannot parse freq interval: {s!r}")
    val = float(m.group(1))
    unit = m.group(2) or "hz"
    if unit == "ghz":
        return val * 1e9
    if unit == "mhz":
        return val * 1e6
    if unit == "khz":
        return val * 1e3
    return val


# ---------------------------------------------------------------------------
# Field group parsing
# ---------------------------------------------------------------------------

def _parse_field_list(raw) -> List[List[str]]:
    """
    Parse field spec for a list of N steps.

    Accepts:
      scalar string  "3C286"         → [[3C286]] * N  (expanded later by caller)
      flat list      [3C147, 3C286]  → ambiguous: if N=2 treat as one group each,
                                       else wrap as [["3C147","3C286"]] * N
                                       → resolved per-call with n_steps
      nested list    [[3C147], [3C147,3C286]] → as-is

    Returns a list of groups, one group per step. Each group is a list of field names.
    Caller is responsible for expanding scalars to n_steps length.
    """
    if raw is None:
        return []

    # Scalar string
    if isinstance(raw, str):
        return [[f.strip() for f in raw.split(",")]]

    # List
    if isinstance(raw, list):
        # Check if it's nested (first element is a list)
        if raw and isinstance(raw[0], list):
            return [[str(f).strip() for f in grp] for grp in raw]
        else:
            # Flat list — each element is either a field name (str) or a sub-list
            groups = []
            for item in raw:
                if isinstance(item, list):
                    groups.append([str(f).strip() for f in item])
                else:
                    groups.append([str(item).strip()])
            return groups

    return [[str(raw).strip()]]


def _expand_field_groups(groups: List[List[str]], n_steps: int) -> List[List[str]]:
    """Expand field groups to exactly n_steps entries."""
    if not groups:
        return [[""] for _ in range(n_steps)]
    if len(groups) == 1:
        return [groups[0]] * n_steps
    if len(groups) != n_steps:
        raise ValueError(
            f"field has {len(groups)} groups but jones has {n_steps} steps"
        )
    return groups


def _parse_scans_per_step(raw, field_groups: List[List[str]]) -> List[List[Optional[List[int]]]]:
    """
    Parse scans aligned to field_groups structure.

    Returns: per step, per field-in-step → Optional[List[int]] (None=all)

    Accepts:
      None                   → all scans for everything
      scalar "1,2,3"         → same for all
      flat list [1,2,3]      → same for all fields, same for all steps
      nested per-step/field  → matched to field_groups structure
    """
    n_steps = len(field_groups)

    if raw is None:
        return [[None] * len(fg) for fg in field_groups]

    # Scalar string
    if isinstance(raw, (str, int)):
        parsed = parse_scan_string(raw)
        return [[parsed] * len(fg) for fg in field_groups]

    if isinstance(raw, list):
        if not raw:
            return [[None] * len(fg) for fg in field_groups]

        # Check if nested
        if isinstance(raw[0], list):
            # Per-step nested
            result = []
            for step_idx, step_scans in enumerate(raw):
                fg = field_groups[step_idx]
                if isinstance(step_scans, list) and step_scans and isinstance(step_scans[0], list):
                    # Per-field within step
                    result.append([parse_scan_string(sc) for sc in step_scans])
                else:
                    # Same scan spec for all fields in this step
                    parsed = parse_scan_string(
                        ",".join(str(x) for x in step_scans) if isinstance(step_scans, list)
                        else step_scans
                    )
                    result.append([parsed] * len(fg))
            return result
        else:
            # Flat list — apply same to all
            parsed = parse_scan_string(",".join(str(x) for x in raw))
            return [[parsed] * len(fg) for fg in field_groups]

    return [[None] * len(fg) for fg in field_groups]


# ---------------------------------------------------------------------------
# ExternalPreapply
# ---------------------------------------------------------------------------

@dataclass
class ExternalPreapply:
    """External solutions to apply before the internal solve chain starts."""
    tables: List[str]                # H5 files to load from
    jones: List[str]                 # Jones types to load from each table
    fields: Optional[List[Optional[List[str]]]]  # per entry: field list or None
    field_select: List[str]          # nearest_time | nearest_sky
    time_interp: List[str]           # exact | nearest_time | nearest_sky | linear | cubic


# ---------------------------------------------------------------------------
# SolveBlock
# ---------------------------------------------------------------------------

@dataclass
class SolveBlock:
    ms: str
    output: str
    ref_ant: int

    # Per-step Jones config
    jones: List[str]                               # [K, G, G, ...]
    field_groups: List[List[str]]                  # per step: list of fields
    scans_per_step: List[List[Optional[List[int]]]]  # per step, per field: scan ids or None
    spw: Optional[List[int]]                       # None = all

    time_interval: List[float]                     # seconds per step
    freq_interval: List[Optional[float]]           # Hz per step, None=full
    phase_only: List[bool]

    # Pre-apply interpolation for internal chain
    preapply_time_interp: List[str]                # per step

    # External preapply (optional)
    external_preapply: Optional[ExternalPreapply]

    # Data columns
    data_col: str = "DATA"
    model_col: str = "MODEL_DATA"
    apply_parang: bool = False
    rfi_threshold: float = 5.0
    max_iter: int = 100
    tol: float = 1e-10
    memory_limit_gb: float = 0.0


# ---------------------------------------------------------------------------
# FluxscaleBlock
# ---------------------------------------------------------------------------

@dataclass
class FluxscaleBlock:
    reference_table: str
    reference_field: List[str]   # one or more reference fields
    transfer_table: str
    transfer_field: List[str]    # one or more transfer fields
    output: str
    jones_type: str = "G"
    spw: Optional[List[int]] = None   # None = all


# ---------------------------------------------------------------------------
# ApplyTerm
# ---------------------------------------------------------------------------

@dataclass
class ApplyTerm:
    jones: str
    table: str
    field_select: str                     # nearest_time | nearest_sky | pinned
    time_interp: str                      # exact | nearest_time | nearest_sky | linear | cubic
    solution_field: Optional[List[str]]   # only when field_select=pinned


# ---------------------------------------------------------------------------
# ApplyBlock
# ---------------------------------------------------------------------------

@dataclass
class ApplyBlock:
    ms: str
    output_col: str
    target_field: Optional[List[str]]    # None = all fields
    target_scans: Optional[List[int]]    # None = all scans
    spw: Optional[List[int]]             # None = all SPWs

    terms: List[ApplyTerm]

    apply_parang: bool = False


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class AlakazamConfig:
    solve_blocks: List[SolveBlock]
    fluxscale_blocks: List[FluxscaleBlock]
    apply_blocks: List[ApplyBlock]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _ensure_list(x, n: int) -> list:
    """If x is scalar, replicate n times. If list, check length."""
    if not isinstance(x, list):
        return [x] * n
    if len(x) != n:
        raise ValueError(f"Expected list of length {n}, got {len(x)}")
    return x


def _parse_solve_block(d: Dict[str, Any]) -> SolveBlock:
    jones_list = d["jones"]
    n_steps = len(jones_list)

    # Field groups
    raw_field = d.get("field")
    groups = _parse_field_list(raw_field)
    field_groups = _expand_field_groups(groups, n_steps)

    # Scans per step per field
    raw_scans = d.get("scans")
    scans_per_step = _parse_scans_per_step(raw_scans, field_groups)

    # SPW
    spw = parse_spw_string(d.get("spw"))

    # Per-step scalars
    def _step_list(key, default, parser=None):
        raw = d.get(key, default)
        vals = _ensure_list(raw, n_steps)
        if parser:
            vals = [parser(v) for v in vals]
        return vals

    time_interval  = _step_list("time_interval", "inf", parse_time_interval)
    freq_interval  = _step_list("freq_interval", "full", parse_freq_interval)
    phase_only     = _step_list("phase_only", False)
    preapply_interp = _step_list("preapply_time_interp", "nearest_time")

    for m in preapply_interp:
        if m not in VALID_INTERP:
            raise ValueError(f"Invalid preapply_time_interp: {m!r}. Valid: {VALID_INTERP}")

    # External preapply
    ext = None
    ep = d.get("external_preapply")
    if ep:
        ep_tables = ep["tables"]
        ep_jones  = ep["jones"]
        ep_n = len(ep_tables)
        ep_field_select = _ensure_list(ep.get("field_select", "nearest_time"), ep_n)
        ep_time_interp  = _ensure_list(ep.get("time_interp", "nearest_time"), ep_n)
        raw_ep_fields = ep.get("fields")
        if raw_ep_fields is not None:
            ep_fields_parsed = []
            for item in raw_ep_fields:
                if item is None:
                    ep_fields_parsed.append(None)
                elif isinstance(item, list):
                    ep_fields_parsed.append([str(f).strip() for f in item])
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
        ref_ant=int(d.get("ref_ant", 0)),
        jones=jones_list,
        field_groups=field_groups,
        scans_per_step=scans_per_step,
        spw=spw,
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
    )


def _parse_fluxscale_block(d: Dict[str, Any]) -> FluxscaleBlock:
    def _to_list(x):
        if isinstance(x, list):
            return [str(f).strip() for f in x]
        return [str(x).strip()]

    return FluxscaleBlock(
        reference_table=d["reference_table"],
        reference_field=_to_list(d["reference_field"]),
        transfer_table=d["transfer_table"],
        transfer_field=_to_list(d["transfer_field"]),
        output=d["output"],
        jones_type=d.get("jones_type", "G"),
        spw=parse_spw_string(d.get("spw")),
    )


def _parse_apply_block(d: Dict[str, Any]) -> ApplyBlock:
    jones_list  = d["jones"]
    tables      = d["tables"]
    n = len(jones_list)

    if len(tables) != n:
        raise ValueError(f"apply: jones has {n} entries but tables has {len(tables)}")

    raw_fs = d.get("field_select", "nearest_time")
    raw_ti = d.get("time_interp", "nearest_time")
    raw_sf = d.get("solution_field")

    field_select   = _ensure_list(raw_fs, n)
    time_interp    = _ensure_list(raw_ti, n)
    sol_field_raw  = _ensure_list(raw_sf, n) if raw_sf is not None else [None] * n

    terms = []
    for i in range(n):
        fs = field_select[i]
        ti = time_interp[i]
        if fs not in VALID_FIELD_SELECT:
            raise ValueError(f"Invalid field_select: {fs!r}. Valid: {VALID_FIELD_SELECT}")
        if ti not in VALID_INTERP:
            raise ValueError(f"Invalid time_interp: {ti!r}. Valid: {VALID_INTERP}")

        sf_raw = sol_field_raw[i]
        if sf_raw is not None:
            if isinstance(sf_raw, list):
                sf = [str(x).strip() for x in sf_raw]
            else:
                sf = [str(sf_raw).strip()]
        else:
            sf = None

        if fs == "pinned" and sf is None:
            raise ValueError(f"apply term {i} ({jones_list[i]}): field_select=pinned requires solution_field")

        terms.append(ApplyTerm(
            jones=jones_list[i],
            table=tables[i],
            field_select=fs,
            time_interp=ti,
            solution_field=sf,
        ))

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
        spw=parse_spw_string(d.get("spw")),
        terms=terms,
        apply_parang=bool(d.get("apply_parang", False)),
    )


def load_config(path: str) -> AlakazamConfig:
    """Load and parse a YAML config file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    solve_blocks = [_parse_solve_block(b) for b in (raw.get("solve") or [])]
    fluxscale_blocks = [_parse_fluxscale_block(b) for b in (raw.get("fluxscale") or [])]
    apply_blocks = [_parse_apply_block(b) for b in (raw.get("apply") or [])]

    return AlakazamConfig(
        solve_blocks=solve_blocks,
        fluxscale_blocks=fluxscale_blocks,
        apply_blocks=apply_blocks,
    )
