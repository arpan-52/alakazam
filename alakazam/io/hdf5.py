"""ALAKAZAM v1 HDF5 I/O.

Universal solution schema:

  jones:  (n_ant, n_freq, n_time, 2, 2) complex128
  flags:  (n_ant, n_freq, n_time)         bool
  time:   (n_time,) float64               MJD seconds
  freq:   (n_freq,) float64               Hz

  Per-cell solver stats:
  converged: (n_freq, n_time) bool
  n_iter:    (n_freq, n_time) int32
  cost:      (n_freq, n_time) float64

  Hierarchy:
    {jones_key}/field_{name}/scan_{id}/spw_{n}/
      jones, flags, time, freq, solver_stats/
      attrs: jones_type, field_name, scan_id, spw, n_ant,
             field_ra, field_dec, matrix_form, ref_ant_constraint,
             ref_ant, ref_ant_name, ant_names, solint_s, freqint_hz,
             phase_only, feed_basis, solver_backend, apply_parang,
             ms, preapply_chain

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations
import json, os, time, logging
from typing import Any, Dict, List, Optional
import h5py
import numpy as np

logger = logging.getLogger("alakazam")
HDF5_VERSION = "1.0"

MATRIX_FORMS = {
    "K":  "diag(exp(-2pi*i*tau_p*nu), exp(-2pi*i*tau_q*nu))  per-ant",
    "G":  "diag(g_p*exp(i*phi_p), g_q*exp(i*phi_q))  per-ant",
    "D":  "[[1, d_pq], [d_qp, 1]]  per-ant",
    "KC": "diag(exp(-2pi*i*tau_cross*nu), 1)  global",
    "CP": "diag(1, exp(i*phi_cross))  global",
}

REF_ANT_CONSTRAINTS = {
    "K":  "delay[ref,:]=0",
    "G":  "phase[ref,:]=0, amp free",
    "D":  "d_pq[ref]=0, d_qp[ref] free",
    "KC": "none (single global param)",
    "CP": "none (single global param)",
}


def _fk(name): return f"field_{name.replace(' ', '_')}"
def _sck(scan_id): return f"scan_{scan_id}"
def _sk(spw): return f"spw_{spw}"
def _exists(p): return os.path.exists(p)


def save_solutions(
    path: str, jones_type: str, field_name: str, spw: int,
    jones: np.ndarray,          # (n_ant, n_freq, n_time, 2, 2)
    flags: np.ndarray,          # (n_ant, n_freq, n_time)
    times: np.ndarray,          # (n_time,) MJD s
    freqs: np.ndarray,          # (n_freq,) Hz
    solver_stats: Dict,         # {converged, n_iter, cost} each (n_freq, n_time)
    meta: Dict[str, Any],
    scan_id: int = 0,
    provenance: Optional[Dict] = None,
    delay: Optional[np.ndarray] = None,  # (n_ant, n_freq, n_time, 2) ns — K/KC only
) -> None:
    """Write solutions in universal schema with scan-level hierarchy."""
    mode = "a" if _exists(path) else "w"
    with h5py.File(path, mode) as f:
        if "version" not in f.attrs:
            f.attrs["version"] = HDF5_VERSION
            f.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            f.attrs["alakazam_version"] = "1.0.0"

        if provenance:
            pg = f.require_group("metadata")
            for k, v in provenance.items():
                pg.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else str(v)

        jkey = f"{jones_type}/{_fk(field_name)}/{_sck(scan_id)}/{_sk(spw)}"
        if jkey in f:
            del f[jkey]
        g = f.require_group(jkey)

        # Data arrays
        g.create_dataset("jones", data=jones, compression="gzip", compression_opts=4)
        g.create_dataset("flags", data=flags)
        g.create_dataset("time", data=times)
        g.create_dataset("freq", data=freqs)

        # Delay (K/KC only): (n_ant, n_freq, n_time, 2) in nanoseconds
        if delay is not None:
            g.create_dataset("delay", data=delay, compression="gzip", compression_opts=4)

        # Solver stats per cell
        if solver_stats:
            sg = g.require_group("solver_stats")
            for k, v in solver_stats.items():
                if isinstance(v, np.ndarray):
                    sg.create_dataset(k, data=v)

        # Metadata
        base_type = jones_type.rstrip("0123456789")
        g.attrs["jones_type"] = jones_type
        g.attrs["field_name"] = field_name
        g.attrs["scan_id"] = scan_id
        g.attrs["spw"] = spw
        g.attrs["n_ant"] = jones.shape[0]
        g.attrs["n_freq"] = jones.shape[1]
        g.attrs["n_time"] = jones.shape[2]
        g.attrs["matrix_form"] = MATRIX_FORMS.get(base_type, "unknown")
        g.attrs["ref_ant_constraint"] = REF_ANT_CONSTRAINTS.get(base_type, "unknown")
        for k, v in meta.items():
            if isinstance(v, (dict, list)):
                g.attrs[k] = json.dumps(v)
            elif isinstance(v, (np.integer, np.floating)):
                g.attrs[k] = v.item()
            elif v is not None:
                g.attrs[k] = v


def load_solutions(path: str, jones_type: str,
                   field_name: str, spw: int,
                   scan_id: Optional[int] = None) -> Dict[str, Any]:
    """Load solutions.  If scan_id given, load one scan; otherwise concatenate all scans."""
    with h5py.File(path, "r") as f:
        fk = _fk(field_name)
        sk = _sk(spw)

        if scan_id is not None:
            # Load a single scan
            key = f"{jones_type}/{fk}/{_sck(scan_id)}/{sk}"
            if key not in f:
                raise KeyError(f"Not found: {key} in {path}")
            return _load_group(f[key])

        # Try new scan-level hierarchy: concatenate all scans
        field_key = f"{jones_type}/{fk}"
        if field_key not in f:
            # Try legacy flat hierarchy
            key = f"{jones_type}/{fk}/{sk}"
            if key not in f:
                raise KeyError(f"Not found: {field_key} in {path}")
            return _load_group(f[key])

        fg = f[field_key]
        scan_keys = sorted([k for k in fg.keys() if k.startswith("scan_")],
                           key=lambda x: int(x[5:]))
        if not scan_keys:
            # Legacy: spw directly under field
            if sk in fg:
                return _load_group(fg[sk])
            raise KeyError(f"No scans or SPWs found under {field_key}")

        # Concatenate across scans
        parts = []
        for sck in scan_keys:
            if sk not in fg[sck]:
                continue
            parts.append(_load_group(fg[sck][sk]))

        if not parts:
            raise KeyError(f"SPW {spw} not found in any scan under {field_key}")
        if len(parts) == 1:
            return parts[0]

        # Concatenate along time axis (axis=2 for jones, axis=0 for time)
        sort_idx = np.argsort(np.concatenate([p["time"] for p in parts]))
        jones = np.concatenate([p["jones"] for p in parts], axis=2)[:, :, sort_idx]
        flags = np.concatenate([p["flags"] for p in parts], axis=2)[:, :, sort_idx]
        times = np.concatenate([p["time"] for p in parts])[sort_idx]
        freqs = parts[0]["freq"]
        attrs = parts[0]["attrs"]

        # Delay (K/KC only)
        delay = None
        if parts[0]["delay"] is not None:
            delay = np.concatenate([p["delay"] for p in parts], axis=2)[:, :, sort_idx]

        # Merge solver stats
        solver_stats = {}
        if parts[0]["solver_stats"]:
            for k in parts[0]["solver_stats"]:
                solver_stats[k] = np.concatenate(
                    [p["solver_stats"][k] for p in parts if k in p["solver_stats"]],
                    axis=1)[:, sort_idx]

        ra = float(attrs.get("field_ra", 0.0))
        dec = float(attrs.get("field_dec", 0.0))
        return {"jones": jones, "flags": flags, "time": times, "freq": freqs,
                "delay": delay, "attrs": attrs, "solver_stats": solver_stats,
                "ra_rad": ra, "dec_rad": dec}


def _load_group(g) -> Dict[str, Any]:
    """Load datasets from a single solution group."""
    jones = g["jones"][:]
    flags = g["flags"][:] if "flags" in g else None
    times = g["time"][:]
    freqs = g["freq"][:] if "freq" in g else None
    delay = g["delay"][:] if "delay" in g else None
    attrs = dict(g.attrs)
    solver_stats = {}
    if "solver_stats" in g:
        for k in g["solver_stats"].keys():
            solver_stats[k] = g["solver_stats"][k][:]
    ra = float(attrs.get("field_ra", 0.0))
    dec = float(attrs.get("field_dec", 0.0))
    return {"jones": jones, "flags": flags, "time": times, "freq": freqs,
            "delay": delay, "attrs": attrs, "solver_stats": solver_stats,
            "ra_rad": ra, "dec_rad": dec}


def _resolve_jones_key(f, jones_type):
    """Resolve jones key: exact match first, then prefix match.

    Solve writes numbered keys (K0, G0, G1). Apply config may use bare
    types (K, G). This resolves "K" → "K0" when exact match fails.
    """
    if jones_type in f:
        return jones_type
    matches = sorted(k for k in f.keys()
                     if k.startswith(jones_type) and k != "metadata")
    if matches:
        logger.debug(f"jones key {jones_type!r} resolved to {matches[0]!r}")
        return matches[0]
    return None


def load_all_fields(path: str, jones_type: str, spw: int,
                    field_names: Optional[List[str]] = None) -> Dict[str, Dict]:
    """Load all fields for a jones type/spw, concatenating scans."""
    result = {}
    with h5py.File(path, "r") as f:
        resolved = _resolve_jones_key(f, jones_type)
        if resolved is None:
            return {}
        jg = f[resolved]
        for fkey in jg.keys():
            if not fkey.startswith("field_"):
                continue
            # Infer field name from first available scan/spw group attrs
            fname = _infer_field_name(jg[fkey], fkey)
            if field_names is not None and fname not in field_names:
                continue
            try:
                sol = load_solutions(path, resolved, fname, spw)
            except KeyError:
                continue
            result[fname] = sol
    return result


def _infer_field_name(field_group, fkey):
    """Get field_name from attrs, checking scan subgroups or legacy layout."""
    for sck in field_group.keys():
        sg = field_group[sck]
        if isinstance(sg, h5py.Group):
            # Check if this is a scan group with spw subgroups
            for sk in sg.keys():
                if isinstance(sg[sk], h5py.Group) and "jones" in sg[sk]:
                    return sg[sk].attrs.get("field_name", fkey[len("field_"):])
            # Legacy: spw directly under field
            if "jones" in sg:
                return sg.attrs.get("field_name", fkey[len("field_"):])
    return fkey[len("field_"):]


def list_jones_types(path):
    with h5py.File(path, "r") as f:
        return [k for k in f.keys() if k not in ("metadata", "fluxscale")]

def list_spws(path, jones_type, field_name):
    """List SPW ids, handling scan-level hierarchy."""
    with h5py.File(path, "r") as f:
        fk = f"{jones_type}/{_fk(field_name)}"
        if fk not in f: return []
        fg = f[fk]
        spws = set()
        for key in fg.keys():
            if key.startswith("spw_"):
                # Legacy layout
                spws.add(int(key[4:]))
            elif key.startswith("scan_"):
                # New scan-level layout
                for sk in fg[key].keys():
                    if sk.startswith("spw_"):
                        spws.add(int(sk[4:]))
        return sorted(spws)

def copy_solutions(src, dst):
    mode = "a" if _exists(dst) else "w"
    with h5py.File(src, "r") as s, h5py.File(dst, mode) as d:
        for k in s.keys():
            if k not in d:
                s.copy(k, d)

def rescale_solutions(path, jones_type, field_name, spw, scale_p, scale_q):
    """Rescale jones solutions across all scans for a field/spw."""
    with h5py.File(path, "a") as f:
        fk = f"{jones_type}/{_fk(field_name)}"
        if fk not in f:
            raise KeyError(f"Not found: {fk}")
        fg = f[fk]
        rescaled = False
        for key in fg.keys():
            if key.startswith("scan_"):
                sk = _sk(spw)
                if sk in fg[key] and "jones" in fg[key][sk]:
                    jpath = f"{fk}/{key}/{sk}/jones"
                    j = f[jpath][:]
                    j[..., 0, 0] *= np.sqrt(scale_p)
                    j[..., 1, 1] *= np.sqrt(scale_q)
                    f[jpath][...] = j
                    rescaled = True
            elif key == _sk(spw) and "jones" in fg[key]:
                # Legacy layout
                jpath = f"{fk}/{key}/jones"
                j = f[jpath][:]
                j[..., 0, 0] *= np.sqrt(scale_p)
                j[..., 1, 1] *= np.sqrt(scale_q)
                f[jpath][...] = j
                rescaled = True
        if not rescaled:
            raise KeyError(f"No jones data found for {fk}/spw_{spw}")

def save_fluxscale(path, transfer_field, spw, scale_p, scale_q,
                   scatter_p, scatter_q, n_ant, reference_field,
                   reference_table, jones_type):
    mode = "a" if _exists(path) else "w"
    with h5py.File(path, mode) as f:
        fk = f"fluxscale/{_fk(transfer_field)}/{_sk(spw)}"
        if fk in f: del f[fk]
        g = f.require_group(fk)
        for k, v in [("reference_field", reference_field),
                      ("reference_table", reference_table),
                      ("jones_type", jones_type),
                      ("transfer_field", transfer_field),
                      ("scale_p", scale_p), ("scale_q", scale_q),
                      ("scatter_p", scatter_p), ("scatter_q", scatter_q),
                      ("n_ant", n_ant)]:
            g.attrs[k] = v

def print_summary(path):
    with h5py.File(path, "r") as f:
        print(f"\n=== {path} ===")
        print(f"  Version: {f.attrs.get('version', '?')}")
        for jt in list_jones_types(path):
            if jt not in f: continue
            for fkey in f[jt].keys():
                if not fkey.startswith("field_"): continue
                fg = f[jt][fkey]
                for sck in sorted(fg.keys()):
                    sg = fg[sck]
                    if sck.startswith("scan_"):
                        for sk in sorted(sg.keys()):
                            if not sk.startswith("spw_"): continue
                            g = sg[sk]
                            if "jones" not in g: continue
                            s = g["jones"].shape
                            delay_str = "  delay=yes" if "delay" in g else ""
                            print(f"  [{jt}] {fkey} {sck} {sk}  shape={s}  "
                                  f"n_ant={s[0]} n_freq={s[1]} n_time={s[2]}{delay_str}")
                            print(f"    matrix: {g.attrs.get('matrix_form', '')}")
                            print(f"    backend: {g.attrs.get('solver_backend', '?')}")
                    elif sck.startswith("spw_") and "jones" in sg:
                        # Legacy layout
                        s = sg["jones"].shape
                        print(f"  [{jt}] {fkey} {sck}  shape={s}  "
                              f"n_ant={s[0]} n_freq={s[1]} n_time={s[2]}")
                        print(f"    matrix: {sg.attrs.get('matrix_form', '')}")
                        print(f"    backend: {sg.attrs.get('solver_backend', '?')}")
        if "fluxscale" in f:
            print("  Fluxscale:")
            for fk in f["fluxscale"].keys():
                for sk in f["fluxscale"][fk].keys():
                    a = f["fluxscale"][fk][sk].attrs
                    print(f"    {a.get('transfer_field','?')} <- "
                          f"{a.get('reference_field','?')}  "
                          f"scale_p={a.get('scale_p',0):.4f}")


def print_fluxscale_summary(path):
    """Print only fluxscale groups with detailed info."""
    with h5py.File(path, "r") as f:
        print(f"\n=== Fluxscale: {path} ===")
        if "fluxscale" not in f:
            print("  No fluxscale data found.")
            return
        for fk in f["fluxscale"].keys():
            for sk in f["fluxscale"][fk].keys():
                a = f["fluxscale"][fk][sk].attrs
                print(f"\n  Transfer: {a.get('transfer_field', '?')}")
                print(f"  Reference: {a.get('reference_field', '?')}")
                print(f"  Reference table: {a.get('reference_table', '?')}")
                print(f"  Jones type: {a.get('jones_type', '?')}")
                print(f"  SPW: {sk}")
                print(f"  Scale p: {a.get('scale_p', 0):.6f}")
                print(f"  Scale q: {a.get('scale_q', 0):.6f}")
                print(f"  Scatter p: {a.get('scatter_p', 0):.6f}")
                print(f"  Scatter q: {a.get('scatter_q', 0):.6f}")
                print(f"  N antennas: {a.get('n_ant', '?')}")
