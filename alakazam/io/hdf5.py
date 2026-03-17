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

  Metadata attrs per solution group:
    jones_type, field_name, field_ra, field_dec, spw, n_ant,
    matrix_form, ref_ant_constraint, ref_ant,
    solint_s, freqint_hz, phase_only, solver_backend,
    ms, scan_ids, preapply_chain

  Native params:
    K:  delay (n_ant, 2) float64 ns  — per time slot
    G:  amp (n_ant, 2), phase (n_ant, 2) float64
    D:  d_pq (n_ant,) complex128, d_qp (n_ant,) complex128
    KC: tau_cross float64 ns
    CP: phi_cross float64 rad

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
    native_params: Optional[Dict] = None,
    provenance: Optional[Dict] = None,
) -> None:
    """Write solutions in universal schema."""
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

        jkey = f"{jones_type}/{_fk(field_name)}/{_sk(spw)}"
        if jkey in f:
            del f[jkey]
        g = f.require_group(jkey)

        # Data arrays
        g.create_dataset("jones", data=jones, compression="gzip", compression_opts=4)
        g.create_dataset("flags", data=flags)
        g.create_dataset("time", data=times)
        g.create_dataset("freq", data=freqs)

        # Solver stats per cell
        if solver_stats:
            sg = g.require_group("solver_stats")
            for k, v in solver_stats.items():
                if isinstance(v, np.ndarray):
                    sg.create_dataset(k, data=v)

        # Metadata
        g.attrs["jones_type"] = jones_type
        g.attrs["field_name"] = field_name
        g.attrs["spw"] = spw
        g.attrs["n_ant"] = jones.shape[0]
        g.attrs["n_freq"] = jones.shape[1]
        g.attrs["n_time"] = jones.shape[2]
        g.attrs["matrix_form"] = MATRIX_FORMS.get(jones_type, "unknown")
        g.attrs["ref_ant_constraint"] = REF_ANT_CONSTRAINTS.get(jones_type, "unknown")
        for k, v in meta.items():
            if isinstance(v, (dict, list)):
                g.attrs[k] = json.dumps(v)
            elif isinstance(v, (np.integer, np.floating)):
                g.attrs[k] = v.item()
            elif v is not None:
                g.attrs[k] = v

        # Native params
        if native_params:
            npkey = f"native_params/{jones_type}/{_fk(field_name)}/{_sk(spw)}"
            if npkey in f:
                del f[npkey]
            ng = f.require_group(npkey)
            for k, v in native_params.items():
                if isinstance(v, np.ndarray):
                    ng.create_dataset(k, data=v)
                elif isinstance(v, (int, float, np.integer, np.floating)):
                    ng.attrs[k] = float(v) if isinstance(v, (float, np.floating)) else int(v)
                elif isinstance(v, complex):
                    ng.attrs[k + "_re"] = v.real
                    ng.attrs[k + "_im"] = v.imag
                elif isinstance(v, str):
                    ng.attrs[k] = v


def load_solutions(path: str, jones_type: str,
                   field_name: str, spw: int) -> Dict[str, Any]:
    """Load one solution.  Returns dict with jones, flags, time, freq, attrs, native_params."""
    with h5py.File(path, "r") as f:
        key = f"{jones_type}/{_fk(field_name)}/{_sk(spw)}"
        if key not in f:
            raise KeyError(f"Not found: {key} in {path}")
        g = f[key]
        jones = g["jones"][:]
        flags = g["flags"][:] if "flags" in g else None
        times = g["time"][:]
        freqs = g["freq"][:] if "freq" in g else None
        attrs = dict(g.attrs)

        # Solver stats
        solver_stats = {}
        if "solver_stats" in g:
            for k in g["solver_stats"].keys():
                solver_stats[k] = g["solver_stats"][k][:]

        # Native params
        npk = f"native_params/{jones_type}/{_fk(field_name)}/{_sk(spw)}"
        native = {}
        if npk in f:
            ng = f[npk]
            for k in ng.keys():
                native[k] = ng[k][:]
            native.update(dict(ng.attrs))

        ra = float(attrs.get("field_ra", 0.0))
        dec = float(attrs.get("field_dec", 0.0))

    return {"jones": jones, "flags": flags, "time": times, "freq": freqs,
            "attrs": attrs, "solver_stats": solver_stats,
            "native_params": native if native else None,
            "ra_rad": ra, "dec_rad": dec}


def load_all_fields(path: str, jones_type: str, spw: int,
                    field_names: Optional[List[str]] = None) -> Dict[str, Dict]:
    result = {}
    with h5py.File(path, "r") as f:
        if jones_type not in f:
            return {}
        jg = f[jones_type]
        for fkey in jg.keys():
            if not fkey.startswith("field_"):
                continue
            sk = _sk(spw)
            if sk not in jg[fkey]:
                continue
            sg = jg[fkey][sk]
            fname = sg.attrs.get("field_name", fkey[len("field_"):])
            if field_names is not None and fname not in field_names:
                continue
            jones = sg["jones"][:]
            flags = sg["flags"][:] if "flags" in sg else None
            times = sg["time"][:]
            freqs = sg["freq"][:] if "freq" in sg else None
            attrs = dict(sg.attrs)
            ra = float(attrs.get("field_ra", 0.0))
            dec = float(attrs.get("field_dec", 0.0))
            npk = f"native_params/{jones_type}/{fkey}/{sk}"
            native = {}
            if npk in f:
                ng = f[npk]
                for k in ng.keys():
                    native[k] = ng[k][:]
                native.update(dict(ng.attrs))
            result[fname] = {
                "jones": jones, "flags": flags, "time": times, "freq": freqs,
                "attrs": attrs, "native_params": native if native else None,
                "ra_rad": ra, "dec_rad": dec,
            }
    return result


def list_jones_types(path):
    with h5py.File(path, "r") as f:
        return [k for k in f.keys() if k not in ("metadata", "native_params", "fluxscale")]

def list_spws(path, jones_type, field_name):
    with h5py.File(path, "r") as f:
        fk = f"{jones_type}/{_fk(field_name)}"
        if fk not in f: return []
        return [int(k[4:]) for k in f[fk].keys() if k.startswith("spw_")]

def copy_solutions(src, dst):
    mode = "a" if _exists(dst) else "w"
    with h5py.File(src, "r") as s, h5py.File(dst, mode) as d:
        for k in s.keys():
            if k not in d:
                s.copy(k, d)

def rescale_solutions(path, jones_type, field_name, spw, scale_p, scale_q):
    with h5py.File(path, "a") as f:
        key = f"{jones_type}/{_fk(field_name)}/{_sk(spw)}/jones"
        if key not in f:
            raise KeyError(f"Not found: {key}")
        j = f[key][:]
        j[..., 0, 0] *= np.sqrt(scale_p)
        j[..., 1, 1] *= np.sqrt(scale_q)
        f[key][...] = j

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
                for sk in f[jt][fkey].keys():
                    g = f[jt][fkey][sk]
                    s = g["jones"].shape
                    print(f"  [{jt}] {fkey} {sk}  shape={s}  "
                          f"n_ant={s[0]} n_freq={s[1]} n_time={s[2]}")
                    print(f"    matrix: {g.attrs.get('matrix_form', '')}")
                    print(f"    backend: {g.attrs.get('solver_backend', '?')}")
        if "fluxscale" in f:
            print("  Fluxscale:")
            for fk in f["fluxscale"].keys():
                for sk in f["fluxscale"][fk].keys():
                    a = f["fluxscale"][fk][sk].attrs
                    print(f"    {a.get('transfer_field','?')} <- "
                          f"{a.get('reference_field','?')}  "
                          f"scale_p={a.get('scale_p',0):.4f}")
