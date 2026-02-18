"""ALAKAZAM HDF5 I/O.

Solution file format v3.0:

  cal.h5
  ├── attrs: version, created_at, alakazam_version
  ├── metadata/
  │   ├── ms, ref_ant, jones_chain, solve_sequence (JSON provenance)
  │   └── fields/ → field names and directions
  └── {JONES_TYPE}/           (K, G, D, Xf, Kcross)
      └── field_{NAME}/
          └── spw_{N}/
              ├── jones        (n_t, n_ant, [n_f,] 2, 2) complex128
              ├── time         (n_t,) float64  MJD s
              ├── freq         (n_f,) float64  Hz  [optional, only if freq-resolved]
              ├── flags        (n_t, n_ant, [n_f,]) bool
              ├── quality      (n_t, n_ant) float64  [optional]
              └── attrs: field_name, field_ra, field_dec, jones_type,
                          spw, n_ant, solint_s, freqint_hz, phase_only,
                          ref_ant, scan_ids, preapply_chain (JSON)
  └── native_params/           [K and Kcross only]
      └── {JONES_TYPE}/
          └── field_{NAME}/
              └── spw_{N}/
                  ├── delay    (n_t, n_ant, 2) float64 ns   [K]
                  ├── delay_pq (n_t, n_ant) float64 ns      [Kcross]
                  └── time     (n_t,) float64
  └── fluxscale/               [written by fluxscale block]
      └── field_{TRANSFER}/
          └── spw_{N}/
              ├── attrs: reference_field, reference_table, jones_type
              ├── scale_p  float64
              ├── scale_q  float64
              ├── scatter_p float64
              ├── scatter_q float64
              └── n_ant    int

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations

import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

logger = logging.getLogger("alakazam")

HDF5_VERSION = "3.0"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _field_key(field_name: str) -> str:
    return f"field_{field_name.replace(' ', '_')}"


def _spw_key(spw: int) -> str:
    return f"spw_{spw}"


def _open(path: str, mode: str = "r"):
    return h5py.File(path, mode)


# ---------------------------------------------------------------------------
# Write solutions
# ---------------------------------------------------------------------------

def save_solutions(
    path: str,
    jones_type: str,
    field_name: str,
    spw: int,
    jones: np.ndarray,          # (n_t, n_ant, [n_f,] 2, 2) complex128
    times: np.ndarray,          # (n_t,) MJD s
    freqs: Optional[np.ndarray],  # (n_f,) Hz or None
    flags: Optional[np.ndarray],
    quality: Optional[np.ndarray],
    meta: Dict[str, Any],
    native_params: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> None:
    """Write or append solutions to HDF5."""
    mode = "a" if _path_exists(path) else "w"
    with h5py.File(path, mode) as f:
        # Top-level attrs
        if "version" not in f.attrs:
            f.attrs["version"]          = HDF5_VERSION
            f.attrs["created_at"]       = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            f.attrs["alakazam_version"] = "3.0.0"

        # Provenance
        if provenance:
            pgrp = f.require_group("metadata")
            for k, v in provenance.items():
                pgrp.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else str(v)

        # Solution group
        jkey = f"{jones_type}/{_field_key(field_name)}/{_spw_key(spw)}"
        if jkey in f:
            del f[jkey]
        grp = f.require_group(jkey)

        grp.create_dataset("jones", data=jones, compression="gzip", compression_opts=4)
        grp.create_dataset("time",  data=times)
        if freqs is not None:
            grp.create_dataset("freq", data=freqs)
        if flags is not None:
            grp.create_dataset("flags", data=flags)
        if quality is not None:
            grp.create_dataset("quality", data=quality)

        # Attributes
        grp.attrs["field_name"]  = field_name
        grp.attrs["jones_type"]  = jones_type
        grp.attrs["spw"]         = spw
        grp.attrs["n_ant"]       = jones.shape[1]
        for k, v in meta.items():
            grp.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v

        # Native params
        if native_params:
            npkey = f"native_params/{jones_type}/{_field_key(field_name)}/{_spw_key(spw)}"
            if npkey in f:
                del f[npkey]
            npgrp = f.require_group(npkey)
            for k, v in native_params.items():
                if isinstance(v, np.ndarray):
                    npgrp.create_dataset(k, data=v)
                else:
                    npgrp.attrs[k] = v


def save_fluxscale(
    path: str,
    transfer_field: str,
    spw: int,
    scale_p: float,
    scale_q: float,
    scatter_p: float,
    scatter_q: float,
    n_ant: int,
    reference_field: str,
    reference_table: str,
    jones_type: str,
) -> None:
    """Append fluxscale results to an HDF5 file."""
    mode = "a" if _path_exists(path) else "w"
    with h5py.File(path, mode) as f:
        fkey = f"fluxscale/{_field_key(transfer_field)}/{_spw_key(spw)}"
        if fkey in f:
            del f[fkey]
        grp = f.require_group(fkey)
        grp.attrs["reference_field"]  = reference_field
        grp.attrs["reference_table"]  = reference_table
        grp.attrs["jones_type"]       = jones_type
        grp.attrs["transfer_field"]   = transfer_field
        grp.attrs["scale_p"]          = scale_p
        grp.attrs["scale_q"]          = scale_q
        grp.attrs["scatter_p"]        = scatter_p
        grp.attrs["scatter_q"]        = scatter_q
        grp.attrs["n_ant"]            = n_ant


# ---------------------------------------------------------------------------
# Read solutions
# ---------------------------------------------------------------------------

def load_solutions(
    path: str,
    jones_type: str,
    field_name: str,
    spw: int,
) -> Dict[str, Any]:
    """Load one solution slot.

    Returns dict with:
      jones, time, freq (or None), flags (or None), quality (or None),
      attrs (dict), native_params (dict or None)
    """
    with h5py.File(path, "r") as f:
        key = f"{jones_type}/{_field_key(field_name)}/{_spw_key(spw)}"
        if key not in f:
            raise KeyError(f"Solution not found in {path}: {key}")
        grp = f[key]

        jones  = grp["jones"][:]
        times  = grp["time"][:]
        freqs  = grp["freq"][:] if "freq" in grp else None
        flags  = grp["flags"][:] if "flags" in grp else None
        quality = grp["quality"][:] if "quality" in grp else None
        attrs  = dict(grp.attrs)

        # Native params
        np_key = f"native_params/{jones_type}/{_field_key(field_name)}/{_spw_key(spw)}"
        native = {}
        if np_key in f:
            npgrp = f[np_key]
            for k in npgrp.keys():
                native[k] = npgrp[k][:]
            native.update(dict(npgrp.attrs))

        # Field direction from attrs
        ra_rad  = float(attrs.get("field_ra",  0.0))
        dec_rad = float(attrs.get("field_dec", 0.0))

    return {
        "jones":         jones,
        "time":          times,
        "freq":          freqs,
        "flags":         flags,
        "quality":       quality,
        "attrs":         attrs,
        "native_params": native if native else None,
        "ra_rad":        ra_rad,
        "dec_rad":       dec_rad,
    }


def load_all_fields(
    path: str,
    jones_type: str,
    spw: int,
    field_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load solutions for all (or specified) fields for a given jones_type and spw.

    Returns: {field_name: load_solutions(...) result}
    """
    with h5py.File(path, "r") as f:
        if jones_type not in f:
            return {}
        jgrp = f[jones_type]
        available = [
            k[len("field_"):].replace("_", " ") if False else k[len("field_"):]
            for k in jgrp.keys()
            if k.startswith("field_")
        ]

    result = {}
    for raw_key in available:
        # Reconstruct field name — stored as field_{name}
        with h5py.File(path, "r") as f:
            jgrp = f[jones_type]
            fkey = f"field_{raw_key}"
            if fkey not in jgrp:
                continue
            spw_key = _spw_key(spw)
            if spw_key not in jgrp[fkey]:
                continue
            fname = jgrp[fkey][spw_key].attrs.get("field_name", raw_key)

        if field_names is not None and fname not in field_names:
            continue
        try:
            result[fname] = load_solutions(path, jones_type, fname, spw)
        except Exception as e:
            logger.warning(f"Could not load {jones_type}/field_{fname}/spw_{spw} from {path}: {e}")

    return result


def load_fluxscale(
    path: str,
    transfer_field: str,
    spw: int,
) -> Optional[Dict[str, Any]]:
    """Load fluxscale factors for a transfer field."""
    with h5py.File(path, "r") as f:
        fkey = f"fluxscale/{_field_key(transfer_field)}/{_spw_key(spw)}"
        if fkey not in f:
            return None
        grp = f[fkey]
        return dict(grp.attrs)


def list_jones_types(path: str) -> List[str]:
    """List all Jones types stored in an HDF5 file."""
    with h5py.File(path, "r") as f:
        return [k for k in f.keys()
                if k not in ("metadata", "native_params", "fluxscale")]


def list_fields(path: str, jones_type: str) -> List[str]:
    """List all fields stored for a given Jones type."""
    with h5py.File(path, "r") as f:
        if jones_type not in f:
            return []
        return [
            f[jones_type][k].attrs.get("field_name",
                k[len("field_"):] if k.startswith("field_") else k)
            for spw_grp in [f[jones_type][k] for k in f[jones_type].keys()
                             if k.startswith("field_")]
            for spw_k in spw_grp.keys()
            if k.startswith("field_")
        ]


def list_spws(path: str, jones_type: str, field_name: str) -> List[int]:
    """List available SPWs for a given Jones type and field."""
    with h5py.File(path, "r") as f:
        fk = f"{jones_type}/{_field_key(field_name)}"
        if fk not in f:
            return []
        return [int(k[len("spw_"):]) for k in f[fk].keys() if k.startswith("spw_")]


def copy_solutions(src_path: str, dst_path: str, exclude_fields: Optional[List[str]] = None):
    """Copy all solutions from src to dst, optionally excluding certain fields."""
    mode = "a" if _path_exists(dst_path) else "w"
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, mode) as dst:
        for key in src.keys():
            if key in dst:
                continue
            src.copy(key, dst)


def rescale_solutions(
    path: str,
    jones_type: str,
    field_name: str,
    spw: int,
    scale_p: float,
    scale_q: float,
) -> None:
    """Rescale G solutions amplitude in-place: g *= sqrt(scale)."""
    with h5py.File(path, "a") as f:
        key = f"{jones_type}/{_field_key(field_name)}/{_spw_key(spw)}/jones"
        if key not in f:
            raise KeyError(f"Solution not found: {key}")
        jones = f[key][:]
        jones[:, :, ..., 0, 0] *= np.sqrt(scale_p)
        jones[:, :, ..., 1, 1] *= np.sqrt(scale_q)
        f[key][:] = jones


def print_summary(path: str) -> None:
    """Print a summary of HDF5 contents."""
    with h5py.File(path, "r") as f:
        print(f"\n=== {path} ===")
        print(f"  Version: {f.attrs.get('version', 'unknown')}")
        print(f"  Created: {f.attrs.get('created_at', 'unknown')}")
        for jtype in list_jones_types(path):
            jgrp = f[jtype]
            for fkey in jgrp.keys():
                if not fkey.startswith("field_"):
                    continue
                fname = fkey[len("field_"):]
                for spw_key in jgrp[fkey].keys():
                    grp = jgrp[fkey][spw_key]
                    n_t   = grp["jones"].shape[0]
                    n_ant = grp.attrs.get("n_ant", "?")
                    print(f"  [{jtype}] field={fname} {spw_key}  n_t={n_t}  n_ant={n_ant}")
        if "fluxscale" in f:
            print("  Fluxscale:")
            fs = f["fluxscale"]
            for fkey in fs.keys():
                for spw_key in fs[fkey].keys():
                    grp = fs[fkey][spw_key]
                    tf = grp.attrs.get("transfer_field", fkey)
                    rf = grp.attrs.get("reference_field", "?")
                    sp = grp.attrs.get("scale_p", float("nan"))
                    sq = grp.attrs.get("scale_q", float("nan"))
                    print(f"    {tf} ← {rf}  scale_p={sp:.4f}  scale_q={sq:.4f}")


def _path_exists(path: str) -> bool:
    import os
    return os.path.exists(path)
