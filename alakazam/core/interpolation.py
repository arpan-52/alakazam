"""ALAKAZAM v1 Interpolation Engine.

Jones schema in HDF5: (n_ant, n_freq, n_time, 2, 2)

For interpolation onto target grids, we transpose internally as needed.
The multifield interpolator is the main entry point used by flow.py and apply.py.

Time modes: exact, nearest, linear, cubic
Field selection: nearest_time, nearest_sky, pinned

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import logging

logger = logging.getLogger("alakazam")

try:
    from scipy.interpolate import CubicSpline
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def angular_distance(ra1, dec1, ra2, dec2):
    dra = ra2 - ra1; ddec = dec2 - dec1
    a = np.sin(ddec/2)**2 + np.cos(dec1)*np.cos(dec2)*np.sin(dra/2)**2
    return 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def select_field_nearest_time(times_per_field, target_time):
    best, best_dt = None, np.inf
    for fn, times in times_per_field.items():
        dt = float(np.min(np.abs(times - target_time)))
        if dt < best_dt:
            best_dt = dt; best = fn
    return best


def select_field_nearest_sky(directions, target_ra, target_dec):
    best, best_d = None, np.inf
    for fn, (ra, dec) in directions.items():
        d = angular_distance(ra, dec, target_ra, target_dec)
        if d < best_d:
            best_d = d; best = fn
    return best


# -------------------------------------------------------------------
# Time interpolation kernels (operate on flattened arrays)
# -------------------------------------------------------------------

def _interp_nearest(sol_times, sol_data, target_times):
    """Nearest in time. sol_data: (n_sol_t, ...) -> (n_target_t, ...)"""
    idx = np.argmin(np.abs(sol_times[:, None] - target_times[None, :]), axis=0)
    return sol_data[idx]


def _interp_linear(sol_times, sol_data, target_times):
    """Linear in time using amp/phase for complex data."""
    if len(sol_times) < 2:
        return _interp_nearest(sol_times, sol_data, target_times)

    n_t = len(target_times)
    shape = sol_data.shape[1:]
    flat = sol_data.reshape(len(sol_times), -1)
    out = np.empty((n_t, flat.shape[1]), dtype=np.complex128)
    tc = np.clip(target_times, sol_times[0], sol_times[-1])

    amp = np.abs(flat); phase = np.unwrap(np.angle(flat), axis=0)
    for k in range(flat.shape[1]):
        a = np.interp(tc, sol_times, amp[:, k])
        p = np.interp(tc, sol_times, phase[:, k])
        out[:, k] = a * np.exp(1j * p)
    return out.reshape((n_t,) + shape)


def _interp_cubic(sol_times, sol_data, target_times):
    if not HAS_SCIPY or len(sol_times) < 4:
        return _interp_linear(sol_times, sol_data, target_times)

    n_t = len(target_times)
    shape = sol_data.shape[1:]
    flat = sol_data.reshape(len(sol_times), -1)
    out = np.empty((n_t, flat.shape[1]), dtype=np.complex128)
    tc = np.clip(target_times, sol_times[0], sol_times[-1])

    amp = np.abs(flat); phase = np.unwrap(np.angle(flat), axis=0)
    for k in range(flat.shape[1]):
        try:
            ca = CubicSpline(sol_times, amp[:, k], extrapolate=False)
            cp = CubicSpline(sol_times, phase[:, k], extrapolate=False)
            a = ca(tc); p = cp(tc)
            bad = np.isnan(a) | np.isnan(p)
            if bad.any():
                a = np.where(bad, np.interp(tc, sol_times, amp[:, k]), a)
                p = np.where(bad, np.interp(tc, sol_times, phase[:, k]), p)
            out[:, k] = a * np.exp(1j * p)
        except Exception:
            out[:, k] = np.interp(tc, sol_times, amp[:, k]) * np.exp(
                1j * np.interp(tc, sol_times, phase[:, k]))
    return out.reshape((n_t,) + shape)


def _interp_time(sol_times, sol_data, target_times, mode):
    if mode == "exact" or mode == "nearest":
        return _interp_nearest(sol_times, sol_data, target_times)
    elif mode == "linear":
        return _interp_linear(sol_times, sol_data, target_times)
    elif mode == "cubic":
        return _interp_cubic(sol_times, sol_data, target_times)
    else:
        return _interp_nearest(sol_times, sol_data, target_times)


# -------------------------------------------------------------------
# Single-field interpolation
# -------------------------------------------------------------------

def interpolate_jones(sol_times, sol_freqs, sol_jones,
                      target_times, target_freqs, time_interp):
    """Interpolate a jones cube onto target time/freq grids.

    sol_jones can be:
      (n_ant, n_freq, n_time, 2, 2) — full schema from HDF5
      (n_time, n_ant, 2, 2) — old format (auto-detected)
      (n_time, n_ant, n_freq, 2, 2) — old format

    Returns: (n_target_t, n_ant, [n_target_f,] 2, 2) for preapply use.
    """
    # Detect and normalise to (n_time, n_ant, [n_freq,] 2, 2) for interp
    j = sol_jones
    if j.ndim == 5 and j.shape[2] != 2:
        # Could be (n_ant, n_freq, n_time, 2, 2) — our schema
        # Transpose to (n_time, n_ant, n_freq, 2, 2)
        if j.shape[-2:] == (2, 2):
            n_ant, n_freq, n_time = j.shape[:3]
            j = j.transpose(2, 0, 1, 3, 4)  # (n_time, n_ant, n_freq, 2, 2)

    # Now j is (n_time, n_ant, [n_freq,] 2, 2)
    j_interp = _interp_time(sol_times, j, target_times, time_interp)

    if target_freqs is None:
        return j_interp

    # Freq handling
    if j_interp.ndim == 4:
        # (n_t, n_ant, 2, 2) — broadcast to freq
        n_t, n_ant = j_interp.shape[:2]
        return np.broadcast_to(
            j_interp[:, :, None, :, :],
            (n_t, n_ant, len(target_freqs), 2, 2)).copy()

    if sol_freqs is not None and j_interp.ndim == 5:
        # (n_t, n_ant, n_sol_f, 2, 2) — interp to target_freqs
        n_t, n_ant, n_sf = j_interp.shape[:3]
        n_tf = len(target_freqs)
        if n_sf == n_tf and np.allclose(sol_freqs, target_freqs):
            return j_interp
        # Nearest freq interp
        idx = np.argmin(np.abs(sol_freqs[:, None] - target_freqs[None, :]), axis=0)
        return j_interp[:, :, idx]

    return j_interp


# -------------------------------------------------------------------
# Multi-field interpolation
# -------------------------------------------------------------------

def interpolate_jones_multifield(
    fields_data, target_times, target_freqs,
    field_select, time_interp,
    target_ra=None, target_dec=None, pinned_fields=None,
):
    """Select field and interpolate.

    fields_data: {name: {times, freqs, jones, ra_rad, dec_rad}}
    Returns: (n_t, n_ant, [n_f,] 2, 2)
    """
    if not fields_data:
        raise ValueError("fields_data empty")

    if field_select == "nearest_sky":
        if target_ra is None:
            field_select = "nearest_time"
        else:
            dirs = {fn: (fd["ra_rad"], fd["dec_rad"])
                    for fn, fd in fields_data.items()
                    if fd.get("ra_rad") is not None}
            if dirs:
                chosen = select_field_nearest_sky(dirs, target_ra, target_dec)
                fd = fields_data[chosen]
                return interpolate_jones(
                    fd["times"], fd.get("freqs"), fd["jones"],
                    target_times, target_freqs, time_interp)
            field_select = "nearest_time"

    if field_select == "nearest_time":
        tpf = {fn: fd["times"] for fn, fd in fields_data.items()}
        fft = np.array([select_field_nearest_time(tpf, t) for t in target_times])
        ufields = list(dict.fromkeys(fft))
        if len(ufields) == 1:
            fd = fields_data[ufields[0]]
            return interpolate_jones(
                fd["times"], fd.get("freqs"), fd["jones"],
                target_times, target_freqs, time_interp)
        result = None
        for fn in ufields:
            mask = fft == fn
            fd = fields_data[fn]
            sub = interpolate_jones(
                fd["times"], fd.get("freqs"), fd["jones"],
                target_times[mask], target_freqs, time_interp)
            if result is None:
                result = np.empty((len(target_times),) + sub.shape[1:], dtype=np.complex128)
            result[mask] = sub
        return result

    if field_select == "pinned":
        if not pinned_fields:
            raise ValueError("pinned requires pinned_fields")
        results = []
        for fn in pinned_fields:
            if fn not in fields_data:
                raise ValueError(f"Pinned field {fn!r} not found")
            fd = fields_data[fn]
            results.append(interpolate_jones(
                fd["times"], fd.get("freqs"), fd["jones"],
                target_times, target_freqs, time_interp))
        if len(results) == 1:
            return results[0]
        return np.mean(np.stack(results), axis=0)

    raise ValueError(f"Unknown field_select: {field_select!r}")
