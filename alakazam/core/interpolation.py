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
# Delay interpolation — reconstruct Jones at target frequencies
# -------------------------------------------------------------------

def interpolate_delay(sol_times, sol_delay, target_times, target_freqs, time_interp):
    """Interpolate delay values in time, then reconstruct Jones at target freqs.

    sol_delay: (n_ant, n_freq, n_time, 2) — delay in nanoseconds (HDF5 schema)
    Returns:   (n_target_t, n_ant, n_target_f, 2, 2) — always 5D Jones.

    The delay is real-valued, so interpolation is straightforward (no amp/phase).
    Jones reconstruction: J[a,f,p,p] = exp(-2pi*i * delay[a,p] * 1e-9 * freq[f])
    """
    # Transpose to (n_time, n_ant, n_freq, 2) for time interp
    d = sol_delay.transpose(2, 0, 1, 3)  # (n_time, n_ant, n_freq, 2)

    # Time interpolation on real-valued delays
    n_sol_t = d.shape[0]
    n_tt = len(target_times)
    if time_interp in ("exact", "nearest") or n_sol_t < 2:
        idx = np.argmin(np.abs(sol_times[:, None] - target_times[None, :]), axis=0)
        d_interp = d[idx]  # (n_tt, n_ant, n_freq, 2)
    elif time_interp == "linear":
        tc = np.clip(target_times, sol_times[0], sol_times[-1])
        shape = d.shape[1:]
        flat = d.reshape(n_sol_t, -1)
        out = np.empty((n_tt, flat.shape[1]), dtype=np.float64)
        for k in range(flat.shape[1]):
            out[:, k] = np.interp(tc, sol_times, flat[:, k])
        d_interp = out.reshape((n_tt,) + shape)
    else:
        idx = np.argmin(np.abs(sol_times[:, None] - target_times[None, :]), axis=0)
        d_interp = d[idx]

    # d_interp: (n_tt, n_ant, n_freq_sol, 2)
    # Average across solution freq bins (delay is same physical quantity per bin)
    delay_avg = np.mean(d_interp, axis=2)  # (n_tt, n_ant, 2)

    # Reconstruct Jones at every target frequency
    n_ant = delay_avg.shape[1]
    n_tf = len(target_freqs)
    J = np.zeros((n_tt, n_ant, n_tf, 2, 2), dtype=np.complex128)
    # delay_avg[:,:,0] = tau_p, delay_avg[:,:,1] = tau_q  (nanoseconds)
    # phase = -2*pi * tau_ns * 1e-9 * freq_hz
    twopi = 2.0 * np.pi * 1e-9
    for p in range(2):
        # (n_tt, n_ant, 1) * (1, 1, n_tf) -> (n_tt, n_ant, n_tf)
        phase = -twopi * delay_avg[:, :, p:p+1] * target_freqs[np.newaxis, np.newaxis, :]
        J[:, :, :, p, p] = np.exp(1j * phase)

    return J


# -------------------------------------------------------------------
# Single-field interpolation
# -------------------------------------------------------------------

def _freq_interp_desc(n_sol_freq, sol_freqs, target_freqs):
    """Return a short string describing the freq interpolation that will happen."""
    n_tf = len(target_freqs)
    if n_sol_freq == 1:
        return f"broadcast 1->{n_tf}"
    if n_sol_freq == n_tf and sol_freqs is not None and np.allclose(sol_freqs, target_freqs):
        return f"exact {n_tf}"
    if sol_freqs is not None:
        return f"nearest {n_sol_freq}->{n_tf}"
    return f"passthrough {n_sol_freq}"


def interpolate_jones(sol_times, sol_freqs, sol_jones,
                      target_times, target_freqs, time_interp):
    """Interpolate a jones cube onto target time/freq grids.

    sol_jones: (n_ant, n_freq, n_time, 2, 2) — HDF5 schema.
    Returns: (n_target_t, n_ant, n_target_f, 2, 2) — always 5D.
    """
    j = sol_jones
    # Normalise HDF5 schema (n_ant, n_freq, n_time, 2, 2)
    # to (n_time, n_ant, n_freq, 2, 2) for time interp
    if j.ndim == 5 and j.shape[-2:] == (2, 2):
        n_ant, n_freq, n_time = j.shape[:3]
        j = j.transpose(2, 0, 1, 3, 4)  # (n_time, n_ant, n_freq, 2, 2)
    elif j.ndim == 4:
        # (n_time, n_ant, 2, 2) — no freq axis, add n_freq=1
        j = j[:, :, np.newaxis, :, :]

    # j is now (n_time, n_ant, n_freq, 2, 2)
    j_interp = _interp_time(sol_times, j, target_times, time_interp)
    # j_interp: (n_target_t, n_ant, n_freq, 2, 2)

    n_t, n_ant, n_sf = j_interp.shape[:3]
    n_tf = len(target_freqs)

    # Freq-independent solution (n_freq=1): broadcast to target freqs
    if n_sf == 1:
        return np.broadcast_to(
            j_interp, (n_t, n_ant, n_tf, 2, 2)).copy()

    # Same freq grid: return as-is
    if n_sf == n_tf and sol_freqs is not None and np.allclose(sol_freqs, target_freqs):
        return j_interp

    # Different freq grids: nearest freq interp
    if sol_freqs is not None:
        idx = np.argmin(np.abs(sol_freqs[:, None] - target_freqs[None, :]), axis=0)
        return j_interp[:, :, idx]

    return j_interp


# -------------------------------------------------------------------
# Multi-field interpolation
# -------------------------------------------------------------------

def _interpolate_field(fd, target_times, target_freqs, time_interp):
    """Interpolate a single field's data. Uses delay reconstruction if available."""
    if fd.get("delay") is not None:
        return interpolate_delay(
            fd["times"], fd["delay"], target_times, target_freqs, time_interp)
    return interpolate_jones(
        fd["times"], fd.get("freqs"), fd["jones"],
        target_times, target_freqs, time_interp)


def _sol_n_freq(fd):
    """Number of solution freq channels from a field_data entry."""
    j = fd["jones"]
    if j.ndim == 5:
        return j.shape[1]  # (n_ant, n_freq, n_time, 2, 2)
    if j.ndim == 4:
        return 1  # no freq axis
    return j.shape[2] if j.ndim >= 3 else 1


def _log_interp(jones_label, chosen_fields, target_field, method,
                time_interp, n_sol_times, n_target_times,
                n_sol_freq, sol_freqs, target_freqs, has_delay=False):
    """Log one line: which field, why, and how interpolation works."""
    if not jones_label:
        return
    if has_delay:
        freq_desc = f"delay->jones {len(target_freqs)}ch"
    else:
        freq_desc = _freq_interp_desc(n_sol_freq, sol_freqs, target_freqs)
    src = "+".join(chosen_fields) if isinstance(chosen_fields, list) else chosen_fields
    logger.info(
        f"    {jones_label}: {src} -> {target_field}  "
        f"({method}, time={time_interp} {n_sol_times}->{n_target_times}, "
        f"freq={freq_desc})")


def interpolate_jones_multifield(
    fields_data, target_times, target_freqs,
    field_select, time_interp,
    target_ra=None, target_dec=None, pinned_fields=None,
    jones_label="", target_field="",
):
    """Select field and interpolate.

    fields_data: {name: {times, freqs, jones, ra_rad, dec_rad}}
    Returns: (n_t, n_ant, [n_f,] 2, 2)
    """
    if not fields_data:
        raise ValueError("fields_data empty")

    n_tt = len(target_times)

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
                _log_interp(jones_label, chosen, target_field, "nearest_sky",
                            time_interp, len(fd["times"]), n_tt,
                            _sol_n_freq(fd), fd.get("freqs"), target_freqs,
                            has_delay=fd.get("delay") is not None)
                return _interpolate_field(fd, target_times, target_freqs, time_interp)
            field_select = "nearest_time"

    if field_select == "nearest_time":
        tpf = {fn: fd["times"] for fn, fd in fields_data.items()}
        fft = np.array([select_field_nearest_time(tpf, t) for t in target_times])
        ufields = list(dict.fromkeys(fft))
        if len(ufields) == 1:
            fd = fields_data[ufields[0]]
            _log_interp(jones_label, ufields[0], target_field, "nearest_time",
                        time_interp, len(fd["times"]), n_tt,
                        _sol_n_freq(fd), fd.get("freqs"), target_freqs,
                        has_delay=fd.get("delay") is not None)
            return _interpolate_field(fd, target_times, target_freqs, time_interp)
        # Multiple fields contribute different time ranges
        parts = []
        for fn in ufields:
            cnt = int((fft == fn).sum())
            parts.append(f"{fn}({cnt}t)")
        fd0 = fields_data[ufields[0]]
        _log_interp(jones_label, parts, target_field, "nearest_time",
                    time_interp,
                    sum(len(fields_data[fn]["times"]) for fn in ufields),
                    n_tt, _sol_n_freq(fd0), fd0.get("freqs"), target_freqs,
                    has_delay=fd0.get("delay") is not None)
        result = None
        for fn in ufields:
            mask = fft == fn
            fd = fields_data[fn]
            sub = _interpolate_field(fd, target_times[mask], target_freqs, time_interp)
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
            _log_interp(jones_label, fn, target_field, "pinned",
                        time_interp, len(fd["times"]), n_tt,
                        _sol_n_freq(fd), fd.get("freqs"), target_freqs,
                        has_delay=fd.get("delay") is not None)
            results.append(_interpolate_field(fd, target_times, target_freqs, time_interp))
        if len(results) == 1:
            return results[0]
        return np.mean(np.stack(results), axis=0)

    raise ValueError(f"Unknown field_select: {field_select!r}")
