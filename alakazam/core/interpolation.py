"""ALAKAZAM Interpolation Engine.

Interpolate Jones solutions in time and frequency onto target time/freq grids.

Modes:
  exact        — no interpolation: stamp each solint block identically (time),
                 recompute from native params at target freqs (freq for K/Kcross)
  nearest_time — nearest solution slot in time across all fields in table
  nearest_sky  — nearest field on sky (by angular distance), then nearest in time
  linear       — linear interpolation in time; linear in frequency between solint blocks
  cubic        — cubic spline in time; cubic in frequency between solint blocks

For time axis:
  exact   → every timestamp in the solint gets the same solution (no interp)
  nearest → argmin |t_target - t_solution|
  linear  → 1-D linear interp on complex (real and imag separately)
  cubic   → scipy CubicSpline on complex

For freq axis (when solution has multiple freq bins):
  exact   → recompute from native delay params at target freqs (K/Kcross only)
             for all other types: broadcast (same solution at all freqs)
  linear  → 1-D linear interp in freq between solution freq bins
  cubic   → cubic spline in freq between solution freq bins

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import logging

logger = logging.getLogger("alakazam")

try:
    from scipy.interpolate import CubicSpline, interp1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not found — cubic/linear interpolation unavailable, falling back to nearest")


# ---------------------------------------------------------------------------
# Angular distance helper (sky-nearest field selection)
# ---------------------------------------------------------------------------

def angular_distance(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle angular distance in radians between two sky positions."""
    dra  = ra2 - ra1
    ddec = dec2 - dec1
    a = (np.sin(ddec / 2) ** 2
         + np.cos(dec1) * np.cos(dec2) * np.sin(dra / 2) ** 2)
    return 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# Field selection
# ---------------------------------------------------------------------------

def select_field_nearest_time(
    solution_times_per_field: Dict[str, np.ndarray],
    target_time: float,
) -> str:
    """Return the field name whose solution grid is nearest in time to target_time."""
    best_field = None
    best_dt    = np.inf
    for fname, times in solution_times_per_field.items():
        dt = float(np.min(np.abs(times - target_time)))
        if dt < best_dt:
            best_dt    = dt
            best_field = fname
    return best_field


def select_field_nearest_sky(
    field_directions: Dict[str, Tuple[float, float]],  # name → (ra_rad, dec_rad)
    target_ra: float,
    target_dec: float,
) -> str:
    """Return the field name closest on sky to (target_ra, target_dec)."""
    best_field = None
    best_dist  = np.inf
    for fname, (ra, dec) in field_directions.items():
        d = angular_distance(ra, dec, target_ra, target_dec)
        if d < best_dist:
            best_dist  = d
            best_field = fname
    return best_field


# ---------------------------------------------------------------------------
# 1-D time interpolation kernels
# ---------------------------------------------------------------------------

def _interp_time_nearest(
    sol_times: np.ndarray,    # (n_sol,) MJD s
    sol_jones: np.ndarray,    # (n_sol, n_ant, [n_freq,] 2, 2) complex128
    target_times: np.ndarray, # (n_t,)
) -> np.ndarray:
    """Nearest-time interpolation. Returns (n_t, n_ant, ..., 2, 2)."""
    idx = np.argmin(np.abs(sol_times[:, None] - target_times[None, :]), axis=0)
    return sol_jones[idx]


def _interp_time_exact(
    sol_times: np.ndarray,
    sol_jones: np.ndarray,
    target_times: np.ndarray,
    solint: float,
) -> np.ndarray:
    """Exact mode: stamp each target time with the solution for its solint block.

    Every target time that falls within a solint block gets the exact same solution.
    No interpolation. Uses nearest to handle boundary cases.
    """
    # Exact = nearest within block. Since blocks are contiguous this is equivalent
    # to nearest for target times inside the observation. Same behaviour as
    # nearest when solution density is one per solint.
    return _interp_time_nearest(sol_times, sol_jones, target_times)


def _interp_time_linear(
    sol_times: np.ndarray,
    sol_jones: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """Linear interpolation in time on complex Jones.

    Interpolates real and imaginary parts separately.
    Edge behaviour: extrapolate with nearest (clamp).
    """
    if not HAS_SCIPY:
        return _interp_time_nearest(sol_times, sol_jones, target_times)

    n_t   = len(target_times)
    shape = sol_jones.shape[1:]   # (n_ant, ..., 2, 2)
    out   = np.empty((n_t,) + shape, dtype=np.complex128)

    # Flatten all axes except time
    flat_sol  = sol_jones.reshape(len(sol_times), -1)
    flat_out  = np.empty((n_t, flat_sol.shape[1]), dtype=np.complex128)

    # Clamp target times to solution range
    t0, t1 = sol_times[0], sol_times[-1]
    tc = np.clip(target_times, t0, t1)

    for k in range(flat_sol.shape[1]):
        re = np.interp(tc, sol_times, flat_sol[:, k].real)
        im = np.interp(tc, sol_times, flat_sol[:, k].imag)
        flat_out[:, k] = re + 1j * im

    return flat_out.reshape((n_t,) + shape)


def _interp_time_cubic(
    sol_times: np.ndarray,
    sol_jones: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """Cubic spline interpolation in time on complex Jones.

    Falls back to linear if fewer than 4 solution slots.
    """
    if not HAS_SCIPY or len(sol_times) < 4:
        return _interp_time_linear(sol_times, sol_jones, target_times)

    n_t   = len(target_times)
    shape = sol_jones.shape[1:]
    flat_sol = sol_jones.reshape(len(sol_times), -1)
    flat_out = np.empty((n_t, flat_sol.shape[1]), dtype=np.complex128)

    t0, t1 = sol_times[0], sol_times[-1]
    tc = np.clip(target_times, t0, t1)

    for k in range(flat_sol.shape[1]):
        cs_re = CubicSpline(sol_times, flat_sol[:, k].real, extrapolate=False)
        cs_im = CubicSpline(sol_times, flat_sol[:, k].imag, extrapolate=False)
        re = cs_re(tc)
        im = cs_im(tc)
        # nan-fill from extrapolation → clamp to nearest
        nan_mask = np.isnan(re) | np.isnan(im)
        if nan_mask.any():
            fallback = _interp_time_nearest(sol_times, flat_sol[:, k:k+1], tc)[:, 0]
            re = np.where(nan_mask, fallback.real, re)
            im = np.where(nan_mask, fallback.imag, im)
        flat_out[:, k] = re + 1j * im

    return flat_out.reshape((n_t,) + shape)


# ---------------------------------------------------------------------------
# 1-D frequency interpolation kernels
# ---------------------------------------------------------------------------

def _interp_freq_nearest(
    sol_freqs: np.ndarray,   # (n_sol_freq,) Hz
    sol_jones: np.ndarray,   # (n_t, n_ant, n_sol_freq, 2, 2)
    target_freqs: np.ndarray,  # (n_freq,)
) -> np.ndarray:
    """Nearest-frequency interpolation. Returns (n_t, n_ant, n_freq, 2, 2)."""
    idx = np.argmin(np.abs(sol_freqs[:, None] - target_freqs[None, :]), axis=0)
    return sol_jones[:, :, idx]


def _interp_freq_linear(
    sol_freqs: np.ndarray,
    sol_jones: np.ndarray,   # (n_t, n_ant, n_sol_freq, 2, 2)
    target_freqs: np.ndarray,
) -> np.ndarray:
    """Linear interpolation in frequency."""
    if not HAS_SCIPY or len(sol_freqs) < 2:
        return _interp_freq_nearest(sol_freqs, sol_jones, target_freqs)

    n_t, n_ant, _, _, _ = sol_jones.shape
    n_freq = len(target_freqs)
    out = np.empty((n_t, n_ant, n_freq, 2, 2), dtype=np.complex128)

    f0, f1 = sol_freqs[0], sol_freqs[-1]
    fc = np.clip(target_freqs, f0, f1)

    flat = sol_jones.reshape(n_t * n_ant, len(sol_freqs), 4)  # 4 = 2x2 flat
    flat_out = np.empty((n_t * n_ant, n_freq, 4), dtype=np.complex128)

    for k in range(4):
        for row in range(flat.shape[0]):
            re = np.interp(fc, sol_freqs, flat[row, :, k].real)
            im = np.interp(fc, sol_freqs, flat[row, :, k].imag)
            flat_out[row, :, k] = re + 1j * im

    return flat_out.reshape(n_t, n_ant, n_freq, 2, 2)


def _interp_freq_cubic(
    sol_freqs: np.ndarray,
    sol_jones: np.ndarray,   # (n_t, n_ant, n_sol_freq, 2, 2)
    target_freqs: np.ndarray,
) -> np.ndarray:
    """Cubic spline interpolation in frequency."""
    if not HAS_SCIPY or len(sol_freqs) < 4:
        return _interp_freq_linear(sol_freqs, sol_jones, target_freqs)

    n_t, n_ant, _, _, _ = sol_jones.shape
    n_freq = len(target_freqs)
    out = np.empty((n_t, n_ant, n_freq, 2, 2), dtype=np.complex128)

    f0, f1 = sol_freqs[0], sol_freqs[-1]
    fc = np.clip(target_freqs, f0, f1)

    flat = sol_jones.reshape(n_t * n_ant, len(sol_freqs), 4)
    flat_out = np.empty((n_t * n_ant, n_freq, 4), dtype=np.complex128)

    for k in range(4):
        for row in range(flat.shape[0]):
            cs_re = CubicSpline(sol_freqs, flat[row, :, k].real, extrapolate=False)
            cs_im = CubicSpline(sol_freqs, flat[row, :, k].imag, extrapolate=False)
            re = cs_re(fc); im = cs_im(fc)
            nan_mask = np.isnan(re) | np.isnan(im)
            if nan_mask.any():
                fb = np.interp(fc, sol_freqs, flat[row, :, k].real)
                re = np.where(nan_mask, fb, re)
                fb = np.interp(fc, sol_freqs, flat[row, :, k].imag)
                im = np.where(nan_mask, fb, im)
            flat_out[row, :, k] = re + 1j * im

    return flat_out.reshape(n_t, n_ant, n_freq, 2, 2)


# ---------------------------------------------------------------------------
# Main interpolation dispatcher
# ---------------------------------------------------------------------------

def interpolate_jones(
    sol_times: np.ndarray,           # (n_sol_t,) — solution timestamps MJD s
    sol_freqs: Optional[np.ndarray], # (n_sol_f,) Hz or None (freq-indep)
    sol_jones: np.ndarray,           # (n_sol_t, n_ant, [n_sol_f,] 2, 2)
    target_times: np.ndarray,        # (n_t,)
    target_freqs: Optional[np.ndarray], # (n_f,) or None
    time_interp: str,                # exact | nearest_time | linear | cubic
    native_params: Optional[Dict[str, Any]] = None,  # for exact freq recompute
) -> np.ndarray:
    """
    Interpolate a solution cube onto target time/freq grids.

    Returns:
      - If target_freqs is None or sol has no freq axis: (n_t, n_ant, 2, 2)
      - Else: (n_t, n_ant, n_f, 2, 2)

    For 'exact' mode:
      - Time axis: stamp exact solution to each solint block (nearest-time lookup)
      - Freq axis: if native_params provided (K/Kcross) recompute from delay;
                   otherwise broadcast (same Jones at all freqs)
    """
    freq_indep = (sol_jones.ndim == 4)   # (n_t, n_ant, 2, 2) → no freq axis

    # ---- Step 1: Time interpolation ----
    if time_interp == "exact":
        j_t = _interp_time_exact(sol_times, sol_jones, target_times, solint=0.0)
    elif time_interp in ("nearest_time", "nearest_sky"):
        # nearest_sky field selection is handled by caller before calling this
        # function — by this point it's always nearest in time within the field
        j_t = _interp_time_nearest(sol_times, sol_jones, target_times)
    elif time_interp == "linear":
        j_t = _interp_time_linear(sol_times, sol_jones, target_times)
    elif time_interp == "cubic":
        j_t = _interp_time_cubic(sol_times, sol_jones, target_times)
    else:
        raise ValueError(f"Unknown time_interp mode: {time_interp!r}")

    # j_t shape: (n_t, n_ant, [n_sol_f,] 2, 2)

    if target_freqs is None:
        return j_t   # caller doesn't need freq axis

    # ---- Step 2: Frequency handling ----

    if time_interp == "exact" and native_params is not None:
        # Recompute from delay params at exact target frequencies
        return _recompute_exact_freq(j_t, native_params, target_freqs)

    if freq_indep:
        # Broadcast across frequencies
        n_t, n_ant = j_t.shape[:2]
        n_f = len(target_freqs)
        return np.broadcast_to(
            j_t[:, :, None, :, :],
            (n_t, n_ant, n_f, 2, 2)
        ).copy()

    # Have solution freq bins → interpolate in freq
    if sol_freqs is None:
        raise ValueError("sol_freqs required for freq interpolation")

    if time_interp in ("exact", "nearest_time", "nearest_sky"):
        return _interp_freq_nearest(sol_freqs, j_t, target_freqs)
    elif time_interp == "linear":
        return _interp_freq_linear(sol_freqs, j_t, target_freqs)
    elif time_interp == "cubic":
        return _interp_freq_cubic(sol_freqs, j_t, target_freqs)
    else:
        return _interp_freq_nearest(sol_freqs, j_t, target_freqs)


def _recompute_exact_freq(
    j_t: np.ndarray,                    # (n_t, n_ant, [n_sol_f,] 2, 2) — ignored for freq axis
    native_params: Dict[str, Any],
    target_freqs: np.ndarray,
) -> np.ndarray:
    """Recompute Jones from stored native params at exact target frequencies.

    Supports K (delay) and Kcross (cross-delay) native params.
    """
    from ..jones.constructors import delay_to_jones, crossdelay_to_jones

    jones_type = native_params.get("type", "K")
    n_t  = j_t.shape[0]
    n_f  = len(target_freqs)

    if jones_type == "K":
        delay = native_params["delay"]  # (n_sol_t, n_ant, 2) ns
        n_ant = delay.shape[1]
        out = np.empty((n_t, n_ant, n_f, 2, 2), dtype=np.complex128)
        for t in range(n_t):
            out[t] = delay_to_jones(delay[t], target_freqs)
        return out

    elif jones_type == "Kcross":
        delay_pq = native_params["delay_pq"]  # (n_sol_t, n_ant) ns
        n_ant = delay_pq.shape[1]
        out = np.empty((n_t, n_ant, n_f, 2, 2), dtype=np.complex128)
        for t in range(n_t):
            out[t] = crossdelay_to_jones(delay_pq[t], target_freqs)
        return out

    else:
        # Unknown native type — broadcast
        n_ant = j_t.shape[1]
        src = j_t if j_t.ndim == 4 else j_t[:, :, 0]
        return np.broadcast_to(src[:, :, None, :, :], (n_t, n_ant, n_f, 2, 2)).copy()


# ---------------------------------------------------------------------------
# Multi-field interpolation (field selection then per-field interp)
# ---------------------------------------------------------------------------

def interpolate_jones_multifield(
    fields_data: Dict[str, Dict[str, Any]],
    # {field_name: {
    #   "times": (n_sol_t,),
    #   "freqs": (n_sol_f,) or None,
    #   "jones": (n_sol_t, n_ant, [n_sol_f,] 2, 2),
    #   "ra_rad": float,
    #   "dec_rad": float,
    #   "native_params": dict or None,
    # }}
    target_times: np.ndarray,   # (n_t,)
    target_freqs: Optional[np.ndarray],
    field_select: str,           # nearest_time | nearest_sky | pinned
    time_interp: str,
    target_ra: Optional[float] = None,
    target_dec: Optional[float] = None,
    pinned_fields: Optional[List[str]] = None,  # for field_select=pinned
) -> np.ndarray:
    """
    Select field(s) and interpolate.

    For nearest_time: globally pick whichever field is closest in time to each
                      target time, then interpolate within that field.
    For nearest_sky:  pick the single field closest on sky, then interpolate
                      across all target times within that field.
    For pinned:       use the specified field(s), average if multiple.

    Returns: (n_t, n_ant, [n_f,] 2, 2)
    """
    if not fields_data:
        raise ValueError("fields_data is empty")

    if field_select == "nearest_sky":
        if target_ra is None or target_dec is None:
            raise ValueError("nearest_sky requires target_ra and target_dec")
        directions = {fn: (fd["ra_rad"], fd["dec_rad"])
                      for fn, fd in fields_data.items()
                      if fd.get("ra_rad") is not None}
        if not directions:
            # Fall back to nearest_time
            logger.warning("nearest_sky: no field directions stored, falling back to nearest_time")
            field_select = "nearest_time"
        else:
            chosen = select_field_nearest_sky(directions, target_ra, target_dec)
            fd = fields_data[chosen]
            return interpolate_jones(
                fd["times"], fd.get("freqs"), fd["jones"],
                target_times, target_freqs, time_interp,
                native_params=fd.get("native_params"),
            )

    if field_select == "nearest_time":
        # For each target time, pick the closest field
        sol_times_per_field = {fn: fd["times"] for fn, fd in fields_data.items()}

        # Group target times by their nearest field
        field_for_time = []
        for t in target_times:
            best = select_field_nearest_time(sol_times_per_field, t)
            field_for_time.append(best)

        field_for_time = np.array(field_for_time)
        unique_fields = list(dict.fromkeys(field_for_time))  # preserve order

        if len(unique_fields) == 1:
            fd = fields_data[unique_fields[0]]
            return interpolate_jones(
                fd["times"], fd.get("freqs"), fd["jones"],
                target_times, target_freqs, time_interp,
                native_params=fd.get("native_params"),
            )

        # Multiple fields needed → interpolate each segment separately
        # and stitch
        result = None
        for fname in unique_fields:
            mask = (field_for_time == fname)
            sub_times = target_times[mask]
            fd = fields_data[fname]
            sub_result = interpolate_jones(
                fd["times"], fd.get("freqs"), fd["jones"],
                sub_times, target_freqs, time_interp,
                native_params=fd.get("native_params"),
            )
            if result is None:
                shape = (len(target_times),) + sub_result.shape[1:]
                result = np.empty(shape, dtype=np.complex128)
            result[mask] = sub_result

        return result

    if field_select == "pinned":
        if not pinned_fields:
            raise ValueError("field_select=pinned requires pinned_fields")
        # Use pinned fields — average if multiple
        results = []
        for fname in pinned_fields:
            if fname not in fields_data:
                raise ValueError(f"Pinned field {fname!r} not found in solution table")
            fd = fields_data[fname]
            r = interpolate_jones(
                fd["times"], fd.get("freqs"), fd["jones"],
                target_times, target_freqs, time_interp,
                native_params=fd.get("native_params"),
            )
            results.append(r)
        if len(results) == 1:
            return results[0]
        return np.mean(np.stack(results, axis=0), axis=0)

    raise ValueError(f"Unknown field_select: {field_select!r}")
