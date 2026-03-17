"""ALAKAZAM v1 Pipeline Orchestrator.

Solve flow per cell:
  LOAD raw (n_row, n_chan, n_corr)    ← native casacore format
  FLAG raw                            ← on native format
  AVERAGE raw → tiny (n_bl, n_corr) or (n_bl, n_chan, n_corr)
  CONVERT tiny to 2x2                ← only here, on averaged data
  APPLY PARANG at t_mid              ← on tiny 2x2
  APPLY PREAPPLY at t_mid            ← on tiny 2x2
  SOLVE                              ← on tiny 2x2

Memory-aware batching: loads groups of time_bins that fit in RAM.
Each batch: load → flag → average → free raw → solve → store.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations
import gc, json, logging, os, time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .config import (AlakazamConfig, SolveBlock,
                     spw_ids_from_selection, chan_slice_for_spw)
from .core.ms_io import (detect_metadata, read_data, query_times_scans,
                         compute_solint_grid, validate_selections,
                         raw_to_2x2)
from .core.averaging import (average_per_baseline_full,
                             average_per_baseline_time_only)
from .core.rfi import flag_rfi
from .core.memory import get_available_ram_gb
from .core.interpolation import interpolate_jones_multifield
from .solvers import get_solver, detect_device
from .io.hdf5 import save_solutions, load_all_fields
from .calibration.fluxscale import run_fluxscale
from .calibration.apply import apply_calibration
from .jones.algebra import compose_jones_chain, unapply_jones_to_rows

logger = logging.getLogger("alakazam")

try:
    from rich.console import Console
    _con = Console()
except ImportError:
    _con = None

def _log(msg, style=""):
    logger.info(msg)
    if _con and style: _con.print(msg, style=style)
    elif _con: _con.print(msg)


# ============================================================
# PUBLIC
# ============================================================

def run_pipeline(config: AlakazamConfig) -> None:
    t0 = _time.time()
    _log("ALAKAZAM v1.0 — Radio Interferometric Calibration", "bold cyan")
    for i, sb in enumerate(config.solve_blocks):
        _log(f"\n{'='*60}", "bold green")
        _log(f"SOLVE BLOCK {i+1}/{len(config.solve_blocks)}: {sb.ms}", "bold green")
        _run_solve_block(sb)
    for i, fb in enumerate(config.fluxscale_blocks):
        _log(f"\n{'='*60}", "bold magenta")
        run_fluxscale(fb)
    for i, ab in enumerate(config.apply_blocks):
        _log(f"\n{'='*60}", "bold yellow")
        _log(f"APPLY {i+1}: {ab.ms}", "bold yellow")
        apply_calibration(ab)
    _log(f"\nDone in {_time.time()-t0:.1f}s", "bold green")


# ============================================================
# SOLVE BLOCK
# ============================================================

def _run_solve_block(sb: SolveBlock) -> None:
    from .config import resolve_ref_ant
    meta = detect_metadata(sb.ms)
    all_fields = set()
    for fg in sb.field_groups: all_fields.update(fg)
    validate_selections(meta, list(all_fields))

    sb.ref_ant = resolve_ref_ant(sb.ref_ant, meta.ant_names)

    # Active antennas: only those with data
    active_ants = sorted(set(meta.ant1.tolist() + meta.ant2.tolist()))
    n_active = len(active_ants)
    active_names = [meta.ant_names[a] for a in active_ants]
    ant_remap = {orig: new for new, orig in enumerate(active_ants)}

    if n_active < meta.n_ant:
        _log(f"  active: {n_active}/{meta.n_ant} ({', '.join(active_names)})")
    else:
        _log(f"  antennas: {n_active}")

    if sb.ref_ant not in ant_remap:
        raise ValueError(
            f"ref_ant {meta.ant_names[sb.ref_ant]} has no data. "
            f"Active: {active_names}")
    ref_remapped = ant_remap[sb.ref_ant]
    _log(f"  ref_ant: {meta.ant_names[sb.ref_ant]} (idx {ref_remapped}/{n_active})")

    device = detect_device(sb.solver_backend, sb.gpu)
    _log(f"  backend={sb.solver_backend} device={device}")

    global_spws = spw_ids_from_selection(sb.spw)
    if global_spws is None: global_spws = list(range(meta.n_spw))

    for spw in global_spws:
        freqs_full = meta.spw_freqs[spw]
        _log(f"  SPW {spw} ({len(freqs_full)} ch)")
        internal_stack: Dict[str, Dict] = {}

        # Build unique keys: K0, G0, G1, G2, D0, ...
        type_counter: Dict[str, int] = {}
        jones_keys = []
        for jt in sb.jones:
            idx = type_counter.get(jt, 0)
            type_counter[jt] = idx + 1
            jones_keys.append(f"{jt}{idx}")

        for step_idx, jones_type in enumerate(sb.jones):
            jones_key = jones_keys[step_idx]

            if sb.step_spw and sb.step_spw[step_idx] is not None:
                sids = spw_ids_from_selection(sb.step_spw[step_idx])
                if spw not in sids: continue
                chan_sl = chan_slice_for_spw(sb.step_spw[step_idx], spw, len(freqs_full))
            else:
                chan_sl = chan_slice_for_spw(sb.spw, spw, len(freqs_full))

            freqs = freqs_full[chan_sl]
            solver = get_solver(
                jones_type, ref_ant=ref_remapped, max_iter=sb.max_iter,
                tol=sb.tol, phase_only=sb.phase_only[step_idx],
                backend=sb.solver_backend, device=device)

            for fi, fn in enumerate(sb.field_groups[step_idx]):
                fscans = sb.scans_per_step[step_idx][fi] if fi < len(sb.scans_per_step[step_idx]) else None
                _log(f"    step {step_idx+1} {jones_key} field={fn}")
                _solve_one_field(sb, step_idx, jones_key, fn, fscans,
                                 spw, freqs, chan_sl, meta, solver,
                                 internal_stack, n_active, ant_remap,
                                 active_names, jones_keys)

            fields = load_all_fields(sb.output, jones_key, spw)
            if jones_key not in internal_stack: internal_stack[jones_key] = {}
            internal_stack[jones_key].update(fields)
        gc.collect()


# ============================================================
# SOLVE ONE FIELD — the core
# ============================================================

def _solve_one_field(sb, step_idx, jones_type, field_name, field_scans,
                     spw, freqs, chan_sl, meta, solver, internal_stack,
                     n_active, ant_remap, active_names, jones_keys):
    """Load raw → flag → average → convert 2x2 → parang/preapply at t_mid → solve.
    jones_type is the unique key (K0, G0, G1, etc)."""
    n_ant = n_active
    n_chan = len(freqs)
    # K0, KC0 etc solve across freq — check base type
    base_type = jones_type.rstrip("0123456789")
    freq_dep = base_type in ("K", "KC")

    # ---- 0. LIGHTWEIGHT QUERY for timestamps ----
    ts = query_times_scans(sb.ms, spw, fields=[field_name], scans=field_scans)
    if not ts:
        logger.warning(f"      no data for {field_name} spw={spw}"); return

    row_times_all = ts["times"]
    row_scans_all = ts["scans"]
    total_rows = len(row_times_all)

    # ---- 1. BUILD SOLVE GRID ----
    time_bins = compute_solint_grid(row_times_all, sb.time_interval[step_idx],
                                    scans=row_scans_all)
    n_time = len(time_bins)
    freq_bins, freq_idx_bins = _compute_freq_grid(freqs, sb.freq_interval[step_idx])
    n_freq = len(freq_bins)

    if n_time == 0: return
    _log(f"      grid: {n_freq}f x {n_time}t = {n_freq*n_time} cells ({total_rows} rows)")

    # ---- 2. MEMORY STRATEGY ----
    bytes_per_row = n_chan * meta.n_corr * 33  # obs(c128) + model(c128) + flags(bool)
    avail_bytes = int((sb.memory_limit_gb if sb.memory_limit_gb > 0
                       else get_available_ram_gb() * 0.4) * 1024**3)
    avail_bytes = max(avail_bytes, 100 * 1024**2)
    max_rows_per_load = max(1, avail_bytes // bytes_per_row)

    rows_per_tbin = []
    for tb in time_bins:
        ts_set = set(tb.tolist())
        rows_per_tbin.append(sum(1 for t in row_times_all if t in ts_set))

    total_data_bytes = sum(rows_per_tbin) * bytes_per_row
    max_tbin_bytes = max(rows_per_tbin) * bytes_per_row if rows_per_tbin else 0

    if total_data_bytes <= avail_bytes:
        tier = 1
    elif max_tbin_bytes <= avail_bytes:
        tier = 2
    else:
        tier = 3

    _log(f"      memory: Tier {tier}, budget {avail_bytes/1e9:.1f} GB, "
         f"data {total_data_bytes/1e9:.2f} GB, "
         f"max_tbin {max_tbin_bytes/1e9:.2f} GB")

    # ---- 3. PRECOMPUTE PREAPPLY at t_mid (tiny) ----
    time_centres = np.array([float(np.mean(tb)) for tb in time_bins])
    preapply_at_tmid = _compute_preapply_at_tmids(
        sb, step_idx, field_name, spw, freqs, time_centres, meta, internal_stack,
        jones_keys=jones_keys)

    # ---- 4. ALLOCATE OUTPUT ----
    jones_grid = np.full((n_ant, n_freq, n_time, 2, 2), np.nan+0j, dtype=np.complex128)
    conv_grid = np.zeros((n_freq, n_time), dtype=bool)
    niter_grid = np.zeros((n_freq, n_time), dtype=np.int32)
    cost_grid = np.zeros((n_freq, n_time), dtype=np.float64)
    native_cells: Dict[Tuple[int, int], Dict] = {}
    freq_centres = np.array([float(np.mean(fb)) for fb in freq_bins])
    n_workers = sb.n_workers if sb.n_workers > 0 else max(
        1, min(os.cpu_count() - 1, n_freq * n_time))

    # ---- 5. PROCESS EACH TIME_BIN ----
    for ti in range(n_time):
        tb = time_bins[ti]
        nr = rows_per_tbin[ti]
        t_lo = float(tb.min()) - 0.01
        t_hi = float(tb.max()) + 0.01

        if nr * bytes_per_row <= avail_bytes:
            # ---- TIER 1/2: one time_bin fits in one read ----
            _log(f"      [{ti+1}/{n_time}] {nr} rows ({nr*bytes_per_row/1e9:.2f} GB) — single read")
            d = read_data(sb.ms, spw, fields=[field_name], scans=field_scans,
                          data_col=sb.data_col, model_col=sb.model_col,
                          chan_slice=chan_sl, time_range=(t_lo, t_hi))
            if not d: continue
            vis_r, mod_r, fl_r = d["vis_obs"], d["vis_model"], d["flags"]
            a1r = np.array([ant_remap.get(a, a) for a in d["ant1"]], dtype=np.int32)
            a2r = np.array([ant_remap.get(a, a) for a in d["ant2"]], dtype=np.int32)
            del d

            # Flag
            _log(f"        flagging...")
            vis_r[fl_r] = 0.0; mod_r[fl_r] = 0.0
            fl_r = flag_rfi_raw(vis_r, fl_r, sb.rfi_threshold)
            vis_r[fl_r] = 0.0; mod_r[fl_r] = 0.0

            # Average + build cells
            _log(f"        averaging {n_freq} freq bin(s)...")
            tasks = _build_cells_from_raw(
                vis_r, mod_r, fl_r, a1r, a2r,
                freq_bins, freq_idx_bins, freq_dep, n_ant, ti, preapply_at_tmid)
            del vis_r, mod_r, fl_r, a1r, a2r; gc.collect()

        else:
            # ---- TIER 3: chunk within this time_bin (running accumulator) ----
            _log(f"      [{ti+1}/{n_time}] {nr} rows ({nr*bytes_per_row/1e9:.2f} GB) — chunked reads")
            n_bl = n_ant * (n_ant - 1) // 2
            accum = {}
            for fi in range(n_freq):
                nc_bin = len(freq_idx_bins[fi])
                sh = (n_bl, nc_bin, meta.n_corr) if freq_dep else (n_bl, meta.n_corr)
                accum[fi] = {"sum_v": np.zeros(sh, np.complex128),
                             "sum_m": np.zeros(sh, np.complex128),
                             "count": np.zeros(sh, np.float64),
                             "bl_a1": None, "bl_a2": None}

            unique_t = np.sort(np.unique(tb))
            ci = 0
            while ci < len(unique_t):
                ce, est = ci, 0
                while ce < len(unique_t):
                    tr = sum(1 for t in row_times_all if t == unique_t[ce])
                    if est + tr > max_rows_per_load and est > 0: break
                    est += tr; ce += 1
                if ce == ci: ce = ci + 1

                ct_lo = float(unique_t[ci]) - 0.01
                ct_hi = float(unique_t[min(ce, len(unique_t)) - 1]) + 0.01
                d = read_data(sb.ms, spw, fields=[field_name], scans=field_scans,
                              data_col=sb.data_col, model_col=sb.model_col,
                              chan_slice=chan_sl, time_range=(ct_lo, ct_hi))
                if d:
                    vr, mr, fr = d["vis_obs"], d["vis_model"], d["flags"]
                    a1r = np.array([ant_remap.get(a, a) for a in d["ant1"]], dtype=np.int32)
                    a2r = np.array([ant_remap.get(a, a) for a in d["ant2"]], dtype=np.int32)
                    del d
                    vr[fr] = 0.0; mr[fr] = 0.0
                    fr = flag_rfi_raw(vr, fr, sb.rfi_threshold)
                    vr[fr] = 0.0; mr[fr] = 0.0
                    for fi, (_, f_idx) in enumerate(zip(freq_bins, freq_idx_bins)):
                        _accumulate_baselines(
                            vr[:, f_idx, :], mr[:, f_idx, :], fr[:, f_idx, :],
                            a1r, a2r, n_ant, freq_dep, accum[fi])
                    del vr, mr, fr, a1r, a2r; gc.collect()
                ci = ce

            # Finalize accumulators → cells
            tasks = []
            for fi in range(n_freq):
                ac = accum[fi]
                if ac["bl_a1"] is None: continue
                mask = ac["count"] > 0
                avg_v = np.where(mask, ac["sum_v"] / np.where(mask, ac["count"], 1), 0)
                avg_m = np.where(mask, ac["sum_m"] / np.where(mask, ac["count"], 1), 0)
                avg_v_22 = raw_to_2x2(avg_v); avg_m_22 = raw_to_2x2(avg_m)
                if preapply_at_tmid is not None and ti < len(preapply_at_tmid):
                    J_pre = preapply_at_tmid[ti]
                    if J_pre is not None:
                        avg_v_22 = _unapply_on_averaged(J_pre, avg_v_22, ac["bl_a1"], ac["bl_a2"])
                tasks.append({"fi": fi, "ti": ti, "vis": avg_v_22, "model": avg_m_22,
                              "a1": ac["bl_a1"], "a2": ac["bl_a2"], "freqs": freq_bins[fi]})
            del accum; gc.collect()

        # SOLVE
        _solve_tasks(tasks, solver, n_ant, n_workers,
                     jones_grid, conv_grid, niter_grid, cost_grid, native_cells)
        del tasks; gc.collect()

    # ---- 6. FLAG + SAVE ----
    sol_flags = _flag_solutions(jones_grid)
    native_save = _build_native_save(native_cells, jones_type, n_freq, n_time, n_ant)

    fid = meta.field_names.index(field_name) if field_name in meta.field_names else 0
    meta_dict = {
        "field_ra": meta.field_ra[fid], "field_dec": meta.field_dec[fid],
        "solint_s": sb.time_interval[step_idx],
        "freqint_hz": sb.freq_interval[step_idx] or 0.0,
        "phase_only": sb.phase_only[step_idx],
        "ref_ant": sb.ref_ant,
        "ref_ant_name": meta.ant_names[sb.ref_ant],
        "ant_names": json.dumps(active_names),
        "n_active": n_active,
        "ms": sb.ms,
        "solver_backend": sb.solver_backend,
        "apply_parang": sb.apply_parang,
        "scan_ids": json.dumps(field_scans or []),
        "preapply_chain": json.dumps(_describe_preapply(sb, step_idx)),
        "n_cells_total": n_freq * n_time,
        "n_cells_converged": int(conv_grid.sum()),
        "n_batches": n_time,  # one read per time_bin
    }

    save_solutions(
        path=sb.output, jones_type=jones_type, field_name=field_name, spw=spw,
        jones=jones_grid, flags=sol_flags, times=time_centres, freqs=freq_centres,
        solver_stats={"converged": conv_grid, "n_iter": niter_grid, "cost": cost_grid},
        meta=meta_dict, native_params=native_save,
        provenance={"ms": sb.ms, "jones_chain": sb.jones, "step_idx": step_idx,
                     "field": field_name, "ref_ant": sb.ref_ant,
                     "preapply_chain": _describe_preapply(sb, step_idx)})

    _log(f"      saved {jones_type}/{field_name}/spw_{spw}  "
         f"shape={jones_grid.shape} conv={int(conv_grid.sum())}/{conv_grid.size}")


# ============================================================
# HELPERS
# ============================================================

def _solve_tasks(tasks, solver, n_ant, n_workers,
                 jones_grid, conv_grid, niter_grid, cost_grid, native_cells):
    """Dispatch tasks to solver, fill grids."""
    import time as _t

    def _do(task):
        t0 = _t.time()
        r = solver.solve(
            vis_obs=task["vis"], vis_model=task["model"],
            ant1=task["a1"], ant2=task["a2"],
            freqs=task["freqs"], n_ant=n_ant)
        dt = _t.time() - t0
        logger.info(f"        cell ({task['fi']},{task['ti']}): "
                     f"{'OK' if r.get('converged') else 'FAIL'} "
                     f"iter={r.get('n_iter',0)} cost={r.get('cost',0):.2e} "
                     f"{dt:.1f}s")
        return r

    def _store(fi, ti, r):
        jones_grid[:, fi, ti] = r["jones"]
        conv_grid[fi, ti] = r.get("converged", False)
        niter_grid[fi, ti] = r.get("n_iter", 0)
        cost_grid[fi, ti] = r.get("cost", 0.0)
        native_cells[(fi, ti)] = r.get("native_params")

    n_tasks = len(tasks)
    logger.info(f"      solving {n_tasks} cell(s) with {n_workers} worker(s)...")

    if n_workers > 1 and n_tasks > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_do, t): (t["fi"], t["ti"]) for t in tasks}
            for fut in as_completed(futs):
                fi, ti = futs[fut]
                try: _store(fi, ti, fut.result())
                except Exception as e: logger.error(f"      cell ({fi},{ti}): {e}")
    else:
        for task in tasks:
            try: _store(task["fi"], task["ti"], _do(task))
            except Exception as e: logger.error(f"      cell ({task['fi']},{task['ti']}): {e}")


def _unapply_on_averaged(J_pre, avg_vis, bl_a1, bl_a2):
    """Correct averaged data: V_corr = J_i^{-1} V J_j^{-H}.
    
    J_pre and avg_vis must have matching dimensions.
    J: (n_ant, 2, 2) for freq-averaged cells
    vis: (n_bl, 2, 2) or (n_bl, n_chan, 2, 2)
    
    For freq-averaged vis (n_bl, 2, 2), add dummy chan dim for numba kernel.
    """
    if avg_vis.ndim == 3:
        # (n_bl, 2, 2) → add dummy chan → (n_bl, 1, 2, 2)
        out = unapply_jones_to_rows(
            J_pre, avg_vis[:, None, :, :], bl_a1, bl_a2)
        return out[:, 0]  # back to (n_bl, 2, 2)
    return unapply_jones_to_rows(J_pre, avg_vis, bl_a1, bl_a2)


def _build_cells_from_raw(vis_raw, mod_raw, fl_raw, a1r, a2r,
                           freq_bins, freq_idx_bins, freq_dep, n_ant, ti,
                           preapply_at_tmid):
    """Average raw data into tiny 2x2 cells for each freq_bin."""
    tasks = []
    for fi, (fb_f, f_idx) in enumerate(zip(freq_bins, freq_idx_bins)):
        vf = vis_raw[:, f_idx, :]
        mf = mod_raw[:, f_idx, :]
        ff = fl_raw[:, f_idx, :]

        if freq_dep:
            avg_v, bl_a1, bl_a2 = average_per_baseline_time_only(
                vf, ff, a1r, a2r, n_ant)
            avg_m, _, _ = average_per_baseline_time_only(
                mf, ff, a1r, a2r, n_ant)
        else:
            avg_v, bl_a1, bl_a2 = average_per_baseline_full(
                vf, ff, a1r, a2r, n_ant)
            avg_m, _, _ = average_per_baseline_full(
                mf, ff, a1r, a2r, n_ant)

        avg_v_22 = raw_to_2x2(avg_v); del avg_v
        avg_m_22 = raw_to_2x2(avg_m); del avg_m

        if preapply_at_tmid is not None and ti < len(preapply_at_tmid):
            J_pre = preapply_at_tmid[ti]
            if J_pre is not None:
                avg_v_22 = _unapply_on_averaged(J_pre, avg_v_22, bl_a1, bl_a2)

        tasks.append({"fi": fi, "ti": ti, "vis": avg_v_22, "model": avg_m_22,
                       "a1": bl_a1, "a2": bl_a2, "freqs": fb_f})
    return tasks


def _accumulate_baselines(vf, mf, ff, ant1, ant2, n_ant, freq_dep, accum):
    """Accumulate per-baseline sums for running average (Tier 3).
    vf/mf/ff: (n_row, n_chan_bin, n_corr) or (n_row, n_corr).
    accum: dict with sum_v, sum_m, count, bl_a1, bl_a2."""
    from .core.averaging import _build_bl_map
    bl_map, oa1, oa2, n_bl = _build_bl_map(n_ant)

    if accum["bl_a1"] is None:
        accum["bl_a1"] = oa1
        accum["bl_a2"] = oa2

    nr = vf.shape[0]
    for r in range(nr):
        a1, a2 = int(ant1[r]), int(ant2[r])
        if a1 >= a2: continue
        bl = bl_map[a1, a2]
        if bl < 0: continue
        if freq_dep:
            # vf: (n_row, n_chan, n_corr)
            for c in range(vf.shape[1]):
                for p in range(vf.shape[2]):
                    if not ff[r, c, p]:
                        accum["sum_v"][bl, c, p] += vf[r, c, p]
                        accum["sum_m"][bl, c, p] += mf[r, c, p]
                        accum["count"][bl, c, p] += 1.0
        else:
            # vf: (n_row, n_chan, n_corr) but we want full average
            for c in range(vf.shape[1]):
                for p in range(vf.shape[2]):
                    if not ff[r, c, p]:
                        accum["sum_v"][bl, p] += vf[r, c, p]
                        accum["sum_m"][bl, p] += mf[r, c, p]
                        accum["count"][bl, p] += 1.0


def _compute_preapply_at_tmids(sb, step_idx, field_name, spw, freqs,
                                time_centres, meta, internal_stack,
                                jones_keys=None):
    """Compute composed preapply Jones at each t_mid.
    Returns list of (n_ant, 2, 2) or None per time_bin."""
    # No preapply needed for step 0 without parang or external
    has_parang = sb.apply_parang
    has_external = sb.external_preapply is not None
    has_internal = step_idx > 0

    if not has_parang and not has_external and not has_internal:
        return None

    parts = []
    if has_parang: parts.append("parang")
    if has_external: parts.append(f"external({len(sb.external_preapply.jones)})")
    if has_internal: parts.append(f"internal({step_idx} prior steps)")
    _log(f"      preapply: {' + '.join(parts)} at t_mid")

    fid = meta.field_names.index(field_name) if field_name in meta.field_names else 0
    tgt_ra = meta.field_ra[fid]
    tgt_dec = meta.field_dec[fid]

    result = []
    for t_mid in time_centres:
        jones_list = []

        # Parang at t_mid
        if has_parang:
            try:
                from .jones.parang import compute_parallactic_angles, parang_to_jones
                from .jones.algebra import detect_feed_basis
                fb = detect_feed_basis(sb.ms)
                pa = compute_parallactic_angles(
                    sb.ms, np.array([t_mid]), field=field_name)
                P = parang_to_jones(pa[0], fb)
                jones_list.append(P)
            except Exception as e:
                logger.warning(f"      parang failed: {e}")

        # External preapply at t_mid
        if has_external:
            ep = sb.external_preapply
            for ei, (etab, ej, efs, eti) in enumerate(zip(
                    ep.tables, ep.jones, ep.field_select, ep.time_interp)):
                try:
                    fdata = load_all_fields(etab, ej, spw)
                    if not fdata: continue
                    fmt = _fmt_fields(fdata, ej)
                    epf = ep.fields[ei] if ep.fields and ei < len(ep.fields) else None
                    J = interpolate_jones_multifield(
                        fmt, np.array([t_mid]), freqs, efs, eti,
                        target_ra=tgt_ra, target_dec=tgt_dec, pinned_fields=epf)
                    if J is not None and J.ndim >= 3:
                        Jt = J[0]  # (n_ant, [n_f,] 2, 2)
                        # Collapse freq if present → always (n_ant, 2, 2)
                        if Jt.ndim == 4:
                            Jt = Jt.mean(axis=1)
                        jones_list.append(Jt)
                except Exception as e:
                    logger.warning(f"      external preapply failed: {e}")

        # Internal chain at t_mid
        if has_internal:
            for pi in range(step_idx):
                pjt = jones_keys[pi] if jones_keys else sb.jones[pi]
                if pjt not in internal_stack: continue
                try:
                    fmt = _fmt_fields(internal_stack[pjt], pjt)
                    if not fmt: continue
                    J = interpolate_jones_multifield(
                        fmt, np.array([t_mid]), freqs, "nearest_time",
                        sb.preapply_time_interp[pi],
                        target_ra=tgt_ra, target_dec=tgt_dec)
                    if J is not None and J.ndim >= 3:
                        Jt = J[0]
                        if Jt.ndim == 4:
                            Jt = Jt.mean(axis=1)
                        jones_list.append(Jt)
                except Exception as e:
                    logger.warning(f"      internal preapply failed: {e}")

        if jones_list:
            J_total = compose_jones_chain(jones_list)
            result.append(J_total)
        else:
            result.append(None)

    return result


def _flag_solutions(jones):
    """flags.shape = jones.shape[:-2]"""
    fshape = jones.shape[:-2]
    flat = jones.reshape(-1, 2, 2)
    fl = np.zeros(flat.shape[0], dtype=bool)
    for i in range(flat.shape[0]):
        J = flat[i]
        if np.any(np.isnan(J)) or np.any(np.isinf(J)):
            fl[i] = True
        elif np.abs(J[0, 0]*J[1, 1] - J[0, 1]*J[1, 0]) < 1e-20:
            fl[i] = True
    n = fl.sum()
    if n: logger.info(f"      flagged {n}/{fl.size} slots")
    return fl.reshape(fshape)


def _compute_freq_grid(freqs, freq_int_hz):
    if freq_int_hz is None or len(freqs) == 0:
        return [freqs], [np.arange(len(freqs))]
    blocks, idx_blocks = [], []
    cf, ci = [freqs[0]], [0]
    for i in range(1, len(freqs)):
        if freqs[i] - freqs[ci[0]] >= freq_int_hz:
            blocks.append(np.array(cf)); idx_blocks.append(np.array(ci))
            cf, ci = [freqs[i]], [i]
        else:
            cf.append(freqs[i]); ci.append(i)
    if cf:
        blocks.append(np.array(cf)); idx_blocks.append(np.array(ci))
    return blocks, idx_blocks


def _build_native_save(cells, jtype, nf, nt, na):
    if not cells: return None
    first = next((v for v in cells.values() if v), None)
    if not first: return None
    tp = first.get("type", jtype)
    r = {"type": tp}
    if tp == "K" and "delay" in first:
        a = np.zeros((na, nf, nt, 2), dtype=np.float64)
        for (fi, ti), d in cells.items():
            if d and "delay" in d: a[:, fi, ti, :] = d["delay"]
        r["delay"] = a
    elif tp == "G":
        amp = np.ones((na, nf, nt, 2)); ph = np.zeros((na, nf, nt, 2))
        for (fi, ti), d in cells.items():
            if d:
                if "amp" in d: amp[:, fi, ti, :] = d["amp"]
                if "phase" in d: ph[:, fi, ti, :] = d["phase"]
        r["amp"] = amp; r["phase"] = ph
    elif tp == "D":
        dpq = np.zeros((na, nf, nt), dtype=np.complex128)
        dqp = np.zeros((na, nf, nt), dtype=np.complex128)
        for (fi, ti), d in cells.items():
            if d:
                if "d_pq" in d: dpq[:, fi, ti] = d["d_pq"]
                if "d_qp" in d: dqp[:, fi, ti] = d["d_qp"]
        r["d_pq"] = dpq; r["d_qp"] = dqp
    elif tp == "KC":
        a = np.zeros((nf, nt))
        for (fi, ti), d in cells.items():
            if d and "tau_cross" in d: a[fi, ti] = d["tau_cross"]
        r["tau_cross"] = a
    elif tp == "CP":
        a = np.zeros((nf, nt))
        for (fi, ti), d in cells.items():
            if d and "phi_cross" in d: a[fi, ti] = d["phi_cross"]
        r["phi_cross"] = a
    return r


def _fmt_fields(fields_data, jones_type):
    out = {}
    for fn, sol in fields_data.items():
        native = sol.get("native_params") or {}
        if native and "type" not in native: native["type"] = jones_type
        out[fn] = {"times": sol["time"], "freqs": sol.get("freq"),
                    "jones": sol["jones"], "ra_rad": sol.get("ra_rad", 0),
                    "dec_rad": sol.get("dec_rad", 0),
                    "native_params": native or None}
    return out


def _describe_preapply(sb, step_idx):
    chain = []
    if sb.apply_parang: chain.append({"source": "parang", "jones": "P"})
    if sb.external_preapply:
        for i, jt in enumerate(sb.external_preapply.jones):
            chain.append({"source": "external", "table": sb.external_preapply.tables[i], "jones": jt})
    for pi in range(step_idx):
        chain.append({"source": "internal", "jones": sb.jones[pi], "table": sb.output})
    return chain


def flag_rfi_raw(vis, flags, threshold):
    """RFI flag on raw (n_row, n_chan, n_corr). All corrs independently.
    Memory-efficient: no full-array copies."""
    n_corr = vis.shape[2]
    bad_any = np.zeros(vis.shape[:2], dtype=bool)
    for p in range(n_corr):
        amp = np.abs(vis[:, :, p])  # (n_row, n_chan) float64
        mask = ~flags[:, :, p]
        clean = amp[mask]  # 1-D subset
        if len(clean) < 10:
            del amp, clean
            continue
        med = np.median(clean)
        mad = np.median(np.abs(clean - med))
        del clean  # free immediately
        if mad < 1e-30:
            del amp
            continue
        cutoff = med + threshold * 1.4826 * mad
        bad = (amp > cutoff) & mask
        bad_any |= bad
        del amp, bad
    if bad_any.any():
        n_flagged = int(bad_any.sum())
        for p in range(n_corr):
            flags[bad_any, p] = True
        logger.debug(f"RFI: flagged {n_flagged} samples")
    del bad_any
    return flags
