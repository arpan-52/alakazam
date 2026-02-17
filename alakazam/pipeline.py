"""
ALAKAZAM Pipeline.

Executes solve: and apply: blocks from config.

Solve loop: Jones (user order) → SPW → 3-tier load → parallel cells
Apply loop: load solutions → compose chain → write CORRECTED_DATA

3-tier memory management:
  T1: Full SPW fits     → single load, all cells parallel
  T2: N slots fit       → batched load, N slots at a time
  T3: One slot too big  → pseudo-chunk (progressive accumulation)

Workers NEVER touch disk. Data shared via fork copy-on-write.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import multiprocessing as mp
import os
import logging
import time as _time
import gc

from .config import AlakazamConfig, SolveBlock, ApplyBlock, SolveStep, config_to_yaml
from .core.ms_io import detect_metadata, compute_solint_grid, parse_spw_selection, MSMetadata
from .core.averaging import average_per_baseline_time_only, average_per_baseline_full
from .core.rfi import flag_rfi
from .core.quality import compute_quality
from .core.memory import get_available_ram_gb
from .solvers.registry import get_solver
from .jones import (
    jones_unapply, jones_unapply_freq,
    delay_to_jones, crossdelay_to_jones,
    compose_jones_chain,
)
from .io.hdf5 import save_solutions

logger = logging.getLogger("alakazam")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console() if HAS_RICH else None
GB = 1024**3


def _log(msg: str, style: str = ""):
    logger.info(msg)
    if console and style:
        console.print(msg, style=style)
    elif console:
        console.print(msg)


# ===== Shared batch data (fork COW) =====
_batch_data: Dict[str, Any] = {}


# ===================================================================
# PUBLIC ENTRY POINT
# ===================================================================

def run_pipeline(config: AlakazamConfig):
    """Execute all solve and apply blocks."""
    t0 = _time.time()

    if console and HAS_RICH:
        console.print(Panel(
            "[bold cyan]ALAKAZAM[/bold cyan] — Radio Interferometric Calibration\n"
            "Arpan Pal 2026, NRAO / NCRA", style="bold blue"))

    for i, sb in enumerate(config.solve_blocks):
        _log(f"\n{'='*60}", style="bold green")
        _log(f"SOLVE BLOCK {i+1}/{len(config.solve_blocks)}: {sb.ms}", style="bold green")
        _run_solve_block(sb)

    for i, ab in enumerate(config.apply_blocks):
        _log(f"\n{'='*60}", style="bold yellow")
        _log(f"APPLY BLOCK {i+1}/{len(config.apply_blocks)}: {ab.ms}", style="bold yellow")
        _run_apply_block(ab)

    _log(f"\nDone in {_time.time()-t0:.1f}s", style="bold green")


# ===================================================================
# SOLVE BLOCK
# ===================================================================

def _run_solve_block(sb: SolveBlock):
    """Run one solve: block."""
    global _batch_data

    from casacore.tables import table
    spw_tab = table(f"{sb.ms}::SPECTRAL_WINDOW", readonly=True, ack=False)
    n_spw_total = spw_tab.nrows()
    spw_tab.close()
    selected_spws = parse_spw_selection(sb.spw, n_spw_total)

    _log(f"  Chain: {' → '.join(s.jones_type for s in sb.steps)}")
    _log(f"  SPWs:  {selected_spws}")
    _log(f"  Output: {sb.output}")

    meta_per_spw: Dict[int, MSMetadata] = {}
    for sid in selected_spws:
        meta_per_spw[sid] = detect_metadata(
            sb.ms, field=sb.field, spw_id=sid, scans=sb.scans)

    first_meta = meta_per_spw[selected_spws[0]]
    obs_meta = {
        "ms_path": sb.ms, "n_ant": first_meta.n_ant,
        "antenna_names": first_meta.antenna_names,
        "working_antennas": first_meta.working_antennas,
        "non_working_antennas": first_meta.non_working_antennas,
        "feed_basis": first_meta.feed_basis,
        "field": sb.field or "",
    }

    n_workers = min(os.cpu_count() or 1, 8)
    avail_gb = _get_avail_gb(sb.memory_limit_gb)

    all_solutions: Dict[str, Dict[int, Dict]] = {}
    prev_solved: Dict[int, List[Dict]] = {s: [] for s in selected_spws}

    # === Jones outer loop (user order) ===
    for step_idx, step in enumerate(sb.steps):
        jt = step.jones_type
        _log(f"\n  Step {step_idx+1}/{len(sb.steps)}: {jt} "
             f"(t={step.time_interval}, f={step.freq_interval})", style="bold cyan")

        if jt not in all_solutions:
            all_solutions[jt] = {}

        # === SPW loop ===
        for spw_id in selected_spws:
            meta = meta_per_spw[spw_id]
            t_spw = _time.time()

            time_edges, freq_chunks = compute_solint_grid(
                meta.unique_times, meta.freq,
                step.time_interval, step.freq_interval)
            nt = len(time_edges) - 1
            nf = len(freq_chunks)
            total = nt * nf

            _log(f"    SPW {spw_id}: {nt}×{nf}={total} cells, "
                 f"{len(meta.working_antennas)}/{meta.n_ant} ants")

            # Memory estimation
            where_full = _build_where(sb, spw_id)
            n_rows = _count_rows(sb.ms, where_full)
            mem_full = _est_gb(n_rows, meta.n_freq)
            rows_per_slot = max(1, n_rows // max(1, nt))
            mem_slot = _est_gb(rows_per_slot, meta.n_freq)

            _log(f"    Mem: full={mem_full:.1f}GB slot={mem_slot:.2f}GB budget={avail_gb:.1f}GB")

            # Allocate solution arrays
            R = _SolArrays(nt, nf, meta.n_ant)

            # === TIER 1: full load ===
            if mem_full <= avail_gb:
                _log(f"    T1: single load")
                batch = _load(sb.ms, where_full, meta, sb)
                if batch is not None:
                    _preapply(batch, prev_solved[spw_id], meta)
                    batch["freq"] = meta.freq
                    _dispatch_and_collect(batch, time_edges, freq_chunks,
                                         0, nt, step, meta, n_workers, R)
                    del batch; _batch_data.clear(); gc.collect()

            # === TIER 2: batched ===
            elif mem_slot <= avail_gb:
                slots_per = max(1, int(avail_gb / mem_slot))
                nb = max(1, int(np.ceil(nt / slots_per)))
                _log(f"    T2: {nb} batches × {slots_per} slots")
                for bi in range(nb):
                    ts = bi * slots_per
                    te = min(ts + slots_per, nt)
                    w = _build_where(sb, spw_id, time_edges[ts], time_edges[te])
                    batch = _load(sb.ms, w, meta, sb)
                    if batch is None:
                        continue
                    _preapply(batch, prev_solved[spw_id], meta)
                    batch["freq"] = meta.freq
                    _dispatch_and_collect(batch, time_edges, freq_chunks,
                                         ts, te, step, meta, n_workers, R)
                    del batch; _batch_data.clear(); gc.collect()

            # === TIER 3: pseudo-chunk ===
            else:
                _log(f"    T3: pseudo-chunking")
                for ti in range(nt):
                    for fi in range(nf):
                        res = _pseudo_chunk(
                            sb, spw_id, meta,
                            time_edges[ti], time_edges[ti+1],
                            freq_chunks[fi], prev_solved[spw_id],
                            step, avail_gb)
                        if res:
                            _store_one(ti, fi, res, R, meta)

            # Count
            ok = int(np.sum(R.q_success))
            empty = total - int(np.sum(R.weights > 0))
            fail = total - ok - empty

            # Freq/time centers
            freq_c = np.array([np.mean(meta.freq[fi]) for fi in freq_chunks])
            time_c = 0.5 * (time_edges[:-1] + time_edges[1:])

            all_solutions[jt][spw_id] = R.to_dict(
                time_c, freq_c, meta, step)

            # Representative for chaining
            _update_prev(R, prev_solved[spw_id], jt, meta)

            _print_summary(jt, spw_id, nt, nf, ok, fail, empty,
                           R.q_chi2, R.q_success, meta, _time.time()-t_spw)

    save_solutions(sb.output, all_solutions, obs_meta, config_to_yaml(
        AlakazamConfig(solve_blocks=[sb])))
    _log(f"  Saved: {sb.output}", style="bold green")


# ===================================================================
# APPLY BLOCK
# ===================================================================

def _run_apply_block(ab: ApplyBlock):
    """Run one apply: block."""
    from casacore.tables import table, taql
    from .io.hdf5 import load_solutions
    from .jones import compose_jones_chain, jones_unapply_freq, delay_to_jones, crossdelay_to_jones
    from .core.interpolation import interpolate_jones_freq

    _log(f"  Jones: {ab.jones}")
    _log(f"  Tables: {ab.tables}")
    _log(f"  Output: {ab.output_col}")

    spw_tab = table(f"{ab.ms}::SPECTRAL_WINDOW", readonly=True, ack=False)
    n_spw = spw_tab.nrows()
    freq_all = [spw_tab.getcol("CHAN_FREQ")[i] for i in range(n_spw)]
    spw_tab.close()

    spw_ids = parse_spw_selection(ab.spw, n_spw)

    # Load all solution tables
    sol_data = {}
    for jt, tbl in zip(ab.jones, ab.tables):
        sol_data[(jt, tbl)] = load_solutions(tbl, jones_types=[jt])

    for spw_id in spw_ids:
        freq = freq_all[spw_id]
        n_chan = len(freq)

        ms = table(ab.ms, readonly=False, ack=False)

        conds = [f"DATA_DESC_ID=={spw_id}"]
        if ab.field:
            field_tab = table(f"{ab.ms}::FIELD", readonly=True, ack=False)
            fn = list(field_tab.getcol("NAME"))
            field_tab.close()
            if ab.field in fn:
                conds.append(f"FIELD_ID=={fn.index(ab.field)}")

        sel = taql(f"SELECT * FROM $ms WHERE {' AND '.join(conds)}")
        if sel.nrows() == 0:
            sel.close(); ms.close()
            continue

        data = sel.getcol("DATA")
        ant1 = sel.getcol("ANTENNA1").astype(np.int32)
        ant2 = sel.getcol("ANTENNA2").astype(np.int32)

        n_row, n_ch, n_corr = data.shape
        if n_corr == 4:
            vis = data.reshape(n_row, n_ch, 2, 2).astype(np.complex128)
        elif n_corr == 2:
            vis = np.zeros((n_row, n_ch, 2, 2), dtype=np.complex128)
            vis[:, :, 0, 0] = data[:, :, 0]
            vis[:, :, 1, 1] = data[:, :, 1]
        else:
            vis = np.zeros((n_row, n_ch, 2, 2), dtype=np.complex128)
            vis[:, :, 0, 0] = data[:, :, 0]

        # Build Jones chain
        chain = []
        for jt, tbl in zip(ab.jones, ab.tables):
            sols = sol_data[(jt, tbl)]
            if jt not in sols:
                continue
            spw_key = spw_id if spw_id in sols[jt] else list(sols[jt].keys())[0]
            sol = sols[jt][spw_key]
            j = sol["jones"]  # (nt, nf, nant, 2, 2)
            # Take first time slot, first freq slot
            j0 = j[0, 0]  # (nant, 2, 2)

            params = sol.get("params", {})
            if jt == "K" and "delay" in params:
                d = params["delay"]
                if d.ndim > 1:
                    d = d[0, 0]  # first time/freq slot
                j_freq = delay_to_jones(d, freq)
                chain.append(j_freq)
            elif jt == "KCROSS" and "cross_delay" in params:
                tau = params["cross_delay"]
                if tau.ndim > 1:
                    tau = tau[0, 0]
                j_freq = crossdelay_to_jones(tau, freq)
                chain.append(j_freq)
            else:
                # Broadcast freq-independent to all channels
                j_bc = np.broadcast_to(j0[:, np.newaxis, :, :],
                                       (j0.shape[0], n_chan, 2, 2)).copy()
                chain.append(j_bc)

        if chain:
            J = compose_jones_chain(chain)
            if J is not None:
                vis = jones_unapply_freq(J, vis, ant1, ant2)

        # Write back
        if n_corr == 4:
            out = vis.reshape(n_row, n_ch, 4)
        elif n_corr == 2:
            out = np.zeros((n_row, n_ch, 2), dtype=np.complex128)
            out[:, :, 0] = vis[:, :, 0, 0]
            out[:, :, 1] = vis[:, :, 1, 1]
        else:
            out = data.astype(np.complex128)
            out[:, :, 0] = vis[:, :, 0, 0]

        if ab.output_col not in sel.colnames():
            from casacore.tables import makecoldesc
            cd = makecoldesc(ab.output_col, sel.getcoldesc("DATA"))
            ms.addcols(cd)

        sel.putcol(ab.output_col, out.astype(data.dtype))
        sel.close(); ms.close()

        _log(f"    SPW {spw_id}: wrote {n_row} rows to {ab.output_col}")


# ===================================================================
# INTERNALS: loading, dispatching, pseudo-chunking
# ===================================================================

def _get_avail_gb(limit: float) -> float:
    if limit > 0:
        return limit
    return get_available_ram_gb() * 0.5


def _est_gb(n_rows: int, n_chan: int) -> float:
    return (2 * n_rows * n_chan * 4 * 16 + n_rows * n_chan * 4 + n_rows * 16) * 1.5 / GB


def _build_where(sb: SolveBlock, spw_id: int,
                 t_lo: float = None, t_hi: float = None) -> str:
    from casacore.tables import table
    conds = [f"DATA_DESC_ID=={spw_id}"]
    if sb.field:
        ft = table(f"{sb.ms}::FIELD", readonly=True, ack=False)
        fn = list(ft.getcol("NAME")); ft.close()
        if sb.field in fn:
            conds.append(f"FIELD_ID=={fn.index(sb.field)}")
    if sb.scans:
        from .core.ms_io import _parse_scan_selection
        sids = _parse_scan_selection(sb.scans)
        conds.append(f"SCAN_NUMBER IN [{','.join(map(str, sids))}]")
    if t_lo is not None:
        conds.append(f"TIME>={t_lo}")
    if t_hi is not None:
        conds.append(f"TIME<{t_hi}")
    return " AND ".join(conds)


def _count_rows(ms_path: str, where: str) -> int:
    from casacore.tables import table, taql
    ms = table(ms_path, readonly=True, ack=False)
    sel = taql(f"SELECT FROM $ms WHERE {where}")
    n = sel.nrows(); sel.close(); ms.close()
    return n


def _load(ms_path: str, where: str, meta: MSMetadata,
          sb: SolveBlock) -> Optional[Dict]:
    from casacore.tables import table, taql
    ms = table(ms_path, readonly=True, ack=False)
    sel = taql(f"SELECT * FROM $ms WHERE {where}")
    if sel.nrows() == 0:
        sel.close(); ms.close(); return None

    a1 = sel.getcol("ANTENNA1").astype(np.int32)
    a2 = sel.getcol("ANTENNA2").astype(np.int32)
    time = sel.getcol("TIME")

    ws = set(int(a) for a in meta.working_antennas)
    ok = np.array([int(x) in ws and int(y) in ws for x, y in zip(a1, a2)])
    if not np.any(ok):
        sel.close(); ms.close(); return None

    a1 = a1[ok]; a2 = a2[ok]; time = time[ok]
    data = sel.getcol(sb.data_col)[ok]
    model = sel.getcol(sb.model_col)[ok] if sb.model_col in sel.colnames() else np.ones_like(data)
    flags = sel.getcol("FLAG")[ok]
    sel.close(); ms.close()

    vis, mod, fl = _reshape(data, model, flags)
    del data, model, flags
    return {"vis_obs": vis, "vis_model": mod, "flags": fl,
            "ant1": a1, "ant2": a2, "time": time}


def _reshape(data, model, flags):
    n, nc, ncorr = data.shape
    if ncorr == 4:
        return (np.ascontiguousarray(data.reshape(n, nc, 2, 2), dtype=np.complex128),
                np.ascontiguousarray(model.reshape(n, nc, 2, 2), dtype=np.complex128),
                np.ascontiguousarray(flags.reshape(n, nc, 2, 2)))
    vis = np.zeros((n, nc, 2, 2), dtype=np.complex128)
    mod = np.zeros_like(vis); fl = np.ones((n, nc, 2, 2), dtype=bool)
    vis[:,:,0,0] = data[:,:,0]; vis[:,:,1,1] = data[:,:,min(1,ncorr-1)]
    mod[:,:,0,0] = model[:,:,0]; mod[:,:,1,1] = model[:,:,min(1,ncorr-1)]
    fl[:,:,0,0] = flags[:,:,0]; fl[:,:,1,1] = flags[:,:,min(1,ncorr-1)]
    return vis, mod, fl


def _preapply(batch: Dict, prev_jones: List[Dict], meta: MSMetadata):
    for prev in prev_jones:
        pj = _build_pre(prev, meta.freq, meta.n_ant)
        if pj is None:
            continue
        if pj.ndim == 4:
            batch["vis_obs"] = jones_unapply_freq(
                pj, batch["vis_obs"], batch["ant1"], batch["ant2"])
        elif pj.ndim == 3:
            for f in range(batch["vis_obs"].shape[1]):
                batch["vis_obs"][:, f] = jones_unapply(
                    pj, batch["vis_obs"][:, f], batch["ant1"], batch["ant2"])


def _build_pre(prev, freq, n_ant):
    jt = prev.get("jones_type", "")
    params = prev.get("params", {})
    if jt == "K" and "delay" in params:
        return delay_to_jones(params["delay"], freq)
    if jt == "KCROSS" and "cross_delay" in params:
        return crossdelay_to_jones(params["cross_delay"], freq)
    return prev.get("jones")


# ===================================================================
# DISPATCH + WORKER
# ===================================================================

def _dispatch_and_collect(batch, time_edges, freq_chunks,
                          t_start, t_end, step, meta, n_workers, R):
    global _batch_data
    _batch_data = batch

    tasks = [(ti, fi, time_edges[ti], time_edges[ti+1], freq_chunks[fi],
              step.jones_type, meta.n_ant, step.ref_ant, meta.working_antennas,
              step.phase_only, step.rfi_threshold, step.max_iter, step.tol)
             for ti in range(t_start, t_end) for fi in range(len(freq_chunks))]

    if n_workers > 1 and len(tasks) > 1:
        with mp.Pool(min(n_workers, len(tasks))) as pool:
            results = pool.map(_solve_cell, tasks)
    else:
        results = [_solve_cell(t) for t in tasks]

    for r in results:
        ti, fi, jsol, native, info = r
        if jsol is not None:
            _store_one(ti, fi, {"jones": jsol, "native": native, "info": info},
                       R, meta)
    _batch_data = {}


def _solve_cell(args) -> Tuple:
    """Worker: solve one cell from shared _batch_data. No disk I/O."""
    (ti, fi, tlo, thi, fidx,
     jt, nant, ref, wa, po, rfi_th, mi, tol) = args

    b = _batch_data
    mask = (b["time"] >= tlo) & (b["time"] < thi)
    rows = np.where(mask)[0]
    if len(rows) == 0:
        return (ti, fi, None, None, None)

    vo = b["vis_obs"][rows][:, fidx]
    vm = b["vis_model"][rows][:, fidx]
    fl = b["flags"][rows][:, fidx]
    a1 = b["ant1"][rows]; a2 = b["ant2"][rows]
    freq = b["freq"][fidx]

    if rfi_th > 0:
        fl, _ = flag_rfi(vo, fl, rfi_th)
    if np.all(fl):
        return (ti, fi, None, None, None)

    solver = get_solver(jt)
    if solver.can_avg_freq:
        va, _, b1, b2 = average_per_baseline_full(vo, fl, a1, a2, nant)
        ma, _, _, _ = average_per_baseline_full(vm, fl, a1, a2, nant)
        sf = None
    else:
        va, _, b1, b2 = average_per_baseline_time_only(vo, fl, a1, a2, nant)
        ma, _, _, _ = average_per_baseline_time_only(vm, fl, a1, a2, nant)
        sf = freq

    if len(b1) == 0:
        return (ti, fi, None, None, None)

    try:
        j, n, i = solver.solve(va, ma, b1, b2, nant, ref, wa,
                               freq=sf, phase_only=po, max_iter=mi, tol=tol)
    except Exception as e:
        return (ti, fi, None, None, {"error": str(e), "success": False})
    return (ti, fi, j, n, i)


# ===================================================================
# PSEUDO-CHUNK (TIER 3)
# ===================================================================

def _pseudo_chunk(sb, spw_id, meta, tlo, thi, fidx, prev_jones, step, avail_gb):
    from casacore.tables import table, taql
    chunk_freq = meta.freq[fidx]

    w = _build_where(sb, spw_id, tlo, thi)
    ms = table(sb.ms, readonly=True, ack=False)
    sel = taql(f"SELECT TIME FROM $ms WHERE {w}")
    if sel.nrows() == 0:
        sel.close(); ms.close(); return None
    times = np.unique(sel.getcol("TIME"))
    sel.close(); ms.close()

    nbl = meta.n_ant * (meta.n_ant - 1) // 2
    mem_ts = _est_gb(max(1, nbl), meta.n_freq)
    ts_per = max(1, int(avail_gb * 0.7 / max(mem_ts, 1e-6)))
    npcs = max(1, int(np.ceil(len(times) / ts_per)))

    sums_o = {}; sums_m = {}; counts = {}
    nc = len(fidx)

    for pc in range(npcs):
        s = pc * ts_per; e = min(s + ts_per, len(times))
        w2 = _build_where(sb, spw_id, times[s], times[e-1] + 0.01)
        batch = _load(sb.ms, w2, meta, sb)
        if batch is None:
            continue

        vo = batch["vis_obs"][:, fidx]
        vm = batch["vis_model"][:, fidx]
        fl = batch["flags"][:, fidx]
        a1 = batch["ant1"]; a2 = batch["ant2"]

        if prev_jones:
            for prev in prev_jones:
                pj = _build_pre(prev, chunk_freq, meta.n_ant)
                if pj is not None:
                    if pj.ndim == 4:
                        pjc = pj[:, fidx] if pj.shape[1] > max(fidx) else pj
                        vo = jones_unapply_freq(pjc, vo, a1, a2)
                    elif pj.ndim == 3:
                        for f in range(nc):
                            vo[:, f] = jones_unapply(pj, vo[:, f], a1, a2)

        if step.rfi_threshold > 0:
            fl, _ = flag_rfi(vo, fl, step.rfi_threshold)

        for r in range(len(a1)):
            k = (int(min(a1[r], a2[r])), int(max(a1[r], a2[r])))
            if k[0] == k[1]:
                continue
            if k not in sums_o:
                sums_o[k] = np.zeros((nc, 2, 2), dtype=np.complex128)
                sums_m[k] = np.zeros((nc, 2, 2), dtype=np.complex128)
                counts[k] = np.zeros((nc, 2, 2), dtype=np.float64)
            for c in range(nc):
                for i in range(2):
                    for j in range(2):
                        if not fl[r, c, i, j]:
                            sums_o[k][c, i, j] += vo[r, c, i, j]
                            sums_m[k][c, i, j] += vm[r, c, i, j]
                            counts[k][c, i, j] += 1.0
        del batch; gc.collect()

    if not sums_o:
        return None

    bls = sorted(sums_o.keys())
    b1 = np.array([b[0] for b in bls], dtype=np.int32)
    b2 = np.array([b[1] for b in bls], dtype=np.int32)

    solver = get_solver(step.jones_type)
    if solver.can_avg_freq:
        va = np.zeros((len(bls), 2, 2), dtype=np.complex128)
        ma = np.zeros_like(va)
        for bi, k in enumerate(bls):
            for i in range(2):
                for j in range(2):
                    tc = np.sum(counts[k][:, i, j])
                    if tc > 0:
                        va[bi, i, j] = np.sum(sums_o[k][:, i, j]) / tc
                        ma[bi, i, j] = np.sum(sums_m[k][:, i, j]) / tc
        sf = None
    else:
        va = np.zeros((len(bls), nc, 2, 2), dtype=np.complex128)
        ma = np.zeros_like(va)
        for bi, k in enumerate(bls):
            for c in range(nc):
                for i in range(2):
                    for j in range(2):
                        if counts[k][c, i, j] > 0:
                            va[bi, c, i, j] = sums_o[k][c, i, j] / counts[k][c, i, j]
                            ma[bi, c, i, j] = sums_m[k][c, i, j] / counts[k][c, i, j]
        sf = chunk_freq

    jsol, native, info = solver.solve(
        va, ma, b1, b2, meta.n_ant, step.ref_ant, meta.working_antennas,
        freq=sf, phase_only=step.phase_only,
        max_iter=step.max_iter, tol=step.tol)
    return {"jones": jsol, "native": native, "info": info}


# ===================================================================
# SOLUTION ARRAYS HELPER
# ===================================================================

class _SolArrays:
    """Pre-allocated solution arrays for one (Jones, SPW)."""
    def __init__(self, nt, nf, nant):
        self.jones = np.full((nt, nf, nant, 2, 2), np.nan+0j, dtype=np.complex128)
        self.flags = np.ones((nt, nf, nant), dtype=bool)
        self.weights = np.zeros((nt, nf))
        self.q_snr = np.zeros((nt, nf))
        self.q_chi2 = np.zeros((nt, nf))
        self.q_rmse = np.zeros((nt, nf))
        self.q_cost_i = np.zeros((nt, nf))
        self.q_cost_f = np.zeros((nt, nf))
        self.q_nfev = np.zeros((nt, nf), dtype=np.int32)
        self.q_success = np.zeros((nt, nf), dtype=bool)
        self.native: Dict[str, Any] = {}

    def to_dict(self, time_c, freq_c, meta, step):
        return {
            "jones": self.jones, "time": time_c, "freq": freq_c,
            "freq_full": meta.freq, "flags": self.flags, "weights": self.weights,
            "errors": None, "params": self.native,
            "quality": {"snr": self.q_snr, "chi2_red": self.q_chi2,
                        "rmse": self.q_rmse, "cost_init": self.q_cost_i,
                        "cost_final": self.q_cost_f, "nfev": self.q_nfev,
                        "success": self.q_success},
            "metadata": {
                "jones_type": step.jones_type, "ref_ant": step.ref_ant,
                "feed_basis": meta.feed_basis,
                "solint_time": step.time_interval,
                "solint_freq": step.freq_interval,
                "phase_only": step.phase_only,
                "rfi_threshold": step.rfi_threshold,
                "max_iter": step.max_iter,
                "n_working": len(meta.working_antennas),
                "working_antennas": meta.working_antennas,
            },
        }


def _store_one(ti, fi, res, R: _SolArrays, meta):
    j = res["jones"]; native = res.get("native", {}); info = res.get("info", {})
    if j.ndim == 3:
        R.jones[ti, fi] = j
    elif j.ndim == 4:
        R.jones[ti, fi] = j[:, j.shape[1]//2]
    for a in meta.working_antennas:
        if np.all(np.isfinite(R.jones[ti, fi, a])):
            R.flags[ti, fi, a] = False
    q = compute_quality(info)
    R.q_snr[ti,fi] = q.snr; R.q_chi2[ti,fi] = q.chi2_red
    R.q_rmse[ti,fi] = q.rmse; R.q_cost_i[ti,fi] = q.cost_init
    R.q_cost_f[ti,fi] = q.cost_final; R.q_nfev[ti,fi] = q.nfev
    R.q_success[ti,fi] = q.success
    R.weights[ti,fi] = 1.0 if q.success else 0.5
    if native:
        nt, nf = R.jones.shape[:2]
        for k, v in native.items():
            if k not in R.native:
                R.native[k] = np.full((nt, nf) + v.shape, np.nan)
            try:
                R.native[k][ti, fi] = v
            except (ValueError, IndexError):
                pass


def _update_prev(R: _SolArrays, prev_list, jt, meta):
    for ti in range(R.jones.shape[0]):
        for fi in range(R.jones.shape[1]):
            if R.q_success[ti, fi]:
                rep = R.jones[ti, fi]
                native = {k: v[ti, fi] for k, v in R.native.items()
                          if np.any(np.isfinite(v[ti, fi]))}
                prev_list.append({"jones_type": jt, "jones": rep, "params": native})
                return
    # Fallback: any non-NaN
    for ti in range(R.jones.shape[0]):
        for fi in range(R.jones.shape[1]):
            if R.weights[ti, fi] > 0:
                prev_list.append({"jones_type": jt, "jones": R.jones[ti,fi], "params": {}})
                return


# ===================================================================
# SUMMARY
# ===================================================================

def _print_summary(jt, sid, nt, nf, ok, fail, empty, chi2, succ, meta, dt):
    total = nt * nf
    chi2_ok = chi2[succ.astype(bool)]
    if console and HAS_RICH:
        t = Table(title=f"{jt} / SPW {sid}", show_lines=False)
        t.add_column("", style="cyan", width=16)
        t.add_column("", style="white")
        t.add_row("Grid", f"{nt}×{nf} = {total}")
        t.add_row("Converged", f"[green]{ok}[/green]/{total}")
        if fail: t.add_row("Failed", f"[red]{fail}[/red]")
        if empty: t.add_row("Empty", f"{empty}")
        if len(chi2_ok): t.add_row("χ² med", f"{np.median(chi2_ok):.4f}")
        t.add_row("Antennas", f"{len(meta.working_antennas)}/{meta.n_ant}")
        t.add_row("Time", f"{dt:.1f}s")
        console.print(t)
    else:
        c = f", χ²={np.median(chi2_ok):.4f}" if len(chi2_ok) else ""
        _log(f"    {jt}/SPW{sid}: {ok}/{total}{c} ({dt:.1f}s)")
