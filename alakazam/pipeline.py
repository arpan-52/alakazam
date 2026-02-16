"""
ALAKAZAM Pipeline.

Memory-safe calibration with 3-tier data loading:

  TIER 1: Full SPW fits in RAM → single load, all cells solve in parallel
  TIER 2: Full SPW too big, but N solint-time-slots fit → batched loading,
           load N time-slots, solve those cells, free, load next N
  TIER 3: Even one solint-time-slot too big → pseudo-chunk that slot:
           load time-slices progressively, RFI + accumulate per-baseline
           running sums, delete each slice, pass tiny averaged result to solver

Loop order (user is boss):
    For each Jones type (user-specified order):
        For each SPW:
            estimate memory → pick tier → load → shared global → parallel solve

Workers NEVER touch disk. They slice shared numpy arrays via fork COW.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import multiprocessing as mp
import os
import logging
import time as _time
import gc

from .config import AlakazamConfig, SolveStep, load_config, config_to_yaml
from .core.ms_io import (
    detect_metadata, compute_solint_grid, parse_spw_selection, MSMetadata,
)
from .core.averaging import average_per_baseline_time_only, average_per_baseline_full
from .core.rfi import flag_rfi
from .core.quality import compute_quality
from .core.memory import get_available_ram_gb
from .solvers.registry import get_solver
from .jones import (
    jones_unapply, jones_unapply_freq,
    delay_to_jones, crossdelay_to_jones,
)
from .io.hdf5 import save_solutions

logger = logging.getLogger("alakazam")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    pass  # rich.progress available if needed
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


# ---------------------------------------------------------------------------
# Shared batch data — set in main process before Pool, read by workers
# via fork copy-on-write. Workers never copy or modify this.
# ---------------------------------------------------------------------------
_batch_data: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------

def _estimate_load_gb(n_rows: int, n_chan: int) -> float:
    """Estimate memory in GB for loading n_rows × n_chan from MS.

    Includes: vis_obs + vis_model (complex128) + flags (bool) + metadata.
    Plus ~1.5× overhead for temporaries during reshape/contiguous copy.
    """
    # vis_obs + vis_model: 2 × n_rows × n_chan × 4_corr × 16_bytes
    vis_bytes = 2 * n_rows * n_chan * 4 * 16
    # flags: n_rows × n_chan × 4 × 1
    flag_bytes = n_rows * n_chan * 4
    # ant1 + ant2 + time: n_rows × (4 + 4 + 8)
    meta_bytes = n_rows * 16
    return (vis_bytes + flag_bytes + meta_bytes) * 1.5 / GB


def _get_available_gb(config: AlakazamConfig) -> float:
    """Get usable RAM in GB with safety factor for fork overhead."""
    if config.memory_limit_gb > 0:
        return config.memory_limit_gb
    # 50% safety: data stays in memory during multiprocessing (fork doubles peak)
    return get_available_ram_gb() * 0.5


# ---------------------------------------------------------------------------
# MS reading helpers
# ---------------------------------------------------------------------------

def _build_taql_where(ms_path: str, spw_id: int, config: AlakazamConfig,
                      time_lo: float = None, time_hi: float = None) -> str:
    """Build TaQL WHERE clause for selection."""
    from casacore.tables import table
    conditions = [f"DATA_DESC_ID=={spw_id}"]

    if config.field is not None:
        field_tab = table(f"{ms_path}::FIELD", readonly=True, ack=False)
        fn = list(field_tab.getcol("NAME"))
        field_tab.close()
        if config.field in fn:
            conditions.append(f"FIELD_ID=={fn.index(config.field)}")

    if config.scans is not None:
        from .core.ms_io import _parse_scan_selection
        sids = _parse_scan_selection(config.scans)
        conditions.append(f"SCAN_NUMBER IN [{','.join(map(str, sids))}]")

    if time_lo is not None:
        conditions.append(f"TIME>={time_lo}")
    if time_hi is not None:
        conditions.append(f"TIME<{time_hi}")

    return " AND ".join(conditions)


def _read_and_reshape(
    ms_path: str, where: str, freq: np.ndarray, working_ants: np.ndarray,
    data_col: str, model_col: str,
) -> Optional[Dict[str, np.ndarray]]:
    """Read rows matching WHERE, filter working ants, reshape to (n,nchan,2,2)."""
    from casacore.tables import table, taql

    ms = table(ms_path, readonly=True, ack=False)
    sel = taql(f"SELECT * FROM $ms WHERE {where}")
    n_rows = sel.nrows()

    if n_rows == 0:
        sel.close(); ms.close()
        return None

    ant1 = sel.getcol("ANTENNA1").astype(np.int32)
    ant2 = sel.getcol("ANTENNA2").astype(np.int32)
    time_arr = sel.getcol("TIME")

    # Filter working antennas
    ws = set(int(a) for a in working_ants)
    valid = np.array([int(a1) in ws and int(a2) in ws
                      for a1, a2 in zip(ant1, ant2)])
    if not np.any(valid):
        sel.close(); ms.close()
        return None

    ant1 = ant1[valid]; ant2 = ant2[valid]; time_arr = time_arr[valid]

    if data_col not in sel.colnames():
        sel.close(); ms.close()
        raise ValueError(f"Column '{data_col}' not found in MS")

    data_raw = sel.getcol(data_col)[valid]
    if model_col in sel.colnames():
        model_raw = sel.getcol(model_col)[valid]
    else:
        model_raw = np.ones_like(data_raw)

    flags_raw = sel.getcol("FLAG")[valid]
    sel.close(); ms.close()

    # Reshape
    n_row, n_chan, n_corr = data_raw.shape
    if n_corr == 4:
        vis_obs = data_raw.reshape(n_row, n_chan, 2, 2)
        vis_model = model_raw.reshape(n_row, n_chan, 2, 2)
        flags = flags_raw.reshape(n_row, n_chan, 2, 2)
    elif n_corr == 2:
        vis_obs = np.zeros((n_row, n_chan, 2, 2), dtype=np.complex128)
        vis_model = np.zeros((n_row, n_chan, 2, 2), dtype=np.complex128)
        flags = np.ones((n_row, n_chan, 2, 2), dtype=bool)
        vis_obs[:, :, 0, 0] = data_raw[:, :, 0]
        vis_obs[:, :, 1, 1] = data_raw[:, :, 1]
        vis_model[:, :, 0, 0] = model_raw[:, :, 0]
        vis_model[:, :, 1, 1] = model_raw[:, :, 1]
        flags[:, :, 0, 0] = flags_raw[:, :, 0]
        flags[:, :, 1, 1] = flags_raw[:, :, 1]
    else:
        vis_obs = np.zeros((n_row, n_chan, 2, 2), dtype=np.complex128)
        vis_model = np.zeros((n_row, n_chan, 2, 2), dtype=np.complex128)
        flags = np.ones((n_row, n_chan, 2, 2), dtype=bool)
        vis_obs[:, :, 0, 0] = data_raw[:, :, 0]
        vis_model[:, :, 0, 0] = model_raw[:, :, 0]
        flags[:, :, 0, 0] = flags_raw[:, :, 0]

    del data_raw, model_raw, flags_raw

    return {
        "vis_obs": np.ascontiguousarray(vis_obs, dtype=np.complex128),
        "vis_model": np.ascontiguousarray(vis_model, dtype=np.complex128),
        "flags": np.ascontiguousarray(flags),
        "ant1": np.ascontiguousarray(ant1),
        "ant2": np.ascontiguousarray(ant2),
        "time": time_arr,
    }


def _count_rows(ms_path: str, where: str) -> int:
    """Lightweight row count without loading data."""
    from casacore.tables import table, taql
    ms = table(ms_path, readonly=True, ack=False)
    sel = taql(f"SELECT FROM $ms WHERE {where}")
    n = sel.nrows()
    sel.close(); ms.close()
    return n


def _pre_apply_jones(batch: Dict, prev_jones: List[Dict],
                     freq: np.ndarray, n_ant: int) -> Dict:
    """Pre-apply previously solved Jones to batch vis_obs in-place."""
    for prev in prev_jones:
        pj = _build_pre_jones(prev, freq, n_ant)
        if pj is None:
            continue
        if pj.ndim == 4:
            batch["vis_obs"] = jones_unapply_freq(
                pj, batch["vis_obs"], batch["ant1"], batch["ant2"])
        elif pj.ndim == 3:
            n_chan = batch["vis_obs"].shape[1]
            for f in range(n_chan):
                batch["vis_obs"][:, f] = jones_unapply(
                    pj, batch["vis_obs"][:, f], batch["ant1"], batch["ant2"])
    return batch


def _build_pre_jones(prev: Dict, freq: np.ndarray, n_ant: int):
    """Build Jones for pre-apply from a previously solved step."""
    jt = prev.get("jones_type", "")
    params = prev.get("params", {})
    if jt == "K" and "delay" in params:
        return delay_to_jones(params["delay"], freq)
    elif jt == "KCROSS" and "cross_delay" in params:
        return crossdelay_to_jones(params["cross_delay"], freq)
    return prev.get("jones")


# ---------------------------------------------------------------------------
# TIER 3: Pseudo-chunking — for when even one solint slot is too big
# ---------------------------------------------------------------------------

def _pseudo_chunk_solve(
    ms_path: str, spw_id: int, config: AlakazamConfig, meta: MSMetadata,
    time_lo: float, time_hi: float, freq_indices: np.ndarray,
    prev_jones: List[Dict],
    step: SolveStep, available_gb: float,
) -> Optional[Dict]:
    """Load a single solint cell via pseudo-chunking.

    Loads time-slices progressively, applies RFI + accumulates per-baseline
    running sums, deletes each slice. Final result is tiny:
    (n_bl, n_chan_chunk, 2, 2) — always fits in RAM.

    Then solves directly on the accumulated average.
    """
    from casacore.tables import table, taql

    chunk_freq = meta.freq[freq_indices]
    n_chan_chunk = len(freq_indices)
    n_ant = meta.n_ant

    # Get unique times in this cell
    where_base = _build_taql_where(ms_path, spw_id, config, time_lo, time_hi)
    ms = table(ms_path, readonly=True, ack=False)
    sel = taql(f"SELECT TIME FROM $ms WHERE {where_base}")
    if sel.nrows() == 0:
        sel.close(); ms.close()
        return None
    all_times = np.unique(sel.getcol("TIME"))
    sel.close(); ms.close()

    # How many timestamps fit in one pseudo-chunk?
    n_bl = n_ant * (n_ant - 1) // 2
    # Estimate rows per timestamp
    rows_per_ts = max(1, n_bl)  # conservative
    mem_per_ts = _estimate_load_gb(rows_per_ts, meta.n_freq)
    # Use 70% safety for pseudo-chunks (sequential, delete before next)
    ts_per_pchunk = max(1, int((available_gb * 0.7) / max(mem_per_ts, 1e-6)))
    n_pchunks = max(1, int(np.ceil(len(all_times) / ts_per_pchunk)))

    _log(f"      Pseudo-chunking: {len(all_times)} timestamps in {n_pchunks} pieces")

    # Accumulators: per-baseline running sums (n_bl, n_chan_chunk, 2, 2)
    n_ant = meta.n_ant
    bl_sum_obs = {}
    bl_sum_model = {}
    bl_counts = {}

    for pc_idx in range(n_pchunks):
        ts_start = pc_idx * ts_per_pchunk
        ts_end = min(ts_start + ts_per_pchunk, len(all_times))
        t_lo_pc = all_times[ts_start]
        t_hi_pc = all_times[ts_end - 1] + 0.01  # tiny epsilon

        where_pc = _build_taql_where(ms_path, spw_id, config, t_lo_pc, t_hi_pc)
        batch_pc = _read_and_reshape(
            ms_path, where_pc, meta.freq, meta.working_antennas,
            config.data_col, config.model_col)

        if batch_pc is None:
            continue

        # Slice freq
        vis_obs_pc = batch_pc["vis_obs"][:, freq_indices]
        vis_model_pc = batch_pc["vis_model"][:, freq_indices]
        flags_pc = batch_pc["flags"][:, freq_indices]
        ant1_pc = batch_pc["ant1"]
        ant2_pc = batch_pc["ant2"]

        # Pre-apply
        if prev_jones:
            for prev in prev_jones:
                pj = _build_pre_jones(prev, chunk_freq, n_ant)
                if pj is not None:
                    if pj.ndim == 4:
                        pj_chunk = pj[:, freq_indices]
                        vis_obs_pc = jones_unapply_freq(
                            pj_chunk, vis_obs_pc, ant1_pc, ant2_pc)
                    elif pj.ndim == 3:
                        for f in range(n_chan_chunk):
                            vis_obs_pc[:, f] = jones_unapply(
                                pj, vis_obs_pc[:, f], ant1_pc, ant2_pc)

        # RFI
        if step.rfi_threshold > 0:
            flags_pc, _ = flag_rfi(vis_obs_pc, flags_pc, step.rfi_threshold)

        # Accumulate per baseline
        for row in range(len(ant1_pc)):
            a1 = int(min(ant1_pc[row], ant2_pc[row]))
            a2 = int(max(ant1_pc[row], ant2_pc[row]))
            if a1 == a2:
                continue
            key = (a1, a2)
            if key not in bl_sum_obs:
                bl_sum_obs[key] = np.zeros((n_chan_chunk, 2, 2), dtype=np.complex128)
                bl_sum_model[key] = np.zeros((n_chan_chunk, 2, 2), dtype=np.complex128)
                bl_counts[key] = np.zeros((n_chan_chunk, 2, 2), dtype=np.float64)
            for c in range(n_chan_chunk):
                for i in range(2):
                    for j in range(2):
                        if not flags_pc[row, c, i, j]:
                            bl_sum_obs[key][c, i, j] += vis_obs_pc[row, c, i, j]
                            bl_sum_model[key][c, i, j] += vis_model_pc[row, c, i, j]
                            bl_counts[key][c, i, j] += 1.0

        # Free piece immediately
        del batch_pc, vis_obs_pc, vis_model_pc, flags_pc
        gc.collect()

    if not bl_sum_obs:
        return None

    # Finalize averages
    bls = sorted(bl_sum_obs.keys())
    n_bl_out = len(bls)
    bl_a1 = np.array([b[0] for b in bls], dtype=np.int32)
    bl_a2 = np.array([b[1] for b in bls], dtype=np.int32)

    solver = get_solver(step.jones_type)

    if solver.can_avg_freq:
        # Average over freq too → (n_bl, 2, 2)
        vis_avg = np.zeros((n_bl_out, 2, 2), dtype=np.complex128)
        model_avg = np.zeros((n_bl_out, 2, 2), dtype=np.complex128)
        for bi, key in enumerate(bls):
            for i in range(2):
                for j in range(2):
                    total_c = np.sum(bl_counts[key][:, i, j])
                    if total_c > 0:
                        vis_avg[bi, i, j] = np.sum(bl_sum_obs[key][:, i, j]) / total_c
                        model_avg[bi, i, j] = np.sum(bl_sum_model[key][:, i, j]) / total_c
        solve_freq = None
    else:
        # Keep freq axis → (n_bl, n_chan, 2, 2)
        vis_avg = np.zeros((n_bl_out, n_chan_chunk, 2, 2), dtype=np.complex128)
        model_avg = np.zeros((n_bl_out, n_chan_chunk, 2, 2), dtype=np.complex128)
        for bi, key in enumerate(bls):
            for c in range(n_chan_chunk):
                for i in range(2):
                    for j in range(2):
                        if bl_counts[key][c, i, j] > 0:
                            vis_avg[bi, c, i, j] = bl_sum_obs[key][c, i, j] / bl_counts[key][c, i, j]
                            model_avg[bi, c, i, j] = bl_sum_model[key][c, i, j] / bl_counts[key][c, i, j]
        solve_freq = chunk_freq

    # Solve
    jones_sol, native, info = solver.solve(
        vis_avg, model_avg, bl_a1, bl_a2,
        n_ant, step.ref_ant, meta.working_antennas,
        freq=solve_freq, phase_only=step.phase_only,
        max_iter=step.max_iter, tol=step.tol,
    )
    return {"jones": jones_sol, "native": native, "info": info}


# ---------------------------------------------------------------------------
# Worker function — reads from shared _batch_data, never touches disk
# ---------------------------------------------------------------------------

def _solve_cell_worker(args) -> Tuple:
    """Solve one solint cell from shared batch data.

    Receives lightweight args (indices + params only).
    Reads from module-global _batch_data via fork COW — no copy, no disk.
    """
    (t_idx, f_idx, time_lo, time_hi, freq_chunk_indices,
     jones_type, n_ant, ref_ant, working_ants,
     phase_only, rfi_threshold, max_iter, tol) = args

    vis_obs_batch = _batch_data["vis_obs"]
    vis_model_batch = _batch_data["vis_model"]
    flags_batch = _batch_data["flags"]
    ant1_batch = _batch_data["ant1"]
    ant2_batch = _batch_data["ant2"]
    time_batch = _batch_data["time"]
    freq_full = _batch_data["freq"]

    # Slice this cell's data
    time_mask = (time_batch >= time_lo) & (time_batch < time_hi)
    rows = np.where(time_mask)[0]
    if len(rows) == 0:
        return (t_idx, f_idx, None, None, None)

    vis_obs = vis_obs_batch[rows][:, freq_chunk_indices]
    vis_model = vis_model_batch[rows][:, freq_chunk_indices]
    flags = flags_batch[rows][:, freq_chunk_indices]
    ant1 = ant1_batch[rows]
    ant2 = ant2_batch[rows]
    chunk_freq = freq_full[freq_chunk_indices]

    # RFI
    if rfi_threshold > 0:
        flags, _ = flag_rfi(vis_obs, flags, rfi_threshold)
    if np.all(flags):
        return (t_idx, f_idx, None, None, None)

    solver = get_solver(jones_type)

    # Average
    if solver.can_avg_freq:
        vis_avg, _, bl_a1, bl_a2 = average_per_baseline_full(
            vis_obs, flags, ant1, ant2, n_ant)
        model_avg, _, _, _ = average_per_baseline_full(
            vis_model, flags, ant1, ant2, n_ant)
        solve_freq = None
    else:
        vis_avg, _, bl_a1, bl_a2 = average_per_baseline_time_only(
            vis_obs, flags, ant1, ant2, n_ant)
        model_avg, _, _, _ = average_per_baseline_time_only(
            vis_model, flags, ant1, ant2, n_ant)
        solve_freq = chunk_freq

    if len(bl_a1) == 0:
        return (t_idx, f_idx, None, None, None)

    try:
        jones_sol, native, info = solver.solve(
            vis_avg, model_avg, bl_a1, bl_a2,
            n_ant, ref_ant, working_ants,
            freq=solve_freq, phase_only=phase_only,
            max_iter=max_iter, tol=tol,
        )
    except Exception as e:
        return (t_idx, f_idx, None, None, {"error": str(e), "success": False})

    return (t_idx, f_idx, jones_sol, native, info)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(config: AlakazamConfig) -> Dict[str, Dict[int, Dict]]:
    """Run the full calibration pipeline."""
    t_wall = _time.time()
    global _batch_data

    if console and HAS_RICH:
        console.print(Panel(
            "[bold cyan]ALAKAZAM[/bold cyan] — A Radio Interferometric Calibration Suite\n"
            "Arpan Pal 2026, NRAO / NCRA",
            style="bold blue",
        ))
    _log(f"MS:     {config.ms_path}")
    _log(f"Output: {config.output}")
    _log(f"Chain:  {' → '.join(s.jones_type for s in config.steps)}")

    # SPWs
    from casacore.tables import table
    spw_tab = table(f"{config.ms_path}::SPECTRAL_WINDOW", readonly=True, ack=False)
    n_spw_total = spw_tab.nrows()
    spw_tab.close()
    selected_spws = parse_spw_selection(config.spw, n_spw_total)
    _log(f"SPWs:   {selected_spws}")

    # Metadata per SPW
    meta_per_spw: Dict[int, MSMetadata] = {}
    for sid in selected_spws:
        meta_per_spw[sid] = detect_metadata(
            config.ms_path, field=config.field, spw_id=sid, scans=config.scans)

    first_meta = meta_per_spw[selected_spws[0]]
    observation_meta = {
        "ms_path": config.ms_path, "n_ant": first_meta.n_ant,
        "antenna_names": first_meta.antenna_names,
        "working_antennas": first_meta.working_antennas,
        "non_working_antennas": first_meta.non_working_antennas,
        "feed_basis": first_meta.feed_basis,
        "field": config.field or "",
        "parang_applied": config.apply_parang,
    }

    n_workers = min(os.cpu_count() or 1, 8)
    available_gb = _get_available_gb(config)
    _log(f"Workers: {n_workers}, RAM budget: {available_gb:.1f} GB")

    all_solutions: Dict[str, Dict[int, Dict]] = {}
    prev_solved: Dict[int, List[Dict]] = {s: [] for s in selected_spws}

    # === OUTER: Jones type (user order) ===
    for step_idx, step in enumerate(config.steps):
        jt = step.jones_type
        _log(f"\n{'='*60}", style="bold green")
        _log(f"Step {step_idx+1}/{len(config.steps)}: {jt} "
             f"(time={step.time_interval}, freq={step.freq_interval}"
             f"{', phase_only' if step.phase_only else ''})", style="bold green")

        if jt not in all_solutions:
            all_solutions[jt] = {}

        # === MIDDLE: SPW ===
        for spw_id in selected_spws:
            meta = meta_per_spw[spw_id]
            n_ant = meta.n_ant
            t_spw = _time.time()

            if step.ref_ant not in meta.working_antennas:
                raise ValueError(
                    f"ref_ant={step.ref_ant} not working in SPW {spw_id}. "
                    f"Working: {sorted(meta.working_antennas[:10])}")

            time_edges, freq_chunks = compute_solint_grid(
                meta.unique_times, meta.freq,
                step.time_interval, step.freq_interval)
            n_sol_time = len(time_edges) - 1
            n_sol_freq = len(freq_chunks)
            total_cells = n_sol_time * n_sol_freq

            _log(f"\n  SPW {spw_id}: {n_sol_time}t × {n_sol_freq}f = {total_cells} cells, "
                 f"{len(meta.working_antennas)}/{n_ant} ants")

            # --- MEMORY ESTIMATION ---
            # Estimate rows for full SPW
            where_full = _build_taql_where(config.ms_path, spw_id, config)
            n_rows_full = _count_rows(config.ms_path, where_full)
            mem_full = _estimate_load_gb(n_rows_full, meta.n_freq)

            # Estimate rows per solint time slot (average)
            rows_per_slot = max(1, n_rows_full // max(1, n_sol_time))
            mem_per_slot = _estimate_load_gb(rows_per_slot, meta.n_freq)

            _log(f"    Memory: full SPW={mem_full:.2f}GB, per slot={mem_per_slot:.2f}GB, "
                 f"budget={available_gb:.1f}GB")

            # Allocate solution arrays
            jones_arr = np.full((n_sol_time, n_sol_freq, n_ant, 2, 2),
                                np.nan + 0j, dtype=np.complex128)
            flags_arr = np.ones((n_sol_time, n_sol_freq, n_ant), dtype=bool)
            weights_arr = np.zeros((n_sol_time, n_sol_freq))
            q_arrays = {k: np.zeros((n_sol_time, n_sol_freq))
                        for k in ("snr", "chi2_red", "rmse", "cost_init", "cost_final")}
            q_arrays["nfev"] = np.zeros((n_sol_time, n_sol_freq), dtype=np.int32)
            q_arrays["success"] = np.zeros((n_sol_time, n_sol_freq), dtype=bool)
            native_all: Dict[str, Any] = {}

            n_solved = 0; n_failed = 0; n_empty = 0
            first_good_jones = None; first_good_native = {}

            # === PICK TIER ===
            if mem_full <= available_gb:
                # --- TIER 1: Full SPW load ---
                _log(f"    TIER 1: Loading full SPW in one read")
                batch = _read_and_reshape(
                    config.ms_path, where_full, meta.freq, meta.working_antennas,
                    config.data_col, config.model_col)

                if batch is not None:
                    if prev_solved[spw_id]:
                        batch = _pre_apply_jones(batch, prev_solved[spw_id], meta.freq, n_ant)
                    batch["freq"] = meta.freq
                    results = _dispatch_batch(
                        batch, time_edges, freq_chunks,
                        0, n_sol_time, step, meta, n_workers)
                    _collect_results(
                        results, jones_arr, flags_arr, weights_arr,
                        q_arrays, native_all, meta)
                    for r in results:
                        _tally(r, locals())
                    del batch; _batch_data.clear(); gc.collect()

            elif mem_per_slot <= available_gb:
                # --- TIER 2: Batched loading (N time-slots at a time) ---
                slots_per_batch = max(1, int(available_gb / mem_per_slot))
                n_batches = max(1, int(np.ceil(n_sol_time / slots_per_batch)))
                _log(f"    TIER 2: {n_batches} batches of {slots_per_batch} time-slots")

                for b_idx in range(n_batches):
                    t_start_slot = b_idx * slots_per_batch
                    t_end_slot = min(t_start_slot + slots_per_batch, n_sol_time)
                    t_lo_batch = time_edges[t_start_slot]
                    t_hi_batch = time_edges[t_end_slot]

                    where_b = _build_taql_where(
                        config.ms_path, spw_id, config, t_lo_batch, t_hi_batch)
                    batch = _read_and_reshape(
                        config.ms_path, where_b, meta.freq, meta.working_antennas,
                        config.data_col, config.model_col)

                    if batch is None:
                        continue
                    if prev_solved[spw_id]:
                        batch = _pre_apply_jones(batch, prev_solved[spw_id], meta.freq, n_ant)
                    batch["freq"] = meta.freq

                    results = _dispatch_batch(
                        batch, time_edges, freq_chunks,
                        t_start_slot, t_end_slot, step, meta, n_workers)
                    _collect_results(
                        results, jones_arr, flags_arr, weights_arr,
                        q_arrays, native_all, meta)
                    for r in results:
                        _tally(r, locals())

                    del batch; _batch_data.clear(); gc.collect()
                    _log(f"      Batch {b_idx+1}/{n_batches} done")

            else:
                # --- TIER 3: Pseudo-chunking (even one slot too big) ---
                _log(f"    TIER 3: Pseudo-chunking (slot {mem_per_slot:.1f}GB > budget {available_gb:.1f}GB)")

                for t_idx in range(n_sol_time):
                    for f_idx in range(n_sol_freq):
                        res = _pseudo_chunk_solve(
                            config.ms_path, spw_id, config, meta,
                            time_edges[t_idx], time_edges[t_idx + 1],
                            freq_chunks[f_idx], prev_solved[spw_id],
                            step, available_gb)
                        if res is None:
                            n_empty += 1; continue
                        _store_one(
                            t_idx, f_idx, res, jones_arr, flags_arr, weights_arr,
                            q_arrays, native_all, meta)
                        q = compute_quality(res["info"])
                        if q.success:
                            n_solved += 1
                            if first_good_jones is None:
                                first_good_jones = res["jones"]
                                first_good_native = res["native"]
                        else:
                            n_failed += 1

            # Count results from tier 1/2
            n_solved = int(np.sum(q_arrays["success"]))
            n_empty = total_cells - int(np.sum(weights_arr > 0))
            n_failed = total_cells - n_solved - n_empty

            # Find representative for chaining
            if first_good_jones is None:
                for ti in range(n_sol_time):
                    for fi in range(n_sol_freq):
                        if q_arrays["success"][ti, fi]:
                            first_good_jones = jones_arr[ti, fi]
                            break
                    if first_good_jones is not None:
                        break

            # Freq/time centers
            freq_centers = np.array([np.mean(meta.freq[fi]) for fi in freq_chunks])
            time_centers = 0.5 * (time_edges[:-1] + time_edges[1:])

            sol_data = {
                "jones": jones_arr, "time": time_centers, "freq": freq_centers,
                "freq_full": meta.freq, "flags": flags_arr, "weights": weights_arr,
                "errors": None, "params": native_all, "quality": q_arrays,
                "metadata": {
                    "jones_type": jt, "ref_ant": step.ref_ant,
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
            all_solutions[jt][spw_id] = sol_data

            # Store for chaining
            if first_good_jones is not None:
                rep_j = first_good_jones
                if hasattr(rep_j, 'ndim') and rep_j.ndim == 4:
                    rep_j = rep_j[:, rep_j.shape[1] // 2]
                prev_solved[spw_id].append({
                    "jones_type": jt, "jones": rep_j, "params": first_good_native})

            _print_summary(jt, spw_id, n_sol_time, n_sol_freq,
                           n_solved, n_failed, n_empty,
                           q_arrays["chi2_red"], q_arrays["success"],
                           meta, _time.time() - t_spw)

    save_solutions(config.output, all_solutions, observation_meta,
                   config_to_yaml(config))
    _log(f"\n{'='*60}", style="bold green")
    _log(f"Pipeline complete in {_time.time() - t_wall:.1f}s", style="bold green")
    _log(f"Solutions: {config.output}", style="bold green")
    return all_solutions


# ---------------------------------------------------------------------------
# Dispatch batch to workers (shared memory pattern)
# ---------------------------------------------------------------------------

def _dispatch_batch(
    batch: Dict, time_edges: np.ndarray, freq_chunks: List,
    t_start: int, t_end: int, step: SolveStep,
    meta: MSMetadata, n_workers: int,
) -> List[Tuple]:
    """Set shared global, build tasks, dispatch to Pool, return results."""
    global _batch_data
    _batch_data = batch

    tasks = []
    for t_idx in range(t_start, t_end):
        for f_idx in range(len(freq_chunks)):
            tasks.append((
                t_idx, f_idx,
                time_edges[t_idx], time_edges[t_idx + 1],
                freq_chunks[f_idx],
                step.jones_type, meta.n_ant, step.ref_ant,
                meta.working_antennas,
                step.phase_only, step.rfi_threshold,
                step.max_iter, step.tol,
            ))

    if n_workers > 1 and len(tasks) > 1:
        with mp.Pool(processes=min(n_workers, len(tasks))) as pool:
            results = pool.map(_solve_cell_worker, tasks)
    else:
        results = [_solve_cell_worker(t) for t in tasks]

    return results


def _collect_results(results, jones_arr, flags_arr, weights_arr,
                     q_arrays, native_all, meta):
    """Collect worker results into solution arrays."""
    for (t_idx, f_idx, j_sol, native, info) in results:
        if j_sol is None:
            continue
        _store_one(t_idx, f_idx, {"jones": j_sol, "native": native, "info": info},
                   jones_arr, flags_arr, weights_arr, q_arrays, native_all, meta)


def _store_one(t_idx, f_idx, res, jones_arr, flags_arr, weights_arr,
               q_arrays, native_all, meta):
    """Store one cell's result into arrays."""
    j_sol = res["jones"]; native = res["native"]; info = res["info"]

    if j_sol.ndim == 3:
        jones_arr[t_idx, f_idx] = j_sol
    elif j_sol.ndim == 4:
        jones_arr[t_idx, f_idx] = j_sol[:, j_sol.shape[1] // 2]

    for a in meta.working_antennas:
        if np.all(np.isfinite(jones_arr[t_idx, f_idx, a])):
            flags_arr[t_idx, f_idx, a] = False

    q = compute_quality(info)
    q_arrays["snr"][t_idx, f_idx] = q.snr
    q_arrays["chi2_red"][t_idx, f_idx] = q.chi2_red
    q_arrays["rmse"][t_idx, f_idx] = q.rmse
    q_arrays["cost_init"][t_idx, f_idx] = q.cost_init
    q_arrays["cost_final"][t_idx, f_idx] = q.cost_final
    q_arrays["nfev"][t_idx, f_idx] = q.nfev
    q_arrays["success"][t_idx, f_idx] = q.success
    weights_arr[t_idx, f_idx] = 1.0 if q.success else 0.5

    if native:
        n_sol_time, n_sol_freq = jones_arr.shape[:2]
        for key, val in native.items():
            if key not in native_all:
                native_all[key] = np.full((n_sol_time, n_sol_freq) + val.shape, np.nan)
            try:
                native_all[key][t_idx, f_idx] = val
            except (ValueError, IndexError):
                pass


def _tally(result, ns):
    """Count solved/failed/empty (called in loop)."""
    pass  # Counting done via q_arrays["success"] after all batches


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(jt, spw_id, nt, nf, ok, fail, empty, chi2, succ, meta, dt):
    total = nt * nf
    chi2_ok = chi2[succ.astype(bool)]
    if console and HAS_RICH:
        tbl = Table(title=f"{jt} / SPW {spw_id}", show_lines=False)
        tbl.add_column("", style="cyan", width=18)
        tbl.add_column("", style="white")
        tbl.add_row("Grid", f"{nt} × {nf} = {total}")
        tbl.add_row("Converged", f"[green]{ok}[/green]/{total}")
        if fail: tbl.add_row("Failed", f"[red]{fail}[/red]")
        if empty: tbl.add_row("Empty", f"{empty}")
        if len(chi2_ok):
            tbl.add_row("χ² median", f"{np.median(chi2_ok):.4f}")
        tbl.add_row("Antennas", f"{len(meta.working_antennas)}/{meta.n_ant}")
        tbl.add_row("Wall time", f"{dt:.1f}s")
        console.print(tbl)
    else:
        c = f", χ²={np.median(chi2_ok):.4f}" if len(chi2_ok) else ""
        _log(f"    {jt}/SPW{spw_id}: {ok}/{total}{c} ({dt:.1f}s)")


def run_from_yaml(config_path: str) -> Dict:
    """Load YAML config and run pipeline."""
    return run_pipeline(load_config(config_path))
