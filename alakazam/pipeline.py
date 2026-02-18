"""ALAKAZAM Pipeline.

Orchestrates: solve → fluxscale → apply.

Solve chain logic:
  Step 1 solves on raw data.
  Step 2 pre-applies step 1, then solves.
  Step N pre-applies all steps 1..N-1, then solves.

  If external_preapply is supplied, those solutions are applied to the data
  before step 1 and remain in the preapply stack throughout.

  Each step can target multiple fields independently:
    field: [[3C147], [3C147, 3C286], [PKS1934, J0538]]
  For each field in a step, the solve runs independently and stores
  per-field solutions in the output H5.

3-tier memory management:
  T1: Full SPW fits in RAM      → single load, all slots parallel
  T2: N slots fit               → batched load
  T3: One slot too big          → pseudo-chunk (progressive)

Workers never touch disk. Data shared via fork COW.

Provenance is recorded at every step: what was pre-applied, what was solved,
which fields and scans, solint, freqint, reference antenna.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations

import gc
import json
import logging
import multiprocessing as mp
import os
import time as _time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import AlakazamConfig, SolveBlock, ApplyBlock, FluxscaleBlock
from .core.ms_io import detect_metadata, read_data, compute_solint_grid
from .core.averaging import average_per_baseline_full, average_per_baseline_time_only
from .core.rfi import flag_rfi
from .core.quality import compute_quality
from .core.memory import tier_strategy
from .core.interpolation import interpolate_jones_multifield
from .solvers import get_solver
from .io.hdf5 import save_solutions, load_all_fields, list_spws
from .calibration.fluxscale import run_fluxscale
from .calibration.apply import apply_calibration
from .jones.algebra import compose_jones_chain, jones_unapply, jones_unapply_freq

logger = logging.getLogger("alakazam")

try:
    from rich.console import Console
    from rich.panel import Panel
    _console = Console()
    HAS_RICH = True
except ImportError:
    _console = None
    HAS_RICH = False

# Fork COW shared data
_batch_data: Dict[str, Any] = {}


def _log(msg: str, style: str = ""):
    logger.info(msg)
    if _console and style:
        _console.print(msg, style=style)
    elif _console:
        _console.print(msg)


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def run_pipeline(config: AlakazamConfig) -> None:
    """Execute all solve, fluxscale, and apply blocks."""
    t0 = _time.time()

    if HAS_RICH:
        _console.print(Panel(
            "[bold cyan]ALAKAZAM[/bold cyan] v3.0 — Radio Interferometric Calibration\n"
            "Arpan Pal 2026, NRAO / NCRA", style="bold blue"
        ))

    for i, sb in enumerate(config.solve_blocks):
        _log(f"\n{'='*60}", "bold green")
        _log(f"SOLVE BLOCK {i+1}/{len(config.solve_blocks)}: {sb.ms}", "bold green")
        _run_solve_block(sb)

    for i, fb in enumerate(config.fluxscale_blocks):
        _log(f"\n{'='*60}", "bold magenta")
        _log(f"FLUXSCALE BLOCK {i+1}/{len(config.fluxscale_blocks)}", "bold magenta")
        run_fluxscale(fb)

    for i, ab in enumerate(config.apply_blocks):
        _log(f"\n{'='*60}", "bold yellow")
        _log(f"APPLY BLOCK {i+1}/{len(config.apply_blocks)}: {ab.ms}", "bold yellow")
        apply_calibration(ab)

    _log(f"\nDone in {_time.time()-t0:.1f}s", "bold green")


# ============================================================
# SOLVE BLOCK
# ============================================================

def _run_solve_block(sb: SolveBlock) -> None:
    """Run all Jones steps in one solve block."""
    global _batch_data

    meta = detect_metadata(sb.ms)
    spws = sb.spw if sb.spw is not None else list(range(meta.n_spw))

    n_steps = len(sb.jones)
    _log(f"  jones chain: {sb.jones}  spws={spws}  ref_ant={sb.ref_ant}")

    for spw in spws:
        freqs = meta.spw_freqs[spw]
        _log(f"  SPW {spw}  ({len(freqs)} channels)")

        # Internal preapply stack: accumulates solved solutions as steps complete
        # Key: jones_type → {field_name → sol_dict}
        internal_stack: Dict[str, Dict[str, Any]] = {}

        for step_idx, jones_type in enumerate(sb.jones):
            field_group = sb.field_groups[step_idx]
            scans_for_step = sb.scans_per_step[step_idx]

            _log(f"    step {step_idx+1}: jones={jones_type}  fields={field_group}")

            solver = get_solver(
                jones_type,
                ref_ant=sb.ref_ant,
                max_iter=sb.max_iter,
                tol=sb.tol,
                phase_only=sb.phase_only[step_idx],
            )

            # Solve independently per field in this step's group
            for field_idx, field_name in enumerate(field_group):
                field_scans = scans_for_step[field_idx] if field_idx < len(scans_for_step) else None

                _log(f"      field={field_name}  scans={field_scans or 'all'}")

                _solve_one_field(
                    sb=sb,
                    step_idx=step_idx,
                    jones_type=jones_type,
                    field_name=field_name,
                    field_scans=field_scans,
                    spw=spw,
                    freqs=freqs,
                    meta=meta,
                    solver=solver,
                    internal_stack=internal_stack,
                )

            # After solving all fields in this step, add to internal stack
            # so next steps can preapply
            _update_internal_stack(internal_stack, sb.output, jones_type, spw)

        gc.collect()


def _solve_one_field(
    sb: SolveBlock,
    step_idx: int,
    jones_type: str,
    field_name: str,
    field_scans: Optional[List[int]],
    spw: int,
    freqs: np.ndarray,
    meta,
    solver,
    internal_stack: Dict,
) -> None:
    """Solve one Jones type for one field."""

    # Read data
    d = read_data(
        sb.ms, spw,
        fields=[field_name],
        scans=field_scans,
        data_col=sb.data_col,
        model_col=sb.model_col,
    )
    if not d:
        logger.warning(f"      no data for field={field_name} spw={spw}")
        return

    vis_obs   = d["vis_obs"].copy()    # (n_row, n_chan, 2, 2)
    vis_model = d["vis_model"]
    flags     = d["flags"]
    ant1      = d["ant1"]
    ant2      = d["ant2"]
    row_times = d["times"]

    # RFI flagging
    flags = flag_rfi(vis_obs, flags, sb.rfi_threshold)

    # Zero flagged data
    vis_obs[flags]   = 0.0
    vis_model[flags] = 0.0

    # Build full preapply stack for this step
    preapply_jones_list = _build_preapply_stack(
        sb=sb,
        step_idx=step_idx,
        field_name=field_name,
        spw=spw,
        freqs=freqs,
        row_times=row_times,
        meta=meta,
        internal_stack=internal_stack,
    )

    # Apply preapply stack to data
    if preapply_jones_list:
        vis_obs = _apply_preapply(vis_obs, preapply_jones_list, ant1, ant2, row_times)

    # Compute solint grid
    unique_times = np.unique(row_times)
    time_int = sb.time_interval[step_idx]
    solint_blocks = compute_solint_grid(unique_times, time_int)

    freq_int = sb.freq_interval[step_idx]
    # Frequency averaging: determine n_chan_avg
    if freq_int is None:
        # Full SPW: one freq solution slot
        freq_blocks = [freqs]
        freq_indices = [np.arange(len(freqs))]
    else:
        freq_blocks, freq_indices = _compute_freqint_grid(freqs, freq_int)

    # Tier strategy
    tier, batch = tier_strategy(
        n_slots=len(solint_blocks) * len(freq_blocks),
        n_baseline=meta.n_baseline,
        n_chan=len(freqs),
        n_ant=meta.n_ant,
        limit_gb=sb.memory_limit_gb,
    )

    _log(f"        solint_blocks={len(solint_blocks)}  freqint_blocks={len(freq_blocks)}"
         f"  tier={tier}  batch={batch}")

    # Collect solutions across all time/freq slots
    all_jones     = []  # per time slot: (n_ant, [n_freq,] 2, 2) or list of freq-resolved
    all_times     = []  # centre time per slot
    all_freqs_sol = []  # centre freq per freq block
    all_native    = []  # per time slot native params

    field_ra  = meta.field_ra[meta.field_names.index(field_name)] if field_name in meta.field_names else 0.0
    field_dec = meta.field_dec[meta.field_names.index(field_name)] if field_name in meta.field_names else 0.0

    for t_block in solint_blocks:
        t_mask = np.isin(row_times, t_block)
        if not t_mask.any():
            continue

        vis_t   = vis_obs[t_mask]
        model_t = vis_model[t_mask]
        a1_t    = ant1[t_mask]
        a2_t    = ant2[t_mask]
        fl_t    = flags[t_mask]

        # Average in time (keep freq for now)
        avg_vis, avg_flags, out_a1, out_a2 = average_per_baseline_time_only(
            vis_t, fl_t, a1_t, a2_t, meta.n_ant
        )
        avg_model, _, _, _ = average_per_baseline_time_only(
            model_t, fl_t, a1_t, a2_t, meta.n_ant
        )

        # Solve per freq block
        slot_jones_freq = []
        slot_native_freq = []

        for f_block_freqs, f_idx in zip(freq_blocks, freq_indices):
            vis_tf   = avg_vis[:, f_idx]   # (n_bl, n_f, 2, 2)
            model_tf = avg_model[:, f_idx]

            result = solver.solve(
                vis_obs=vis_tf,
                vis_model=model_tf,
                ant1=out_a1,
                ant2=out_a2,
                freqs=f_block_freqs,
                n_ant=meta.n_ant,
            )
            slot_jones_freq.append(result["jones"])
            slot_native_freq.append(result.get("native_params"))

            if not result["converged"]:
                logger.debug(
                    f"        {jones_type} field={field_name} spw={spw} "
                    f"t={t_block[0]:.0f} f={f_block_freqs[0]/1e6:.1f}MHz "
                    f"did not converge in {result['n_iter']} iters"
                )

        # Stack freq blocks: shape (n_ant, n_freq_total, 2, 2) or (n_ant, 2, 2)
        if len(slot_jones_freq) == 1:
            jones_slot = slot_jones_freq[0]
        else:
            # Concatenate along freq axis
            if slot_jones_freq[0].ndim == 4:
                jones_slot = np.concatenate(slot_jones_freq, axis=1)
            else:
                jones_slot = slot_jones_freq[0]  # freq-indep, just use first

        all_jones.append(jones_slot)
        all_times.append(float(np.mean(t_block)))
        all_native.append(slot_native_freq[0])  # use first freq block's native params

    if not all_jones:
        logger.warning(f"      no solutions produced for {jones_type} field={field_name}")
        return

    # Stack time axis: (n_sol_t, n_ant, [n_freq,] 2, 2)
    jones_all = np.stack(all_jones, axis=0)
    times_all = np.array(all_times)

    # Solution freq grid
    if freq_blocks[0] is not None and len(freq_blocks) > 1:
        sol_freqs = np.array([float(np.mean(fb)) for fb in freq_blocks])
    else:
        sol_freqs = freqs if jones_all.ndim == 5 else None

    # Build native params for all time slots
    native_params_save = None
    if all_native and all_native[0] is not None:
        np0 = all_native[0]
        ntype = np0.get("type", jones_type)
        if ntype == "K" and "delay" in np0:
            delays = np.stack([n["delay"] for n in all_native if n and "delay" in n], axis=0)
            native_params_save = {"type": "K", "delay": delays}
        elif ntype == "Kcross" and "delay_pq" in np0:
            dpq = np.stack([n["delay_pq"] for n in all_native if n and "delay_pq" in n], axis=0)
            native_params_save = {"type": "Kcross", "delay_pq": dpq}

    # Build provenance
    preapply_chain = _describe_preapply(sb, step_idx)

    meta_dict = {
        "field_ra":        field_ra,
        "field_dec":       field_dec,
        "solint_s":        sb.time_interval[step_idx],
        "freqint_hz":      sb.freq_interval[step_idx] or 0.0,
        "phase_only":      sb.phase_only[step_idx],
        "ref_ant":         sb.ref_ant,
        "scan_ids":        json.dumps(field_scans or []),
        "preapply_chain":  json.dumps(preapply_chain),
        "ms":              sb.ms,
    }

    provenance = {
        "ms":              sb.ms,
        "jones_chain":     sb.jones,
        "step_idx":        step_idx,
        "field":           field_name,
        "ref_ant":         sb.ref_ant,
        "preapply_chain":  preapply_chain,
    }

    save_solutions(
        path=sb.output,
        jones_type=jones_type,
        field_name=field_name,
        spw=spw,
        jones=jones_all,
        times=times_all,
        freqs=sol_freqs,
        flags=None,
        quality=None,
        meta=meta_dict,
        native_params=native_params_save,
        provenance=provenance,
    )

    _log(f"        saved {jones_type}/field_{field_name}/spw_{spw}  "
         f"n_slots={len(times_all)}  → {sb.output}")


# ============================================================
# PREAPPLY HELPERS
# ============================================================

def _update_internal_stack(
    stack: Dict,
    h5_path: str,
    jones_type: str,
    spw: int,
) -> None:
    """Load just-solved solutions into the internal stack."""
    fields = load_all_fields(h5_path, jones_type, spw)
    if jones_type not in stack:
        stack[jones_type] = {}
    stack[jones_type].update(fields)


def _build_preapply_stack(
    sb: SolveBlock,
    step_idx: int,
    field_name: str,
    spw: int,
    freqs: np.ndarray,
    row_times: np.ndarray,
    meta,
    internal_stack: Dict,
) -> List[np.ndarray]:
    """Build the ordered list of Jones arrays to preapply before this step's solve.

    Order: [external_preapply terms] + [internal chain steps 0..step_idx-1]

    Returns list of (n_row, n_ant, [n_freq,] 2, 2) — one per preapply term.
    """
    unique_times = np.unique(row_times)
    n_rows = len(row_times)
    preapply_list = []

    fid = meta.field_names.index(field_name) if field_name in meta.field_names else 0
    target_ra  = meta.field_ra[fid]
    target_dec = meta.field_dec[fid]

    # External preapply (always first, if supplied)
    if sb.external_preapply is not None:
        ep = sb.external_preapply
        for ep_idx, (ep_table, ep_jones, ep_fs, ep_ti) in enumerate(zip(
            ep.tables, ep.jones, ep.field_select, ep.time_interp
        )):
            fields_data = load_all_fields(ep_table, ep_jones, spw)
            if not fields_data:
                logger.warning(f"external_preapply: no {ep_jones} in {ep_table}")
                continue
            fields_data_fmt = {
                fn: {
                    "times":         sol["time"],
                    "freqs":         sol.get("freq"),
                    "jones":         sol["jones"],
                    "ra_rad":        sol.get("ra_rad", 0.0),
                    "dec_rad":       sol.get("dec_rad", 0.0),
                    "native_params": sol.get("native_params"),
                }
                for fn, sol in fields_data.items()
            }
            ep_fields = ep.fields[ep_idx] if ep.fields else None
            J = interpolate_jones_multifield(
                fields_data=fields_data_fmt,
                target_times=unique_times,
                target_freqs=freqs,
                field_select=ep_fs,
                time_interp=ep_ti,
                target_ra=target_ra,
                target_dec=target_dec,
                pinned_fields=ep_fields,
            )
            # Expand to row axis
            preapply_list.append(_expand_to_rows(J, unique_times, row_times))

    # Internal chain: steps 0..step_idx-1
    for prev_idx in range(step_idx):
        prev_jones_type = sb.jones[prev_idx]
        if prev_jones_type not in internal_stack:
            continue
        fields_data_fmt = {
            fn: {
                "times":         sol["time"],
                "freqs":         sol.get("freq"),
                "jones":         sol["jones"],
                "ra_rad":        sol.get("ra_rad", 0.0),
                "dec_rad":       sol.get("dec_rad", 0.0),
                "native_params": sol.get("native_params"),
            }
            for fn, sol in internal_stack[prev_jones_type].items()
        }
        if not fields_data_fmt:
            continue

        interp_mode = sb.preapply_time_interp[prev_idx]

        J = interpolate_jones_multifield(
            fields_data=fields_data_fmt,
            target_times=unique_times,
            target_freqs=freqs,
            field_select="nearest_time",
            time_interp=interp_mode,
            target_ra=target_ra,
            target_dec=target_dec,
        )
        preapply_list.append(_expand_to_rows(J, unique_times, row_times))

    return preapply_list


def _expand_to_rows(
    J: np.ndarray,         # (n_unique_t, n_ant, [n_freq,] 2, 2)
    unique_times: np.ndarray,
    row_times: np.ndarray,
) -> np.ndarray:
    """Expand (n_unique_t, ...) to (n_row, ...) by mapping row times to slots."""
    t_to_idx = {t: i for i, t in enumerate(unique_times)}
    row_indices = np.array([t_to_idx.get(t, 0) for t in row_times])
    return J[row_indices]


def _apply_preapply(
    vis: np.ndarray,          # (n_row, n_chan, 2, 2)
    preapply_list: List[np.ndarray],  # each (n_row, n_ant, [n_freq,] 2, 2)
    ant1: np.ndarray,
    ant2: np.ndarray,
    row_times: np.ndarray,
) -> np.ndarray:
    """Apply preapply Jones chain to visibilities (unapply = correct)."""
    result = vis.copy()
    n_row, n_chan = vis.shape[:2]

    for J_rows in preapply_list:
        corrected = np.empty_like(result)
        for r in range(n_row):
            J_r = J_rows[r]  # (n_ant, [n_freq,] 2, 2)
            if J_r.ndim == 3:
                # freq-independent — broadcast across channels
                from .jones.algebra import _inv22, _herm22, _mul22
                Ji_inv  = np.linalg.inv(J_r[ant1[r]])
                JjH_inv = np.conj(np.linalg.inv(J_r[ant2[r]])).T
                for c in range(n_chan):
                    corrected[r, c] = Ji_inv @ result[r, c] @ JjH_inv
            else:
                # freq-dependent
                for c in range(n_chan):
                    Ji_inv  = np.linalg.inv(J_r[ant1[r], c])
                    JjH_inv = np.conj(np.linalg.inv(J_r[ant2[r], c])).T
                    corrected[r, c] = Ji_inv @ result[r, c] @ JjH_inv
        result = corrected

    return result


def _compute_freqint_grid(
    freqs: np.ndarray,
    freq_int_hz: Optional[float],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Partition frequency axis into freq-interval blocks.

    Returns: (freq_blocks, freq_index_blocks)
    """
    if freq_int_hz is None:
        return [freqs], [np.arange(len(freqs))]

    blocks = []
    idx_blocks = []
    current_freqs = [freqs[0]]
    current_idx   = [0]

    for i in range(1, len(freqs)):
        if freqs[i] - freqs[current_idx[0]] >= freq_int_hz:
            blocks.append(np.array(current_freqs))
            idx_blocks.append(np.array(current_idx))
            current_freqs = [freqs[i]]
            current_idx   = [i]
        else:
            current_freqs.append(freqs[i])
            current_idx.append(i)

    if current_freqs:
        blocks.append(np.array(current_freqs))
        idx_blocks.append(np.array(current_idx))

    return blocks, idx_blocks


def _describe_preapply(sb: SolveBlock, step_idx: int) -> List[Dict]:
    """Describe what was preapplied before this step for provenance."""
    chain = []
    if sb.external_preapply:
        ep = sb.external_preapply
        for i, jt in enumerate(ep.jones):
            chain.append({
                "source":       "external",
                "table":        ep.tables[i],
                "jones":        jt,
                "field_select": ep.field_select[i],
                "time_interp":  ep.time_interp[i],
            })
    for prev_idx in range(step_idx):
        chain.append({
            "source":      "internal",
            "jones":       sb.jones[prev_idx],
            "table":       sb.output,
            "time_interp": sb.preapply_time_interp[prev_idx],
        })
    return chain
