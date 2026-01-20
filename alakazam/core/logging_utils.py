"""
Logging utilities for calibration solver.

Provides formatted tables and progress tracking.
"""

import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger('jackal')


def format_solution_table(jones_type: str, solution_data: Dict, working_ants: np.ndarray) -> str:
    """
    Format solution data as ASCII table.

    Parameters
    ----------
    jones_type : str
        Type of Jones matrix
    solution_data : dict
        Solution statistics per antenna
    working_ants : np.ndarray
        Working antenna indices

    Returns
    -------
    table_str : str
        Formatted table string
    """
    lines = []
    lines.append("")
    lines.append("="*80)
    lines.append(f"  {jones_type} SOLUTION SUMMARY")
    lines.append("="*80)
    lines.append(f"  {'Ant':>4} | {'Status':>10} | {'Data':>40}")
    lines.append("  " + "-"*76)

    for ant in range(solution_data.get('n_ant', len(working_ants))):
        if ant in working_ants:
            status = "WORKING"
            # Extract relevant data based on jones_type
            if jones_type == 'K':
                delay_x = solution_data.get('delay', np.array([[np.nan, np.nan]]))[ant, 0]
                delay_y = solution_data.get('delay', np.array([[np.nan, np.nan]]))[ant, 1]
                data_str = f"τ_X={delay_x:7.3f}ns, τ_Y={delay_y:7.3f}ns"
            elif jones_type in ['G', 'B']:
                # Complex gain
                data_str = "gain data available"
            elif jones_type == 'D':
                data_str = "leakage data available"
            elif jones_type == 'Xf':
                data_str = "cross-phase data available"
            elif jones_type == 'Kcross':
                data_str = "cross-delay data available"
            else:
                data_str = "solution available"
        else:
            status = "FLAGGED"
            data_str = "no data"

        lines.append(f"  {ant:4d} | {status:>10} | {data_str:>40}")

    lines.append("  " + "-"*76)
    lines.append("")

    return "\n".join(lines)


def log_convergence_summary(costs_init: np.ndarray, costs_final: np.ndarray,
                           success_flags: np.ndarray, nfev_counts: np.ndarray) -> None:
    """
    Log convergence summary statistics.

    Parameters
    ----------
    costs_init : np.ndarray
        Initial costs (n_sol_time, n_sol_freq)
    costs_final : np.ndarray
        Final costs
    success_flags : np.ndarray
        Success flags
    nfev_counts : np.ndarray
        Function evaluation counts
    """
    lines = []
    lines.append("")
    lines.append("="*80)
    lines.append("  CONVERGENCE SUMMARY")
    lines.append("="*80)

    n_total = success_flags.size
    n_success = np.sum(success_flags)
    n_failed = n_total - n_success

    lines.append(f"  Total solutions: {n_total}")
    lines.append(f"  Converged:       {n_success} ({100*n_success/n_total:.1f}%)")
    lines.append(f"  Failed:          {n_failed} ({100*n_failed/n_total:.1f}%)")
    lines.append("")

    # Cost reduction statistics
    cost_reduction = costs_init / np.maximum(costs_final, 1e-20)
    valid_reductions = cost_reduction[success_flags & np.isfinite(cost_reduction)]

    if len(valid_reductions) > 0:
        lines.append(f"  Cost reduction:")
        lines.append(f"    Median:  {np.median(valid_reductions):8.2f}x")
        lines.append(f"    Mean:    {np.mean(valid_reductions):8.2f}x")
        lines.append(f"    Min:     {np.min(valid_reductions):8.2f}x")
        lines.append(f"    Max:     {np.max(valid_reductions):8.2f}x")
        lines.append("")

    # Function evaluation statistics
    valid_nfev = nfev_counts[success_flags]
    if len(valid_nfev) > 0:
        lines.append(f"  Function evaluations:")
        lines.append(f"    Median:  {np.median(valid_nfev):.0f}")
        lines.append(f"    Mean:    {np.mean(valid_nfev):.0f}")
        lines.append(f"    Min:     {np.min(valid_nfev):.0f}")
        lines.append(f"    Max:     {np.max(valid_nfev):.0f}")

    lines.append("="*80)
    lines.append("")

    logger.info("\n".join(lines))


def log_rfi_summary(rfi_stats: Dict) -> None:
    """
    Log RFI flagging summary.

    Parameters
    ----------
    rfi_stats : dict
        RFI statistics
    """
    lines = []
    lines.append("")
    lines.append("="*80)
    lines.append("  RFI FLAGGING SUMMARY")
    lines.append("="*80)

    total_samples = rfi_stats.get('total_samples', 0)
    total_flagged = rfi_stats.get('total_flagged', 0)

    if total_samples > 0:
        frac = total_flagged / total_samples
        lines.append(f"  Total samples:   {total_samples}")
        lines.append(f"  Flagged:         {total_flagged} ({100*frac:.2f}%)")
        lines.append(f"  Unflagged:       {total_samples - total_flagged} ({100*(1-frac):.2f}%)")
    else:
        lines.append("  No RFI statistics available")

    lines.append("="*80)
    lines.append("")

    logger.info("\n".join(lines))


def log_progress(current: int, total: int, message: str = "Processing") -> None:
    """
    Log progress indicator.

    Parameters
    ----------
    current : int
        Current item number
    total : int
        Total items
    message : str
        Progress message
    """
    pct = 100 * current / total if total > 0 else 0
    bar_length = 40
    filled = int(bar_length * current / total) if total > 0 else 0
    bar = '█' * filled + '░' * (bar_length - filled)

    logger.info(f"  {message}: [{bar}] {current}/{total} ({pct:.1f}%)")


__all__ = ['format_solution_table', 'log_convergence_summary', 'log_rfi_summary', 'log_progress']
