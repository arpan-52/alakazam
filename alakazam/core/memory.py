"""ALAKAZAM Memory Management.

3-tier strategy:
  T1: Full SPW fits in RAM      → single load, all cells parallel
  T2: N slots fit               → batched load, N slots at a time
  T3: One slot too big          → pseudo-chunk (progressive accumulation)

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import psutil
import numpy as np
import logging

logger = logging.getLogger("alakazam")

BYTES_PER_COMPLEX = 16  # complex128


def get_available_ram_gb() -> float:
    """Return available system RAM in GB."""
    return psutil.virtual_memory().available / (1024 ** 3)


def estimate_slot_memory_gb(
    n_baseline: int,
    n_chan: int,
    n_ant: int,
    n_matrices: int = 3,  # obs, model, preapply
) -> float:
    """Estimate RAM for one solint slot in GB."""
    vis_bytes = n_baseline * n_chan * 4 * BYTES_PER_COMPLEX * n_matrices
    jones_bytes = n_ant * n_chan * 4 * BYTES_PER_COMPLEX
    return (vis_bytes + jones_bytes) / (1024 ** 3)


def tier_strategy(
    n_slots: int,
    n_baseline: int,
    n_chan: int,
    n_ant: int,
    limit_gb: float = 0.0,
) -> tuple:
    """Return (tier, batch_size).

    tier 1 → load all slots at once
    tier 2 → load batch_size slots at a time
    tier 3 → one slot at a time (pseudo-chunk)
    """
    avail = limit_gb if limit_gb > 0 else get_available_ram_gb() * 0.7
    slot_gb = estimate_slot_memory_gb(n_baseline, n_chan, n_ant)

    if slot_gb == 0:
        return (1, n_slots)

    if slot_gb * n_slots <= avail:
        return (1, n_slots)

    batch = max(1, int(avail / slot_gb))
    if batch >= n_slots:
        return (1, n_slots)
    if batch >= 1:
        return (2, batch)
    return (3, 1)
