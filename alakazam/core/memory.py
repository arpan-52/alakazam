"""ALAKAZAM v1 Memory Management.

3-tier strategy:
  T1: Full SPW fits in RAM  -> single load, all cells parallel
  T2: N slots fit           -> batched load
  T3: One slot too big      -> single slot at a time

Also detects GPU VRAM for JAX/PyTorch GPU backends.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import logging
from typing import Tuple

import psutil
import numpy as np

logger = logging.getLogger("alakazam")

BYTES_PER_COMPLEX = 16  # complex128


def get_available_ram_gb() -> float:
    """Return available system RAM in GB."""
    return psutil.virtual_memory().available / (1024 ** 3)


def get_available_vram_gb() -> float:
    """Return available GPU VRAM in GB.  0 if no GPU."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Take first GPU
            free_mb = float(result.stdout.strip().split("\n")[0])
            return free_mb / 1024.0
    except Exception:
        pass

    # Try JAX
    try:
        import jax
        devs = [d for d in jax.devices() if d.platform == "gpu"]
        if devs:
            # JAX doesn't easily expose free VRAM; estimate 80% of total
            stats = devs[0].memory_stats()
            if stats:
                return stats.get("bytes_limit", 0) * 0.8 / (1024**3)
    except Exception:
        pass

    return 0.0


def estimate_slot_memory_gb(n_baseline: int, n_chan: int,
                            n_ant: int, n_matrices: int = 3) -> float:
    """Estimate RAM for one solint slot in GB."""
    vis_bytes = n_baseline * n_chan * 4 * BYTES_PER_COMPLEX * n_matrices
    jones_bytes = n_ant * n_chan * 4 * BYTES_PER_COMPLEX
    return (vis_bytes + jones_bytes) / (1024 ** 3)


def tier_strategy(n_slots: int, n_baseline: int, n_chan: int,
                  n_ant: int, limit_gb: float = 0.0) -> Tuple[int, int]:
    """Return (tier, batch_size).

    tier 1: load all slots at once
    tier 2: load batch_size slots at a time
    tier 3: one slot at a time
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
