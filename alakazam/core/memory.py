"""
ALAKAZAM Memory Prediction.

Estimates peak memory usage before loading data.
Actually used by the pipeline to decide chunk sizes.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
import logging

logger = logging.getLogger("alakazam")

GB = 1024**3

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def get_available_ram_gb() -> float:
    """Get available system RAM in GB."""
    if HAS_PSUTIL:
        return psutil.virtual_memory().available / GB
    return 8.0  # conservative default


def estimate_chunk_memory_gb(
    n_rows: int, n_chan: int, n_ant: int, jones_type: str = "G"
) -> float:
    """Estimate memory for one chunk in GB.

    Accounts for: vis_obs, vis_model, flags, averaging accumulators,
    solver Jacobian, temporaries.
    """
    # Input data: 2 visibility arrays + flags
    # Each vis: n_rows × n_chan × 4 × 16 bytes (complex128)
    # Flags: n_rows × n_chan × 4 × 1 byte
    vis_bytes = n_rows * n_chan * 4 * 16 * 2  # obs + model
    flag_bytes = n_rows * n_chan * 4 * 1
    input_bytes = vis_bytes + flag_bytes

    # Averaging accumulators: n_bl × n_chan × 4 × 24 bytes (sum + count + output)
    n_bl = n_ant * (n_ant - 1) // 2
    avg_bytes = n_bl * n_chan * 4 * 24

    # Solver: Jacobian is the big one
    # Jacobian shape: (n_residuals, n_params)
    # n_residuals ≈ n_bl × 4 (or n_bl × n_chan × 4 for K)
    # n_params ≈ n_ant × 4
    if jones_type in ("K", "KCROSS"):
        n_resid = n_bl * n_chan * 4
    else:
        n_resid = n_bl * 4  # after freq averaging
    n_params = n_ant * 4
    jac_bytes = n_resid * n_params * 8  # float64

    # Temporaries: ~2× input
    temp_bytes = input_bytes

    total = input_bytes + avg_bytes + jac_bytes + temp_bytes
    return total / GB


def compute_safe_time_chunk(
    n_time: int,
    n_rows_per_time: int,
    n_chan: int,
    n_ant: int,
    jones_type: str = "G",
    memory_limit_gb: float = 0.0,
    safety_factor: float = 0.6,
) -> int:
    """Compute maximum number of time steps per chunk that fits in memory.

    Returns n_time_per_chunk (at least 1).
    """
    if memory_limit_gb <= 0:
        available = get_available_ram_gb() * safety_factor
    else:
        available = memory_limit_gb

    # Binary search for max time steps
    for n_t in range(n_time, 0, -1):
        n_rows = n_t * n_rows_per_time
        mem = estimate_chunk_memory_gb(n_rows, n_chan, n_ant, jones_type)
        if mem <= available:
            return n_t

    return 1  # minimum
