"""ALAKAZAM v1 Memory Management.

RAM probing for the 3-tier load strategy in flow.py:
  T1: all time bins fit in RAM  -> single read per bin
  T2: one time bin fits         -> read bin by bin
  T3: one bin too big           -> chunked reads with running accumulator

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import psutil


def get_available_ram_gb() -> float:
    """Return available system RAM in GB."""
    return psutil.virtual_memory().available / (1024 ** 3)
