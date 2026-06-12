"""ALAKAZAM v1 — Radio Interferometric Calibration Pipeline.

Jones types: K (parallel delay), G (gains), D (leakage),
             KC (cross delay), CP (cross phase)

Solver backend: boa (Kokkos LM — CPU/OpenMP or GPU/CUDA depending on build)
Feed bases:     LINEAR (XX/XY/YX/YY), CIRCULAR (RR/RL/LR/LL)

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import os as _os

# Suppress casacore C++ ZENITH warnings
_os.environ.setdefault("CASACORE_LOG_LEVEL", "SEVERE")

__version__ = "1.0.0"
__author__ = "Arpan Pal"

JONES_TYPES = ("K", "G", "D", "KC", "CP")
SOLVER_BACKENDS = ("boa",)
