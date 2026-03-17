"""ALAKAZAM v1 — Radio Interferometric Calibration Pipeline.

Jones types: K (parallel delay), G (gains), D (leakage),
             KC (cross delay), CP (cross phase)

Solver backends: jax_scipy (default, CPU/GPU auto), torch_lbfgs
Feed bases:      LINEAR (XX/XY/YX/YY), CIRCULAR (RR/RL/LR/LL)

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import os as _os
# Suppress casacore C++ ZENITH warnings and JAX TPU probes
_os.environ.setdefault("CASACORE_LOG_LEVEL", "SEVERE")
_os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
_os.environ.setdefault("JAX_ENABLE_X64", "True")

import logging as _logging
_logging.getLogger("jax._src.xla_bridge").setLevel(_logging.WARNING)

__version__ = "1.0.0"
__author__ = "Arpan Pal"

JONES_TYPES = ("K", "G", "D", "KC", "CP")
SOLVER_BACKENDS = ("jax_scipy", "torch_lbfgs")
