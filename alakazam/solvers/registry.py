"""Solver registry — append to solvers/__init__.py."""
# This is imported at the bottom of solvers/__init__.py
from .k_delay import KDelaySolver
from .g_gain import GGainSolver
from .d_leakage import DLeakageSolver
from .xf_crossphase import XfCrossPhaseSolver
from .kcross_delay import KcrossDelaySolver

SOLVER_REGISTRY = {
    "K": KDelaySolver,
    "G": GGainSolver,
    "D": DLeakageSolver,
    "XF": XfCrossPhaseSolver,
    "KCROSS": KcrossDelaySolver,
}


def get_solver(jones_type: str) -> "JonesSolver":
    """Get solver instance by Jones type string."""
    key = jones_type.upper()
    if key not in SOLVER_REGISTRY:
        raise ValueError(
            f"Unknown Jones type '{jones_type}'. "
            f"Available: {list(SOLVER_REGISTRY.keys())}"
        )
    return SOLVER_REGISTRY[key]()
