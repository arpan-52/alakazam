"""
ALAKAZAM Solvers - One solver per Jones type.

Available Solvers:
- KDelaySolver: Antenna delays
- GGainSolver: Complex gains
- BBandpassSolver: Frequency-dependent gains
- DLeakageSolver: Polarization leakage

Usage:
    from alakazam.solvers import get_solver

    solver = get_solver('K', verbose=True)
"""

from .base import JonesSolverBase, SolverMetadata
from .k_delay import KDelaySolver
from .g_gain import GGainSolver
from .b_bandpass import BBandpassSolver
from .d_leakage import DLeakageSolver
from .kcross_delay import KcrossDelaySolver
from .xf_crossphase import XfCrossphaseSolver


# Solver registry
SOLVER_REGISTRY = {
    'K': KDelaySolver,
    'G': GGainSolver,
    'B': BBandpassSolver,
    'D': DLeakageSolver,
    'KCROSS': KcrossDelaySolver,
    'XF': XfCrossphaseSolver,
}


def get_solver(jones_type: str, **kwargs) -> JonesSolverBase:
    """
    Get solver instance for Jones type.

    Parameters
    ----------
    jones_type : str
        Jones type ('K', 'G', 'B', 'D')
    **kwargs
        Solver initialization parameters (e.g., verbose=True)

    Returns
    -------
    solver : JonesSolverBase
        Solver instance

    Raises
    ------
    ValueError
        If jones_type not recognized
    """
    jones_type = jones_type.upper()

    if jones_type not in SOLVER_REGISTRY:
        available = ', '.join(SOLVER_REGISTRY.keys())
        raise ValueError(
            f"Unknown Jones type '{jones_type}'. "
            f"Available: {available}"
        )

    solver_class = SOLVER_REGISTRY[jones_type]
    return solver_class(**kwargs)


def list_solvers():
    """List all available solvers."""
    print("Available ALAKAZAM Solvers:")
    print("=" * 60)

    for jones_type, solver_class in SOLVER_REGISTRY.items():
        meta = solver_class.metadata
        print(f"\n{jones_type} - {meta.description}")
        print(f"  Ref constraint: {meta.ref_constraint}")
        print(f"  Can avg time: {meta.can_avg_time}")
        print(f"  Can avg freq: {meta.can_avg_freq}")

    print("\n" + "=" * 60)


__all__ = [
    'JonesSolverBase',
    'SolverMetadata',
    'KDelaySolver',
    'GGainSolver',
    'BBandpassSolver',
    'DLeakageSolver',
    'KcrossDelaySolver',
    'XfCrossphaseSolver',
    'get_solver',
    'list_solvers',
    'SOLVER_REGISTRY'
]
