"""
ALAKAZAM — A Radio Interferometric Calibration Suite for Arrays.

Fast radio interferometry calibration with a single, clean architecture.

Developed by Arpan Pal 2026, NRAO / NCRA

Solvers: K (delay), G (gain/bandpass), D (leakage), Xf (cross-phase), Kcross (cross-delay)
All solvers use chain initial guess + Levenberg-Marquardt optimization.
Both LINEAR and CIRCULAR feeds supported with parallactic angle correction.

Usage:
    # From YAML config
    from alakazam.pipeline import run_from_yaml
    run_from_yaml('calibration.yaml')

    # Programmatic
    from alakazam.config import AlakazamConfig, SolveStep
    from alakazam.pipeline import run_pipeline

    config = AlakazamConfig(
        ms_path='observation.ms',
        output='solutions.h5',
        steps=[
            SolveStep(jones_type='K', time_interval='inf', freq_interval='full'),
            SolveStep(jones_type='G', time_interval='inf', freq_interval='4MHz'),
            SolveStep(jones_type='G', time_interval='2min', freq_interval='full'),
        ],
    )
    solutions = run_pipeline(config)

    # Apply
    from alakazam.apply import apply_calibration
    apply_calibration('target.ms', 'solutions.h5')

CLI:
    alakazam run config.yaml
    alakazam apply target.ms solutions.h5
    alakazam info solutions.h5
    alakazam version
"""

__version__ = "2.0.0"

# Public API
from .config import AlakazamConfig, SolveStep, load_config
from .pipeline import run_pipeline, run_from_yaml
from .apply import apply_calibration
from .fluxscale import compute_fluxscale, apply_fluxscale
from .io import save_solutions, load_solutions, print_summary

# Jones operations
from .jones import (
    FeedBasis,
    jones_multiply,
    jones_inverse,
    jones_apply,
    jones_unapply,
    delay_to_jones,
    crossdelay_to_jones,
    crossphase_to_jones,
    gain_to_jones,
    leakage_to_jones,
    compute_parallactic_angles,
    parang_to_jones,
    compose_jones_chain,
)

# Solvers
from .solvers.registry import get_solver, SOLVER_REGISTRY

__all__ = [
    "__version__",
    # Config
    "AlakazamConfig", "SolveStep", "load_config",
    # Pipeline
    "run_pipeline", "run_from_yaml",
    # Apply
    "apply_calibration",
    # Fluxscale
    "compute_fluxscale", "apply_fluxscale",
    # I/O
    "save_solutions", "load_solutions", "print_summary",
    # Jones
    "FeedBasis", "jones_multiply", "jones_inverse",
    "jones_apply", "jones_unapply",
    "delay_to_jones", "crossdelay_to_jones", "crossphase_to_jones",
    "gain_to_jones", "leakage_to_jones",
    "compute_parallactic_angles", "parang_to_jones", "compose_jones_chain",
    # Solvers
    "get_solver", "SOLVER_REGISTRY",
]
