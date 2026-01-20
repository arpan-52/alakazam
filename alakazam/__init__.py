"""
ALAKAZAM - Radio Interferometry Calibration

Fast radio interferometry calibration with complete refactored architecture.

Version 2.0 Features:
- Complete feed basis support (LINEAR and CIRCULAR) for all Jones types
- Scan/SPW/Field selection for targeted calibration
- Multi-SPW automatic iteration
- Model column selection (solve against any model column)
- Chunked metadata detection (no full MS loading)
- Memory-efficient per-solint processing
- RFI flagging with MAD per baseline
- Proper time/freq averaging per Jones type
- Jones interpolation for pre-apply corrections
- Comprehensive HDF5 I/O with full metadata
- Rich logging with progress tracking and timestamped log files
- All Jones types: K, B, G, D, Xf, Kcross

New Architecture Usage:
    from alakazam import CalibrationSolver, SolverConfig
    from alakazam.jones.delay import KDelay

    # Configure solver with scan/SPW selection
    config = SolverConfig(
        jones_effect=KDelay(),
        jones_type='K',
        solint_time='60s',
        solint_freq='4MHz',
        ref_ant=0,
        feed_basis='linear',
        # MS Selection (optional)
        field='3C286',          # Field name
        spw='0',                # SPW: '0', '0~2', '0,2,4'
        scans='10~20',          # Scans: '10', '10~20', '10,15,20'
        # Column selection
        data_col='CORRECTED_DATA',  # or 'DATA'
        model_col='MODEL_DATA'      # Model to solve against
    )

    # Run calibration
    solver = CalibrationSolver('obs.ms', config)
    result = solver.solve()  # Uses columns from config

    # Save solution
    from alakazam.io import write_solution_hdf5
    write_solution_hdf5('cal.h5', result)

Legacy Usage:
    from alakazam import solve, save_jones, load_jones

    # Solve for gains (legacy)
    jones, params, info = solve('G', vis_obs, vis_model, ant1, ant2, n_ant, ref_ant=0)

    # Save (legacy)
    save_jones('cal.h5', 'G', jones, time, freq)
"""

__version__ = "2.0.0"

# Core solver (legacy)
from .core import solve, flag_rfi_mad, average_per_baseline

# New architecture
from .core import (
    CalibrationSolver,
    SolverConfig,
    SolverResult,
    detect_ms_metadata,
    detect_non_working_antennas_chunked,
    MSMetadata,
    apply_calibration,
)

# Jones operations
from .jones import (
    jones_multiply,
    jones_inverse,
    jones_apply,
    jones_unapply,
    compute_residual,
    delay_to_jones,
    solve_jones,
    solve_delay,
    solve_diagonal,
    solve_leakage,
    solve_xf,
    FeedBasis,
    JonesType,
)

# Individual Jones solvers (new)
from .jones.delay import KDelay
from .jones.gain import GGain, BBandpass
from .jones.leakage import DLeakage
from .jones.crosshand import Xf, Kcross

# I/O
from .io import save_jones, load_jones, list_jones, print_summary, write_solution_hdf5, read_solution_hdf5

# Config I/O
from .config_io import config_to_yaml, config_from_yaml, solve_from_yaml

# Pipeline
from .pipeline import run_pipeline, quick_solve

__all__ = [
    # Version
    '__version__',
    # Core (legacy)
    'solve',
    'flag_rfi_mad',
    'average_per_baseline',
    # New architecture
    'CalibrationSolver',
    'SolverConfig',
    'SolverResult',
    'detect_ms_metadata',
    'detect_non_working_antennas_chunked',
    'MSMetadata',
    'apply_calibration',
    # Jones (legacy)
    'jones_multiply',
    'jones_inverse',
    'jones_apply',
    'jones_unapply',
    'compute_residual',
    'delay_to_jones',
    'solve_jones',
    'solve_delay',
    'solve_diagonal',
    'solve_leakage',
    'solve_xf',
    'FeedBasis',
    'JonesType',
    # Individual Jones solvers (new)
    'KDelay',
    'GGain',
    'BBandpass',
    'DLeakage',
    'Xf',
    'Kcross',
    # I/O
    'save_jones',
    'load_jones',
    'list_jones',
    'print_summary',
    'write_solution_hdf5',
    'read_solution_hdf5',
    # Config I/O
    'config_to_yaml',
    'config_from_yaml',
    'solve_from_yaml',
    # Pipeline
    'run_pipeline',
    'quick_solve',
]
