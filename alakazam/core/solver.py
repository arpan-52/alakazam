"""
Core Calibration Solver Engine.

Implements the complete calibration flow:
1. Detect MS metadata (chunked loading)
2. Compute solution intervals
3. Allocate solution arrays
4. Detect non-working antennas globally
5. For each solint chunk:
   - Load data
   - Apply pre-solved Jones (with interpolation)
   - RFI flagging
   - Average based on Jones type
   - Chain + LM solve
   - Store solution
6. Write solutions to HDF5
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import logging

from .metadata import MSMetadata, detect_ms_metadata, detect_non_working_antennas_chunked
from .chunking import iterate_chunks, DataChunk
from .interpolation import interpolate_jones_to_chunk
from .rfi import flag_rfi_mad
from .averaging import average_visibilities, average_model_visibilities
from ..jones.base import JonesEffect, SolveResult

logger = logging.getLogger('jackal')


@dataclass
class SolverConfig:
    """Configuration for calibration solver."""

    # Jones solver
    jones_effect: JonesEffect
    jones_type: str  # 'K', 'G', 'B', 'D', 'Xf', 'Kcross'

    # Solution intervals
    solint_time: str  # e.g., '60s', '1min'
    solint_freq: str  # e.g., '4MHz', '8chan'

    # Reference antenna
    ref_ant: int = 0

    # Feed basis
    feed_basis: str = 'linear'

    # MS Selection
    field: Optional[str] = None  # Field name
    spw: Optional[str] = None  # Spectral window: '0', '0~2', '0,2,4'
    scans: Optional[str] = None  # Scan selection: '1', '1~10', '1,3,5'

    # Column selection
    data_col: str = 'DATA'  # Observed data column
    model_col: str = 'MODEL_DATA'  # Model data column

    # RFI flagging
    rfi_enable: bool = True
    rfi_threshold: float = 5.0

    # Solver parameters
    max_iter: int = 100
    tol: float = 1e-10

    # Pre-apply Jones
    pre_apply_jones: Optional[Dict[str, np.ndarray]] = None  # {'K': jones_K, 'G': jones_G, ...}
    pre_apply_metadata: Optional[Dict[str, MSMetadata]] = None  # Metadata for pre-apply Jones

    # Additional solver kwargs
    solver_kwargs: Dict = None

    def __post_init__(self):
        if self.solver_kwargs is None:
            self.solver_kwargs = {}


@dataclass
class SolverResult:
    """Result from calibration solver."""

    # Solutions
    jones: np.ndarray  # (n_sol_time, n_sol_freq, n_ant, 2, 2)
    native_params: Dict

    # Metadata
    metadata: MSMetadata
    config: SolverConfig

    # Convergence info per solution
    costs_init: np.ndarray  # (n_sol_time, n_sol_freq)
    costs_final: np.ndarray
    success_flags: np.ndarray  # bool (n_sol_time, n_sol_freq)
    nfev_counts: np.ndarray  # int (n_sol_time, n_sol_freq)

    # Working antennas
    working_ants: np.ndarray  # Global working antennas

    # RFI statistics
    rfi_stats: Dict


class CalibrationSolver:
    """
    Main calibration solver implementing the complete flow.
    """

    def __init__(self, ms_path: Union[str, Path], config: SolverConfig):
        """
        Initialize calibration solver.

        Parameters
        ----------
        ms_path : str or Path
            Path to measurement set
        config : SolverConfig
            Solver configuration
        """
        self.ms_path = Path(ms_path)
        self.config = config
        self.metadata = None

        logger.info("="*80)
        logger.info(f"Initializing calibration solver: {config.jones_type}")
        logger.info(f"  MS: {self.ms_path}")
        logger.info(f"  Solution intervals: time={config.solint_time}, freq={config.solint_freq}")
        logger.info(f"  Reference antenna: {config.ref_ant}")
        logger.info(f"  Feed basis: {config.feed_basis}")
        if config.field:
            logger.info(f"  Field: {config.field}")
        if config.spw:
            logger.info(f"  SPW: {config.spw}")
        if config.scans:
            logger.info(f"  Scans: {config.scans}")
        logger.info(f"  Data column: {config.data_col}")
        logger.info(f"  Model column: {config.model_col}")
        logger.info("="*80)

    def solve(self) -> SolverResult:
        """
        Run complete calibration solver.

        Uses field/spw/scan selections and data/model columns from SolverConfig.

        Returns
        -------
        result : SolverResult
            Calibration result with solutions and metadata
        """
        # Step 1: Detect MS metadata
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Detecting MS metadata")
        logger.info("="*80)
        self.metadata = detect_ms_metadata(
            self.ms_path,
            field=self.config.field,
            spw=self.config.spw,
            scans=self.config.scans,
            solint_time=self.config.solint_time,
            solint_freq=self.config.solint_freq,
            data_col=self.config.data_col,
            model_col=self.config.model_col
        )

        logger.info(f"  MS contains:")
        logger.info(f"    Antennas: {self.metadata.n_ant}")
        logger.info(f"    Baselines: {self.metadata.n_bl}")
        logger.info(f"    Time samples: {self.metadata.n_time}")
        logger.info(f"    Frequency channels: {self.metadata.n_freq}")
        logger.info(f"    Time range: {self.metadata.time_min:.2f} - {self.metadata.time_max:.2f} s")
        logger.info(f"    Frequency range: {self.metadata.freq_min/1e6:.3f} - {self.metadata.freq_max/1e6:.3f} MHz")
        logger.info(f"  Solution intervals:")
        logger.info(f"    Time: {self.metadata.n_sol_time} intervals × {self.metadata.sol_time_seconds:.1f}s")
        logger.info(f"    Frequency: {self.metadata.n_sol_freq} intervals × {self.metadata.sol_freq_channels} channels")

        # Step 2: Detect non-working antennas
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Detecting non-working antennas")
        logger.info("="*80)
        working_ants, non_working_ants, _ = detect_non_working_antennas_chunked(
            self.ms_path,
            field=self.config.field,
            spw=self.config.spw,
            scans=self.config.scans,
            data_col=self.config.data_col,
            model_col=self.config.model_col
        )
        logger.info(f"  Working antennas: {len(working_ants)}/{self.metadata.n_ant}")
        logger.info(f"  Working antenna IDs: {list(working_ants)}")

        if self.config.ref_ant not in working_ants:
            logger.error(f"  Reference antenna {self.config.ref_ant} is not working!")
            raise ValueError(f"Reference antenna {self.config.ref_ant} is not working")

        # Step 3: Allocate solution arrays
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Allocating solution arrays")
        logger.info("="*80)
        shape = (self.metadata.n_sol_time, self.metadata.n_sol_freq, self.metadata.n_ant, 2, 2)
        logger.info(f"  Solution shape: {shape}")

        jones_solutions = np.full(shape, np.nan, dtype=np.complex128)
        costs_init = np.full((self.metadata.n_sol_time, self.metadata.n_sol_freq), np.nan, dtype=np.float64)
        costs_final = np.full((self.metadata.n_sol_time, self.metadata.n_sol_freq), np.nan, dtype=np.float64)
        success_flags = np.zeros((self.metadata.n_sol_time, self.metadata.n_sol_freq), dtype=bool)
        nfev_counts = np.zeros((self.metadata.n_sol_time, self.metadata.n_sol_freq), dtype=np.int32)

        rfi_stats_global = {
            'total_flagged': 0,
            'total_samples': 0,
            'per_chunk': []
        }

        # Step 4: Solve per solint chunk
        logger.info("\n" + "="*80)
        logger.info("STEP 4: Solving per solution interval")
        logger.info("="*80)

        n_chunks = self.metadata.n_sol_time * self.metadata.n_sol_freq
        chunk_idx = 0

        for chunk in iterate_chunks(self.metadata, self.config.data_col, self.config.model_col):
            chunk_idx += 1
            t_idx, f_idx = chunk.time_idx, chunk.freq_idx

            logger.info(f"\n--- Chunk {chunk_idx}/{n_chunks}: time={t_idx}, freq={f_idx} ---")
            logger.info(f"  Data shape: {chunk.vis_obs.shape}")
            logger.info(f"  Time range: [{chunk.time_start:.2f}, {chunk.time_end:.2f}] s")
            logger.info(f"  Freq range: [{chunk.freq_start/1e6:.3f}, {chunk.freq_end/1e6:.3f}] MHz")

            # 4a: Apply pre-solved Jones
            if self.config.pre_apply_jones:
                logger.info("  Applying pre-solved Jones corrections...")
                chunk.vis_obs = self._apply_pre_jones(chunk)

            # 4b: RFI flagging
            if self.config.rfi_enable:
                logger.info("  RFI flagging...")
                chunk.flags, rfi_stats = flag_rfi_mad(
                    chunk.vis_obs, chunk.antenna1, chunk.antenna2,
                    threshold=self.config.rfi_threshold,
                    existing_flags=chunk.flags
                )
                rfi_stats_global['total_flagged'] += rfi_stats['n_flagged_after']
                rfi_stats_global['total_samples'] += rfi_stats['n_total']
                rfi_stats_global['per_chunk'].append(rfi_stats)

            # 4c: Average based on Jones type
            logger.info(f"  Averaging for {self.config.jones_type}...")
            vis_avg, flags_avg = average_visibilities(
                chunk.vis_obs, chunk.flags, self.config.jones_type
            )
            vis_model_avg = average_model_visibilities(
                chunk.vis_model, self.config.jones_type
            )

            # 4d: Solve
            logger.info(f"  Solving {self.config.jones_type}...")
            solve_result = self.config.jones_effect.solve(
                vis_obs=vis_avg,
                vis_model=vis_model_avg,
                antenna1=chunk.antenna1,
                antenna2=chunk.antenna2,
                n_ant=self.metadata.n_ant,
                working_ants=working_ants,
                freq=chunk.freq if self.config.jones_type.upper() in ['K', 'KCROSS'] else None,
                flags=flags_avg,
                ref_ant=self.config.ref_ant,
                max_iter=self.config.max_iter,
                tol=self.config.tol,
                feed_basis=self.config.feed_basis,
                **self.config.solver_kwargs
            )

            # 4e: Store solution
            jones_solutions[t_idx, f_idx] = solve_result.jones
            costs_init[t_idx, f_idx] = solve_result.cost_init
            costs_final[t_idx, f_idx] = solve_result.cost_final
            success_flags[t_idx, f_idx] = solve_result.success
            nfev_counts[t_idx, f_idx] = solve_result.nfev

            logger.info(f"  Solution stored: success={solve_result.success}, "
                       f"cost={solve_result.cost_final:.6e}, nfev={solve_result.nfev}")

        # Step 5: Summary
        logger.info("\n" + "="*80)
        logger.info("STEP 5: Calibration complete")
        logger.info("="*80)
        n_success = np.sum(success_flags)
        n_total = success_flags.size
        logger.info(f"  Solutions converged: {n_success}/{n_total} ({100*n_success/n_total:.1f}%)")
        logger.info(f"  Cost reduction (median): {np.nanmedian(costs_init/np.maximum(costs_final, 1e-20)):.2f}x")
        if self.config.rfi_enable:
            frac_flagged = rfi_stats_global['total_flagged'] / max(rfi_stats_global['total_samples'], 1)
            logger.info(f"  RFI flagged: {rfi_stats_global['total_flagged']} / {rfi_stats_global['total_samples']} "
                       f"({100*frac_flagged:.2f}%)")

        return SolverResult(
            jones=jones_solutions,
            native_params={'jones_type': self.config.jones_type},
            metadata=self.metadata,
            config=self.config,
            costs_init=costs_init,
            costs_final=costs_final,
            success_flags=success_flags,
            nfev_counts=nfev_counts,
            working_ants=working_ants,
            rfi_stats=rfi_stats_global
        )

    def _apply_pre_jones(self, chunk: DataChunk) -> np.ndarray:
        """Apply pre-solved Jones corrections to chunk data."""
        vis_corrected = chunk.vis_obs.copy()

        for jones_type, jones_array in self.config.pre_apply_jones.items():
            jones_meta = self.config.pre_apply_metadata.get(jones_type)

            # Interpolate Jones to chunk's solution intervals
            jones_interp = interpolate_jones_to_chunk(
                jones_array, jones_meta, self.metadata,
                chunk.time_idx, chunk.freq_idx,
                jones_type
            )

            # Apply (unapply actually - divide out)
            jones_effect = self.config.jones_effect  # Use same effect class for apply
            vis_corrected = jones_effect.unapply(
                jones_interp, vis_corrected,
                chunk.antenna1, chunk.antenna2
            )

            logger.info(f"    Applied {jones_type} correction")

        return vis_corrected


__all__ = ['CalibrationSolver', 'SolverConfig', 'SolverResult']
