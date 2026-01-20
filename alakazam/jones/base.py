"""
Jones Effect Base Class.

Abstract base class for all Jones matrix effects (K, B, G, D, Xf, Kcross).
Each Jones type implements solving, application, and parameter conversion.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


class JonesTypeEnum(Enum):
    """Jones matrix types."""
    K = "K"           # Delay (diagonal, frequency-dependent)
    B = "B"           # Bandpass (diagonal, frequency-dependent)
    G = "G"           # Gain (diagonal, time-dependent)
    D = "D"           # Leakage (off-diagonal)
    Kcross = "Kcross" # Cross-hand delay
    Xf = "Xf"         # Cross-hand phase


@dataclass
class JonesMetadata:
    """Metadata for a Jones effect."""
    jones_type: JonesTypeEnum
    description: str
    is_diagonal: bool
    is_frequency_dependent: bool
    is_time_dependent: bool
    needs_freq_axis: bool  # True if solver needs frequency axis (K, Kcross)
    native_param_names: List[str]  # e.g., ['delay_X', 'delay_Y']
    constraints: Dict[str, any]  # e.g., {'ref_ant': 0, 'd_constraint': 'XY'}


@dataclass
class SolveResult:
    """Result from a Jones solve operation."""
    jones: np.ndarray           # (n_sol_time, n_sol_freq, n_ant, 2, 2) complex128
    native_params: Dict[str, np.ndarray]  # e.g., {'delay': (n_sol_time, n_sol_freq, n_ant, 2)}
    cost_init: float
    cost_final: float
    nfev: int
    success: bool
    convergence_info: Dict[str, any]
    flagging_stats: Dict[str, any]


class JonesEffect(ABC):
    """
    Abstract base class for Jones matrix effects.

    Each Jones type (K, B, G, D, Xf, Kcross) implements this interface.
    """

    def __init__(self):
        """Initialize Jones effect."""
        self._metadata = self._create_metadata()

    @abstractmethod
    def _create_metadata(self) -> JonesMetadata:
        """Create metadata for this Jones type."""
        pass

    @property
    def metadata(self) -> JonesMetadata:
        """Get metadata."""
        return self._metadata

    @property
    def jones_type(self) -> str:
        """Get Jones type as string."""
        return self._metadata.jones_type.value

    @abstractmethod
    def solve(
        self,
        vis_obs: np.ndarray,
        vis_model: np.ndarray,
        antenna1: np.ndarray,
        antenna2: np.ndarray,
        n_ant: int,
        working_ants: np.ndarray,
        freq: Optional[np.ndarray] = None,
        flags: Optional[np.ndarray] = None,
        ref_ant: int = 0,
        max_iter: int = 100,
        tol: float = 1e-10,
        **kwargs
    ) -> SolveResult:
        """
        Solve for Jones matrices.

        Parameters
        ----------
        vis_obs : ndarray
            Observed visibilities (n_bl, 2, 2) or (n_bl, n_freq, 2, 2)
        vis_model : ndarray
            Model visibilities, same shape as vis_obs
        antenna1, antenna2 : ndarray
            Antenna indices (n_bl,)
        n_ant : int
            Total number of antennas
        working_ants : ndarray
            Array of working antenna indices
        freq : ndarray, optional
            Frequencies in Hz (required for K, Kcross)
        flags : ndarray, optional
            Flags, same shape as vis_obs
        ref_ant : int
            Reference antenna index
        max_iter : int
            Maximum iterations for optimization
        tol : float
            Convergence tolerance
        **kwargs
            Additional solver-specific parameters

        Returns
        -------
        SolveResult
            Contains jones, native_params, and convergence info
        """
        pass

    @abstractmethod
    def apply(
        self,
        jones: np.ndarray,
        vis: np.ndarray,
        antenna1: np.ndarray,
        antenna2: np.ndarray
    ) -> np.ndarray:
        """
        Apply Jones correction: V_out = J_i @ V @ J_j^H

        Parameters
        ----------
        jones : ndarray
            Jones matrices (n_ant, 2, 2) or (n_freq, n_ant, 2, 2)
        vis : ndarray
            Visibilities (n_bl, 2, 2) or (n_bl, n_freq, 2, 2)
        antenna1, antenna2 : ndarray
            Antenna indices (n_bl,)

        Returns
        -------
        vis_out : ndarray
            Corrected visibilities, same shape as vis
        """
        pass

    @abstractmethod
    def unapply(
        self,
        jones: np.ndarray,
        vis: np.ndarray,
        antenna1: np.ndarray,
        antenna2: np.ndarray
    ) -> np.ndarray:
        """
        Unapply Jones correction: V_out = J_i^{-1} @ V @ J_j^{-H}

        Parameters
        ----------
        jones : ndarray
            Jones matrices (n_ant, 2, 2) or (n_freq, n_ant, 2, 2)
        vis : ndarray
            Visibilities (n_bl, 2, 2) or (n_bl, n_freq, 2, 2)
        antenna1, antenna2 : ndarray
            Antenna indices (n_bl,)

        Returns
        -------
        vis_out : ndarray
            Corrected visibilities, same shape as vis
        """
        pass

    @abstractmethod
    def inverse(self, jones: np.ndarray) -> np.ndarray:
        """
        Compute inverse of Jones matrices.

        Parameters
        ----------
        jones : ndarray
            Jones matrices (n_ant, 2, 2) or (n_freq, n_ant, 2, 2)

        Returns
        -------
        jones_inv : ndarray
            Inverse Jones matrices, same shape as input
        """
        pass

    @abstractmethod
    def to_native_params(self, jones: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Convert Jones matrices to native parameters.

        For K: extract delays in nanoseconds
        For G/B: extract amplitudes and phases
        For D: extract d_xy, d_yx

        Parameters
        ----------
        jones : ndarray
            Jones matrices

        Returns
        -------
        params : dict
            Native parameters
        """
        pass

    @abstractmethod
    def from_native_params(
        self,
        params: Dict[str, np.ndarray],
        freq: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert native parameters to Jones matrices.

        For K: convert delays to phase slopes
        For G/B: construct diagonal from amp/phase

        Parameters
        ----------
        params : dict
            Native parameters
        freq : ndarray, optional
            Frequencies (required for K, Kcross)

        Returns
        -------
        jones : ndarray
            Jones matrices
        """
        pass

    def validate_inputs(
        self,
        vis_obs: np.ndarray,
        vis_model: np.ndarray,
        antenna1: np.ndarray,
        antenna2: np.ndarray,
        n_ant: int,
        working_ants: np.ndarray,
        freq: Optional[np.ndarray] = None
    ):
        """Validate input arrays."""
        # Check shapes
        if vis_obs.shape != vis_model.shape:
            raise ValueError(f"vis_obs and vis_model shapes don't match: {vis_obs.shape} vs {vis_model.shape}")

        if len(antenna1) != len(antenna2):
            raise ValueError("antenna1 and antenna2 must have same length")

        if vis_obs.shape[0] != len(antenna1):
            raise ValueError(f"First dimension of vis_obs ({vis_obs.shape[0]}) must match antenna arrays ({len(antenna1)})")

        # Check dtypes
        if not np.iscomplexobj(vis_obs):
            raise ValueError("vis_obs must be complex")

        if not np.iscomplexobj(vis_model):
            raise ValueError("vis_model must be complex")

        # Check working antennas
        if len(working_ants) == 0:
            raise ValueError("No working antennas found")

        if np.max(working_ants) >= n_ant:
            raise ValueError(f"working_ants contains antenna index >= n_ant ({n_ant})")

        # Check frequency requirement
        if self._metadata.needs_freq_axis and freq is None:
            raise ValueError(f"{self.jones_type} solver requires freq array")

    def allocate_solution_array(
        self,
        n_sol_time: int,
        n_sol_freq: int,
        n_ant: int
    ) -> np.ndarray:
        """
        Allocate solution array with proper shape.

        Parameters
        ----------
        n_sol_time : int
            Number of solution intervals in time
        n_sol_freq : int
            Number of solution intervals in frequency
        n_ant : int
            Number of antennas

        Returns
        -------
        jones : ndarray
            Solution array (n_sol_time, n_sol_freq, n_ant, 2, 2) initialized with NaN
        """
        jones = np.full((n_sol_time, n_sol_freq, n_ant, 2, 2), np.nan, dtype=np.complex128)
        return jones

    def set_non_working_to_nan(
        self,
        jones: np.ndarray,
        working_ants: np.ndarray,
        n_ant: int
    ) -> np.ndarray:
        """
        Set non-working antennas to NaN.

        Parameters
        ----------
        jones : ndarray
            Jones matrices (n_sol_time, n_sol_freq, n_ant, 2, 2)
        working_ants : ndarray
            Array of working antenna indices
        n_ant : int
            Total number of antennas

        Returns
        -------
        jones : ndarray
            Jones with non-working antennas set to NaN
        """
        # Create mask of non-working antennas
        non_working = np.ones(n_ant, dtype=bool)
        non_working[working_ants] = False

        # Set non-working antennas to NaN
        jones[..., non_working, :, :] = np.nan

        return jones

    def __str__(self) -> str:
        """String representation."""
        return f"JonesEffect({self.jones_type})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"JonesEffect(\n"
            f"  type={self.jones_type},\n"
            f"  diagonal={self._metadata.is_diagonal},\n"
            f"  freq_dependent={self._metadata.is_frequency_dependent},\n"
            f"  time_dependent={self._metadata.is_time_dependent}\n"
            f")"
        )


__all__ = [
    'JonesEffect',
    'JonesTypeEnum',
    'JonesMetadata',
    'SolveResult',
]
