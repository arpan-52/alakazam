"""
JACKAL Solvers - Base class for all Jones type solvers.

Each Jones solver implements:
- chain_initial_guess(): Initial parameters from chain equation
- residual(): Residual function with flag handling
- pack_params() / unpack_params(): Parameter conversion
- print_solution(): Pretty printing

Design:
- One solver per Jones type
- Clean separation of algorithm and orchestration
- Flag-aware from the start
- User-defined solint intervals
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SolverMetadata:
    """Metadata for a Jones solver."""
    jones_type: str           # 'K', 'G', 'B', 'D', 'Xf', 'Kcross'
    ref_constraint: str       # 'identity', 'phase_zero', 'zero'
    can_avg_time: bool = True # Can average time dimension?
    can_avg_freq: bool = True # Can average freq dimension?
    description: str = ""     # Human-readable description


class JonesSolverBase(ABC):
    """
    Base class for all Jones solvers.

    Subclasses must implement:
    - chain_initial_guess()
    - residual()
    - pack_params()
    - unpack_params()
    - print_solution()
    """

    # Override in subclass
    metadata: SolverMetadata = None

    def __init__(self, verbose: bool = True):
        """
        Parameters
        ----------
        verbose : bool
            Print progress information
        """
        self.verbose = verbose

        if self.metadata is None:
            raise NotImplementedError("Subclass must define 'metadata'")

    @abstractmethod
    def chain_initial_guess(
        self,
        vis_obs: np.ndarray,
        vis_model: np.ndarray,
        antenna1: np.ndarray,
        antenna2: np.ndarray,
        freq: np.ndarray,
        flags: np.ndarray,
        ref_ant: int,
        n_ant: int,
        **kwargs
    ) -> np.ndarray:
        """
        Compute initial parameter guess using chain solver.

        Chain equation: J_j = V_ref,j × (J_ref × M_ref,j)^(-1)

        For ref antenna with known constraint (e.g., J_ref = I for delays),
        directly solve for all other antennas from baselines to ref.

        Parameters
        ----------
        vis_obs : ndarray
            Observed visibilities (after averaging per solint)
            Shape depends on Jones:
            - K: (n_bl, n_freq, 2, 2) [time-averaged, freq-full]
            - G: (n_bl, 2, 2) [time+freq averaged]
            - B: (n_bl, n_freq, 2, 2) [time-averaged, freq-chunked]
        vis_model : ndarray
            Model visibilities (same shape as vis_obs)
        antenna1, antenna2 : ndarray (n_bl,)
            Antenna indices (working antenna indices: 0 to n_ant-1)
        freq : ndarray (n_freq,)
            Frequencies in Hz
        flags : ndarray
            Flags (same shape as vis_obs, bool)
        ref_ant : int
            Reference antenna index (working antenna index)
        n_ant : int
            Number of working antennas
        **kwargs : dict
            Solver-specific parameters

        Returns
        -------
        params_init : ndarray
            Initial parameters (working antennas only)
            Shape depends on Jones type
        """
        pass

    @abstractmethod
    def residual(
        self,
        params: np.ndarray,
        vis_obs: np.ndarray,
        vis_model: np.ndarray,
        antenna1: np.ndarray,
        antenna2: np.ndarray,
        freq: np.ndarray,
        flags: np.ndarray,
        n_ant: int,
        ref_ant: int,
        **kwargs
    ) -> np.ndarray:
        """
        Compute residuals with flag handling.

        Only compute residuals for unflagged data points.

        Parameters
        ----------
        params : ndarray
            Current parameter values
        vis_obs, vis_model, antenna1, antenna2, freq, flags : ndarray
            Same as chain_initial_guess()
        n_ant : int
            Number of working antennas
        ref_ant : int
            Reference antenna index
        **kwargs : dict
            Solver-specific parameters

        Returns
        -------
        residuals : ndarray (real)
            Flattened residuals (real + imag parts) for unflagged data only
        """
        pass

    @abstractmethod
    def pack_params(
        self,
        jones: np.ndarray,
        ref_ant: int,
        **kwargs
    ) -> np.ndarray:
        """
        Pack Jones matrix to optimization parameters.

        Parameters
        ----------
        jones : ndarray
            Jones matrices (working antennas only)
        ref_ant : int
            Reference antenna index
        **kwargs : dict
            Solver-specific options

        Returns
        -------
        params : ndarray
            Flattened parameter array
        """
        pass

    @abstractmethod
    def unpack_params(
        self,
        params: np.ndarray,
        n_ant: int,
        ref_ant: int,
        **kwargs
    ) -> np.ndarray:
        """
        Unpack optimization parameters to Jones matrix.

        Parameters
        ----------
        params : ndarray
            Flattened parameter array
        n_ant : int
            Number of working antennas
        ref_ant : int
            Reference antenna index
        **kwargs : dict
            Solver-specific options

        Returns
        -------
        jones : ndarray
            Jones matrices (working antennas only)
        """
        pass

    @abstractmethod
    def print_solution(
        self,
        jones: np.ndarray,
        working_ants: np.ndarray,
        ref_ant: int,
        convergence_info: Dict[str, Any]
    ):
        """
        Print solution summary.

        Parameters
        ----------
        jones : ndarray
            Solved Jones matrices (working antennas only)
        working_ants : ndarray
            Full antenna indices of working antennas
        ref_ant : int
            Reference antenna (full index)
        convergence_info : dict
            Solver convergence information
        """
        pass

    def validate_averaging(self, avg_time: bool, avg_freq: bool):
        """
        Validate that averaging is allowed for this Jones type.

        Parameters
        ----------
        avg_time : bool
            Will time dimension be averaged?
        avg_freq : bool
            Will freq dimension be averaged?

        Raises
        ------
        ValueError
            If averaging not allowed for this Jones type
        """
        if avg_freq and not self.metadata.can_avg_freq:
            raise ValueError(
                f"{self.metadata.jones_type} requires full frequency resolution "
                f"(cannot average freq). Delay fitting needs freq dependence."
            )

        if avg_time and not self.metadata.can_avg_time:
            raise ValueError(
                f"{self.metadata.jones_type} cannot average time dimension"
            )

    def print_solver_info(
        self,
        n_working: int,
        n_total: int,
        n_bl: int,
        n_freq: int,
        ref_ant: int,
        time_solint: str,
        freq_solint: str
    ):
        """
        Print solver initialization info.

        Parameters
        ----------
        n_working : int
            Number of working antennas
        n_total : int
            Total antennas in array
        n_bl : int
            Number of baselines
        n_freq : int
            Number of frequency channels
        ref_ant : int
            Reference antenna (full index)
        time_solint : str
            Time solution interval (e.g., 'scan', '60s', 'inf')
        freq_solint : str
            Freq solution interval (e.g., 'full', '1MHz', '4MHz')
        """
        if not self.verbose:
            return

        print(f"\n{'='*60}")
        print(f"JACKAL {self.metadata.jones_type} Solver")
        print(f"{'='*60}")
        print(f"  Solint: Time={time_solint}, Freq={freq_solint}")
        print(f"  Working ants: {n_working}/{n_total}, ref={ref_ant}")
        print(f"  Baselines: {n_bl}")
        print(f"  Channels: {n_freq}")
        print(f"  {self.metadata.description}")
        print()


__all__ = ['JonesSolverBase', 'SolverMetadata']
