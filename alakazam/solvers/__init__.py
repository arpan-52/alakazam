"""
ALAKAZAM Solver Base Class.

Single ABC that all 5 Jones solvers implement.
Chain initial-guess utilities (BFS propagation).

Developed by Arpan Pal 2026, NRAO / NCRA
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, List
from scipy.optimize import least_squares
from collections import deque
import logging

logger = logging.getLogger("alakazam")


class JonesSolver(ABC):
    """Abstract base class for all Jones solvers.

    Subclasses must implement:
        jones_type, can_avg_freq, correlations, ref_constraint
        n_params, chain_initial_guess, pack_params, unpack_params,
        residual_func, native_params
    """

    # --- Class-level properties (override in subclasses) ---
    jones_type: str = ""           # 'K', 'G', 'D', 'Xf', 'Kcross'
    can_avg_freq: bool = True      # True: avg freq in chunk. False: keep channels.
    correlations: str = "parallel" # 'parallel', 'cross', 'full'
    ref_constraint: str = "identity"  # 'identity', 'phase_zero', 'leakage_zero'

    @abstractmethod
    def n_params(self, n_working: int, n_freq: int = 1,
                 phase_only: bool = False) -> int:
        """Total number of real parameters in the optimization vector."""

    @abstractmethod
    def chain_initial_guess(
        self, vis_avg: np.ndarray, model_avg: np.ndarray,
        ant1: np.ndarray, ant2: np.ndarray,
        n_ant: int, ref_ant: int, working_ants: np.ndarray,
        freq: Optional[np.ndarray] = None,
        phase_only: bool = False,
    ) -> np.ndarray:
        """Compute initial parameter guess using reference antenna chain + BFS.

        Returns: packed parameter vector (1D real array).
        """

    @abstractmethod
    def pack_params(
        self, jones: np.ndarray, n_ant: int, ref_ant: int,
        working_ants: np.ndarray, phase_only: bool = False,
    ) -> np.ndarray:
        """Convert Jones matrices → flat real parameter vector."""

    @abstractmethod
    def unpack_params(
        self, params: np.ndarray, n_ant: int, ref_ant: int,
        working_ants: np.ndarray,
        freq: Optional[np.ndarray] = None,
        phase_only: bool = False,
    ) -> np.ndarray:
        """Flat real vector → Jones matrices (n_ant, 2, 2) or (n_ant, n_freq, 2, 2)."""

    @abstractmethod
    def residual_func(
        self, params: np.ndarray,
        vis_avg: np.ndarray, model_avg: np.ndarray,
        ant1: np.ndarray, ant2: np.ndarray,
        n_ant: int, ref_ant: int, working_ants: np.ndarray,
        freq: Optional[np.ndarray] = None,
        phase_only: bool = False,
    ) -> np.ndarray:
        """Compute residual vector for least_squares.

        Returns: 1D real array of residuals.
        """

    @abstractmethod
    def get_native_params(
        self, params: np.ndarray, n_ant: int, ref_ant: int,
        working_ants: np.ndarray,
        freq: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Extract native parameters from packed vector.

        Returns: dict, e.g. {'delay': (n_ant, 2)} for K solver.
        """

    # --- Concrete solve method ---

    def solve(
        self,
        vis_avg: np.ndarray,
        model_avg: np.ndarray,
        ant1: np.ndarray,
        ant2: np.ndarray,
        n_ant: int,
        ref_ant: int,
        working_ants: np.ndarray,
        freq: Optional[np.ndarray] = None,
        phase_only: bool = False,
        max_iter: int = 100,
        tol: float = 1e-10,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict]:
        """Full solve: chain guess → least_squares → unpack.

        Parameters
        ----------
        vis_avg : ndarray
            Averaged observed visibilities.
            Shape: (n_bl, 2, 2) or (n_bl, n_freq, 2, 2) depending on solver.
        model_avg : ndarray
            Averaged model visibilities. Same shape as vis_avg.
        ant1, ant2 : ndarray (n_bl,) int32
            Antenna indices per baseline.
        n_ant : int
            Total number of antennas.
        ref_ant : int
            Reference antenna index.
        working_ants : ndarray
            Indices of working antennas.
        freq : ndarray (n_freq,) float64, optional
            Frequencies in Hz. Required for K and Kcross.
        phase_only : bool
            Phase-only mode for G solver.
        max_iter : int
            Maximum LM iterations.
        tol : float
            Convergence tolerance.

        Returns
        -------
        jones : ndarray (n_ant, 2, 2) or (n_ant, n_freq, 2, 2)
            Solved Jones matrices (NaN for non-working antennas).
        native : dict
            Native parameters (delays, leakages, etc.)
        info : dict
            Convergence info, quality metrics.
        """
        # Ensure contiguous
        vis_avg = np.ascontiguousarray(vis_avg, dtype=np.complex128)
        model_avg = np.ascontiguousarray(model_avg, dtype=np.complex128)
        ant1 = np.ascontiguousarray(ant1, dtype=np.int32)
        ant2 = np.ascontiguousarray(ant2, dtype=np.int32)

        # Chain initial guess
        x0 = self.chain_initial_guess(
            vis_avg, model_avg, ant1, ant2, n_ant, ref_ant,
            working_ants, freq, phase_only,
        )

        # Compute initial cost
        r0 = self.residual_func(
            x0, vis_avg, model_avg, ant1, ant2,
            n_ant, ref_ant, working_ants, freq, phase_only,
        )
        cost_init = float(np.sum(r0**2))

        # Check for degenerate case
        if len(x0) == 0 or len(r0) == 0:
            jones = self.unpack_params(x0, n_ant, ref_ant, working_ants, freq, phase_only)
            native = self.get_native_params(x0, n_ant, ref_ant, working_ants, freq)
            info = {
                "cost_init": 0.0, "cost_final": 0.0, "nfev": 0,
                "success": True, "n_data": 0, "n_params": 0,
                "chi2_red": 0.0, "message": "no data",
            }
            return jones, native, info

        # Optimize
        try:
            result = least_squares(
                self.residual_func, x0, method="lm",
                args=(vis_avg, model_avg, ant1, ant2, n_ant, ref_ant,
                      working_ants, freq, phase_only),
                max_nfev=max_iter * len(x0),
                ftol=tol, xtol=tol, gtol=tol,
            )
            x_opt = result.x
            cost_final = float(2 * result.cost)
            nfev = result.nfev
            success = bool(result.success)
            message = result.message
            jac = result.jac
        except Exception as e:
            logger.warning(f"LM optimization failed: {e}. Using chain initial guess.")
            x_opt = x0
            cost_final = cost_init
            nfev = 0
            success = False
            message = str(e)
            jac = None

        # Unpack solution
        jones = self.unpack_params(x_opt, n_ant, ref_ant, working_ants, freq, phase_only)

        # Native parameters
        native = self.get_native_params(x_opt, n_ant, ref_ant, working_ants, freq)

        # Error estimation
        errors = self._compute_errors(jac, cost_final, len(r0), len(x_opt))

        # Info dict
        n_data = len(r0)
        n_par = len(x_opt)
        dof = max(1, n_data - n_par)
        chi2_red = cost_final / dof if dof > 0 else 0.0

        info = {
            "cost_init": cost_init,
            "cost_final": cost_final,
            "nfev": nfev,
            "success": success,
            "n_data": n_data,
            "n_params": n_par,
            "chi2_red": chi2_red,
            "message": message,
            "errors": errors,
        }

        return jones, native, info

    def _compute_errors(
        self, jac: Optional[np.ndarray], cost: float,
        n_data: int, n_params: int,
    ) -> Optional[np.ndarray]:
        """Compute 1-sigma parameter errors from Jacobian."""
        if jac is None or n_params == 0:
            return None

        dof = max(1, n_data - n_params)
        chi2_red = cost / dof

        try:
            H = jac.T @ jac
            Cov = np.linalg.inv(H) * chi2_red
            sigma = np.sqrt(np.abs(np.diag(Cov)))
            return sigma
        except np.linalg.LinAlgError:
            return np.full(n_params, np.nan)


# ---------------------------------------------------------------------------
# BFS chain propagation utilities
# ---------------------------------------------------------------------------

def build_antenna_graph(ant1: np.ndarray, ant2: np.ndarray, n_ant: int):
    """Build adjacency list from baseline arrays.

    Returns: dict mapping antenna → set of (neighbor, baseline_index, is_ant1)
    """
    graph = {a: [] for a in range(n_ant)}
    for bl_idx in range(len(ant1)):
        a1, a2 = int(ant1[bl_idx]), int(ant2[bl_idx])
        if a1 != a2:
            graph[a1].append((a2, bl_idx, True))   # a1 is ant1
            graph[a2].append((a1, bl_idx, False))   # a2 is ant2 → flip sign
    return graph


def bfs_order(graph: dict, ref_ant: int, working_ants: np.ndarray):
    """BFS from reference antenna, returning visit order.

    Returns: list of (antenna, parent_antenna, baseline_index, is_ref_ant1)
    """
    working_set = set(int(a) for a in working_ants)
    visited = {ref_ant}
    queue = deque()
    order = []

    # Seed from reference
    for (nbr, bl_idx, is_ant1) in graph.get(ref_ant, []):
        if nbr in working_set and nbr not in visited:
            queue.append((nbr, ref_ant, bl_idx, is_ant1))

    while queue:
        ant, parent, bl_idx, is_parent_ant1 = queue.popleft()
        if ant in visited:
            continue
        visited.add(ant)
        order.append((ant, parent, bl_idx, is_parent_ant1))

        for (nbr, bl2, is_ant1) in graph.get(ant, []):
            if nbr in working_set and nbr not in visited:
                queue.append((nbr, ant, bl2, is_ant1))

    return order


def get_ref_baseline_data(vis_avg, model_avg, ant1, ant2, ref_ant, target_ant):
    """Extract averaged visibility and model for a specific baseline.

    Returns: (vis_2x2, model_2x2, ref_is_ant1)
    """
    for bl_idx in range(len(ant1)):
        a1, a2 = int(ant1[bl_idx]), int(ant2[bl_idx])
        if a1 == ref_ant and a2 == target_ant:
            if vis_avg.ndim == 3:
                return vis_avg[bl_idx], model_avg[bl_idx], True
            else:
                return vis_avg[bl_idx], model_avg[bl_idx], True
        elif a1 == target_ant and a2 == ref_ant:
            if vis_avg.ndim == 3:
                return vis_avg[bl_idx], model_avg[bl_idx], False
            else:
                return vis_avg[bl_idx], model_avg[bl_idx], False
    return None, None, None
