"""ALAKAZAM Solvers.

ABC, BFS graph helpers, and registry.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations

import abc
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class JonesSolver(abc.ABC):
    """Abstract base for all Jones solvers."""

    jones_type: str = ""

    def __init__(self, ref_ant: int = 0, max_iter: int = 100,
                 tol: float = 1e-10, phase_only: bool = False):
        self.ref_ant    = ref_ant
        self.max_iter   = max_iter
        self.tol        = tol
        self.phase_only = phase_only

    @abc.abstractmethod
    def solve(
        self,
        vis_obs: np.ndarray,     # (n_bl, [n_chan,] 2, 2) complex128
        vis_model: np.ndarray,
        ant1: np.ndarray,        # (n_bl,) int32
        ant2: np.ndarray,
        freqs: np.ndarray,       # (n_chan,) Hz
        n_ant: int,
        init_jones: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Solve and return dict with 'jones', 'native_params', 'converged', 'n_iter'."""
        ...


# ---------------------------------------------------------------------------
# BFS helpers for initial-guess propagation
# ---------------------------------------------------------------------------

def build_antenna_graph(ant1: np.ndarray, ant2: np.ndarray, n_ant: int) -> List[List[int]]:
    """Build adjacency list from baseline arrays."""
    adj = [[] for _ in range(n_ant)]
    for a, b in zip(ant1, ant2):
        adj[a].append(b)
        adj[b].append(a)
    return adj


def bfs_order(adj: List[List[int]], root: int) -> List[int]:
    """BFS traversal order from root antenna."""
    visited = [False] * len(adj)
    order   = []
    queue   = deque([root])
    visited[root] = True
    while queue:
        node = queue.popleft()
        order.append(node)
        for nb in adj[node]:
            if not visited[nb]:
                visited[nb] = True
                queue.append(nb)
    return order


# ---------------------------------------------------------------------------
# Registry — import solvers after ABC is defined
# ---------------------------------------------------------------------------

from .k_delay    import KDelaySolver
from .g_gain     import GGainSolver
from .d_leakage  import DLeakageSolver
from .xf_crossphase import XfCrossPhaseSolver
from .kcross_delay  import KcrossDelaySolver

SOLVER_REGISTRY: Dict[str, Type[JonesSolver]] = {
    "K":      KDelaySolver,
    "G":      GGainSolver,
    "D":      DLeakageSolver,
    "Xf":     XfCrossPhaseSolver,
    "Kcross": KcrossDelaySolver,
}


def get_solver(jones_type: str, **kwargs) -> JonesSolver:
    """Instantiate a solver by Jones type string."""
    if jones_type not in SOLVER_REGISTRY:
        raise ValueError(
            f"Unknown Jones type: {jones_type!r}. "
            f"Valid: {list(SOLVER_REGISTRY.keys())}"
        )
    return SOLVER_REGISTRY[jones_type](**kwargs)
