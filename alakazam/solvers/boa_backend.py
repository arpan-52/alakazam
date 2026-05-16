"""boa backend helpers — import _boa, convert numpy↔boa formats.

Called by the solver classes when backend="boa".
"""

from __future__ import annotations
import importlib
import sys
from pathlib import Path
import numpy as np

_BOA_MOD = None  # cached after first successful import


def import_boa():
    """Return the _boa extension module, raising ImportError with help if absent."""
    global _BOA_MOD
    if _BOA_MOD is not None:
        return _BOA_MOD

    # Try direct import first (module on sys.path already).
    try:
        import _boa as m
        _BOA_MOD = m
        return m
    except ImportError:
        pass

    # Probe the boa build tree relative to this file.
    # alakazam/solvers/boa_backend.py → alakazam/boa/build/bindings/
    probe = Path(__file__).parent.parent / "boa" / "build" / "bindings"
    if probe.exists():
        sys.path.insert(0, str(probe))
        try:
            import _boa as m
            _BOA_MOD = m
            return m
        except ImportError:
            sys.path.pop(0)

    raise ImportError(
        "boa extension module (_boa.so) not found. "
        "Build it with: cmake --build boa/build --target _boa  "
        "(requires Kokkos/KokkosKernels built with BUILD_SHARED_LIBS=ON)"
    )


def vis22_to_boa(vis: np.ndarray) -> np.ndarray:
    """(n_bl, 2, 2) complex → (n_bl, 4) complex128 C-contiguous."""
    return np.ascontiguousarray(vis.reshape(vis.shape[0], 4), dtype=np.complex128)


def vis22f_to_boa(vis: np.ndarray) -> np.ndarray:
    """(n_bl, n_freq, 2, 2) complex → (n_bl*n_freq, 4) complex128 C-contiguous.

    Row order: bl0_f0, bl0_f1, ..., bl0_fn-1, bl1_f0, ... (baseline-major).
    This matches the boa K/KC solver's (b*n_freq + f) indexing.
    """
    n_bl, n_freq = vis.shape[0], vis.shape[1]
    return np.ascontiguousarray(vis.reshape(n_bl * n_freq, 4), dtype=np.complex128)


def jones4_to_22(j4: np.ndarray) -> np.ndarray:
    """(n_ant, 4) complex → (n_ant, 2, 2) complex."""
    return j4.reshape(j4.shape[0], 2, 2)


def make_opts(max_iter: int, tol: float, linear_solver: str = "cholesky"):
    """Build a SolverOptions with common settings."""
    boa = import_boa()
    opts = boa.SolverOptions()
    opts.max_iter = max_iter
    opts.tol = tol
    opts.linear_solver = linear_solver
    return opts


def reconstruct_delay(params: np.ndarray, n_ant: int, ref_ant: int) -> np.ndarray:
    """Reconstruct (n_ant, 2) delay array [tau_p, tau_q] in ns from boa params.

    boa K solver stores [tau_p, tau_q] per non-ref antenna in ascending order.
    ref_ant delay is set to 0.
    """
    delay = np.zeros((n_ant, 2), dtype=np.float64)
    k = 0
    for a in range(n_ant):
        if a != ref_ant:
            delay[a, 0] = params[k]
            delay[a, 1] = params[k + 1]
            k += 2
    return delay
