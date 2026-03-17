"""ALAKAZAM v1 Solvers.

ABC, BFS helpers, backend detection, optimizer wrappers, registry.

Every solver:
  - receives averaged data for ONE cell (n_bl, 2, 2) or (n_bl, n_chan, 2, 2)
  - returns (n_ant, 2, 2) jones + native_params + stats
  - computes its own initial guess from the data it receives

Backends: jax_scipy (default), torch_lbfgs. Falls back to scipy LM.

Developed by Arpan Pal 2026, NRAO / NCRA
"""

from __future__ import annotations

import abc
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Type

import numpy as np

logger = logging.getLogger("alakazam")


# -------------------------------------------------------------------
# Backend detection
# -------------------------------------------------------------------

def detect_device(backend: str, force_gpu: bool = False) -> str:
    if backend == "jax_scipy":
        try:
            import jax
            if any(d.platform == "gpu" for d in jax.devices()) or force_gpu:
                return "gpu"
        except Exception:
            pass
    elif backend == "torch_lbfgs":
        try:
            import torch
            if torch.cuda.is_available() or force_gpu:
                return "gpu"
        except Exception:
            pass
    return "cpu"


# -------------------------------------------------------------------
# ABC
# -------------------------------------------------------------------

class JonesSolver(abc.ABC):
    jones_type: str = ""

    def __init__(self, ref_ant: int = 0, max_iter: int = 100,
                 tol: float = 1e-10, phase_only: bool = False,
                 backend: str = "jax_scipy", device: str = "cpu"):
        self.ref_ant = ref_ant
        self.max_iter = max_iter
        self.tol = tol
        self.phase_only = phase_only
        self.backend = backend
        self.device = device

    @abc.abstractmethod
    def solve(self, vis_obs, vis_model, ant1, ant2, freqs, n_ant,
              init_jones=None) -> Dict[str, Any]:
        ...


# -------------------------------------------------------------------
# BFS helpers
# -------------------------------------------------------------------

def build_antenna_graph(ant1, ant2, n_ant):
    adj = [[] for _ in range(n_ant)]
    for a, b in zip(ant1, ant2):
        if a != b:
            adj[a].append(b)
            adj[b].append(a)
    return adj


def bfs_order(adj, root):
    visited = [False] * len(adj)
    order = []
    queue = deque([root])
    visited[root] = True
    while queue:
        node = queue.popleft()
        order.append(node)
        for nb in adj[node]:
            if not visited[nb]:
                visited[nb] = True
                queue.append(nb)
    return order


# -------------------------------------------------------------------
# Optimizer wrappers
# -------------------------------------------------------------------

def solve_lbfgsb_scipy(cost_fn, x0, max_iter, tol):
    from scipy.optimize import minimize
    result = minimize(cost_fn, x0, method="L-BFGS-B",
                      options={"maxiter": max_iter, "ftol": tol, "gtol": tol})
    return result.x, float(result.fun), result.nit, result.success


def solve_lbfgsb_jax(cost_fn_jax, x0, max_iter, tol):
    try:
        import jax
        import jax.numpy as jnp
        from jax import grad
        from scipy.optimize import minimize

        grad_fn = grad(cost_fn_jax)

        def cost_and_grad(x):
            x_jax = jnp.array(x, dtype=jnp.float64)
            c = float(cost_fn_jax(x_jax))
            g = np.array(grad_fn(x_jax), dtype=np.float64)
            return c, g

        result = minimize(cost_and_grad, np.array(x0, dtype=np.float64),
                          method="L-BFGS-B", jac=True,
                          options={"maxiter": max_iter, "ftol": tol, "gtol": tol})
        return result.x, float(result.fun), result.nit, result.success
    except ImportError:
        return None
    except Exception as e:
        logger.warning(f"JAX solver failed ({e}), falling back to scipy LM")
        return None


def solve_lbfgs_torch(cost_fn_np, x0, max_iter, tol):
    try:
        import torch
        x_t = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [x_t], max_iter=max_iter, tolerance_grad=tol,
            tolerance_change=tol, line_search_fn="strong_wolfe")
        n_iter = [0]
        final_cost = [0.0]

        def closure():
            optimizer.zero_grad()
            x_np = x_t.detach().numpy().copy()
            c = cost_fn_np(x_np)
            eps = 1e-7
            g = np.zeros_like(x_np)
            for i in range(len(x_np)):
                xp = x_np.copy(); xp[i] += eps
                xm = x_np.copy(); xm[i] -= eps
                g[i] = (cost_fn_np(xp) - cost_fn_np(xm)) / (2 * eps)
            x_t.grad = torch.tensor(g, dtype=torch.float64)
            n_iter[0] += 1
            final_cost[0] = c
            return torch.tensor(c, dtype=torch.float64)

        optimizer.step(closure)
        return x_t.detach().numpy().copy(), final_cost[0], n_iter[0], True
    except ImportError:
        return None


# -------------------------------------------------------------------
# Initial guess helpers
# -------------------------------------------------------------------

def initial_guess_gain_bfs(vis_obs, vis_model, ant1, ant2, n_ant, ref_ant):
    """BFS gain-ratio extraction from parallel hands.

    For baselines to ref_ant:
      ratio = vis_obs[bl, pp] / vis_model[bl, pp]
      amp_a = |ratio|, phase_a = angle(ratio)
    BFS propagation for antennas not directly connected.

    Returns: (amp(n_ant,2), phase(n_ant,2))
    """
    adj = build_antenna_graph(ant1, ant2, n_ant)
    order = bfs_order(adj, ref_ant)

    amp = np.ones((n_ant, 2), dtype=np.float64)
    phase = np.zeros((n_ant, 2), dtype=np.float64)
    solved = np.zeros(n_ant, dtype=bool)
    solved[ref_ant] = True

    # Pre-compute per-baseline gain ratios
    n_bl = len(ant1)
    bl_ratio = np.zeros((n_bl, 2), dtype=np.complex128)
    for k in range(n_bl):
        for pol in range(2):
            m = vis_model[k, pol, pol]
            if np.abs(m) > 1e-30:
                bl_ratio[k, pol] = vis_obs[k, pol, pol] / m

    for a in order:
        if a == ref_ant:
            continue
        for k in range(n_bl):
            a1, a2 = ant1[k], ant2[k]
            if a1 == a and solved[a2]:
                # ratio ≈ g_a * conj(g_a2)
                for pol in range(2):
                    g_a2 = amp[a2, pol] * np.exp(1j * phase[a2, pol])
                    if np.abs(g_a2) > 1e-30 and np.abs(bl_ratio[k, pol]) > 1e-30:
                        g_a = bl_ratio[k, pol] / np.conj(g_a2)
                        amp[a, pol] = np.abs(g_a)
                        phase[a, pol] = np.angle(g_a)
                solved[a] = True
                break
            if a2 == a and solved[a1]:
                for pol in range(2):
                    g_a1 = amp[a1, pol] * np.exp(1j * phase[a1, pol])
                    if np.abs(g_a1) > 1e-30 and np.abs(bl_ratio[k, pol]) > 1e-30:
                        g_a = np.conj(bl_ratio[k, pol] / g_a1)
                        amp[a, pol] = np.abs(g_a)
                        phase[a, pol] = np.angle(g_a)
                solved[a] = True
                break

    phase[ref_ant, :] = 0.0
    return amp, phase


def initial_guess_leakage(vis_obs, vis_model, ant1, ant2, n_ant, ref_ant):
    """First-order leakage estimate from cross/parallel ratio.

    d_pq_a ≈ V_pq / M_pp  on baselines to ref_ant
    d_qp_a ≈ V_qp / M_qq  on baselines to ref_ant

    Returns: (d_pq(n_ant,) complex, d_qp(n_ant,) complex)
    """
    d_pq = np.zeros(n_ant, dtype=np.complex128)
    d_qp = np.zeros(n_ant, dtype=np.complex128)
    count_pq = np.zeros(n_ant, dtype=np.float64)
    count_qp = np.zeros(n_ant, dtype=np.float64)

    for k in range(len(ant1)):
        a1, a2 = ant1[k], ant2[k]
        m_pp = vis_model[k, 0, 0]
        m_qq = vis_model[k, 1, 1]

        if a2 == ref_ant and np.abs(m_pp) > 1e-30:
            # V_pq ≈ d_pq_a1 * M_pp  (d_pq_ref = 0)
            d_pq[a1] += vis_obs[k, 0, 1] / m_pp
            count_pq[a1] += 1.0
        if a1 == ref_ant and np.abs(m_qq) > 1e-30:
            # V_qp ≈ d_qp_a2 * M_qq
            d_qp[a2] += vis_obs[k, 1, 0] / m_qq
            count_qp[a2] += 1.0
        # Also other direction
        if a1 == ref_ant and np.abs(m_pp) > 1e-30:
            d_pq[a2] += np.conj(vis_obs[k, 1, 0] / m_pp)
            count_pq[a2] += 1.0
        if a2 == ref_ant and np.abs(m_qq) > 1e-30:
            d_qp[a1] += np.conj(vis_obs[k, 0, 1] / m_qq)
            count_qp[a1] += 1.0

    for a in range(n_ant):
        if count_pq[a] > 0:
            d_pq[a] /= count_pq[a]
        if count_qp[a] > 0:
            d_qp[a] /= count_qp[a]

    d_pq[ref_ant] = 0.0
    return d_pq, d_qp


def initial_guess_cross_phase(vis_obs, vis_model, ant1, ant2):
    """Mean cross-hand phase from corrected data.

    RIME: V_pq = J_i[0,0] M_pq conj(J_j[1,1])
    With J=diag(1, e^{i phi}): V_pq = M_pq e^{-i phi}
    So phi = -angle(mean(V_pq / M_pq))

    Returns: scalar float (radians)
    """
    n_bl = len(ant1)
    ratios = []
    for k in range(n_bl):
        m = vis_model[k, 0, 1]
        if np.abs(m) > 1e-30:
            ratios.append(vis_obs[k, 0, 1] / m)
    if not ratios:
        return 0.0
    mean_ratio = np.mean(ratios)
    return -float(np.angle(mean_ratio))


# -------------------------------------------------------------------
# Registry
# -------------------------------------------------------------------

from .parallel_delay import ParallelDelaySolver
from .gains import GainsSolver
from .leakage import LeakageSolver
from .cross_delay import CrossDelaySolver
from .cross_phase import CrossPhaseSolver

SOLVER_REGISTRY: Dict[str, Type[JonesSolver]] = {
    "K":  ParallelDelaySolver,
    "G":  GainsSolver,
    "D":  LeakageSolver,
    "KC": CrossDelaySolver,
    "CP": CrossPhaseSolver,
}


def get_solver(jones_type: str, **kwargs) -> JonesSolver:
    if jones_type not in SOLVER_REGISTRY:
        raise ValueError(f"Unknown Jones type: {jones_type!r}. "
                          f"Valid: {list(SOLVER_REGISTRY.keys())}")
    return SOLVER_REGISTRY[jones_type](**kwargs)
