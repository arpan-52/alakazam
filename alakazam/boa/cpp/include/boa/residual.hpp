// boa/residual.hpp — Residual kernels for the calibration RIME.
//
// Two residual equations, covering all 5 solver types:
//
// 1. Diagonal RIME (G, K, KC, CP):
//    r_pp = V_obs_pp - g_i_p * V_M_pp * conj(g_j_p)
//    r_qq = V_obs_qq - g_i_q * V_M_qq * conj(g_j_q)
//    Each baseline produces 4 real residuals (re/im for pp and qq).
//    For freq-dependent solvers (K, KC), this is repeated per channel.
//
// 2. Full 2x2 RIME (D):
//    R = V_obs - J_i * V_M * J_j†
//    Each baseline produces 8 real residuals (re/im for all 4 elements).
//
// The residual vector r is a flat real array. The ordering is:
//   baseline 0 residuals, baseline 1 residuals, ...
// Within each baseline (diagonal):
//   [Re(r_pp), Im(r_pp), Re(r_qq), Im(r_qq)]
// Within each baseline (full 2x2):
//   [Re(R00), Im(R00), Re(R01), Im(R01), Re(R10), Im(R10), Re(R11), Im(R11)]

#ifndef BOA_RESIDUAL_HPP
#define BOA_RESIDUAL_HPP

#include <boa/types.hpp>

namespace boa {

// ---------------------------------------------------------------------------
// Diagonal residual (freq-independent): used by G solver
// ---------------------------------------------------------------------------
// g_p, g_q: (n_ant,) complex — precomputed Jones diagonal elements.
// vis_obs, vis_model: (n_bl, 4) complex — flattened 2x2.
// ant1, ant2: (n_bl,) int.
// r: output (n_bl * 4,) real.

void build_residual_diagonal(
    const view_1d_real& r,
    const view_1d_complex& g_p,
    const view_1d_complex& g_q,
    const view_2d_complex& vis_obs,
    const view_2d_complex& vis_model,
    const view_1d_int& ant1,
    const view_1d_int& ant2,
    int n_bl);

// ---------------------------------------------------------------------------
// Diagonal residual (freq-dependent): used by K, KC solvers
// ---------------------------------------------------------------------------
// g_p, g_q: (n_ant, n_freq) complex — Jones diagonal per antenna per freq.
// vis_obs, vis_model: (n_bl * n_freq, 4) complex — flattened, freq-major.
//   Row index = bl * n_freq + f.
// r: output (n_bl * n_freq * 4,) real.

void build_residual_diagonal_freq(
    const view_1d_real& r,
    const view_2d_complex& g_p,
    const view_2d_complex& g_q,
    const view_2d_complex& vis_obs,
    const view_2d_complex& vis_model,
    const view_1d_int& ant1,
    const view_1d_int& ant2,
    int n_bl,
    int n_freq);

// ---------------------------------------------------------------------------
// Full 2x2 residual: used by D solver
// ---------------------------------------------------------------------------
// J_current: (n_ant,) Mat2 — current Jones estimates.
// vis_obs, vis_model: (n_bl, 4) complex — flattened 2x2.
// r: output (n_bl * 8,) real.

void build_residual_full_2x2(
    const view_1d_real& r,
    const view_1d_mat2& J_current,
    const view_2d_complex& vis_obs,
    const view_2d_complex& vis_model,
    const view_1d_int& ant1,
    const view_1d_int& ant2,
    int n_bl);

// ---------------------------------------------------------------------------
// Residual norm: ||r||^2
// ---------------------------------------------------------------------------
// Computes the squared L2 norm of the residual vector using KokkosBlas::dot.
real_type residual_norm_sq(const view_1d_real& r);

}  // namespace boa

#endif  // BOA_RESIDUAL_HPP
