// boa/residual.cpp — Residual kernel implementations.
//
// Each kernel is a Kokkos::parallel_for over baselines.
// Each baseline writes to its own block of the output vector r,
// so there are no race conditions and no atomics needed.

#include <boa/residual.hpp>
#include <KokkosBlas1_dot.hpp>

namespace boa {

// ---------------------------------------------------------------------------
// Diagonal residual (freq-independent)
// ---------------------------------------------------------------------------
// For each baseline b with antennas (i, j):
//   pred_pp = g_p[i] * V_M_pp * conj(g_p[j])
//   pred_qq = g_q[i] * V_M_qq * conj(g_q[j])
//   r[4b+0] = Re(V_obs_pp - pred_pp)
//   r[4b+1] = Im(V_obs_pp - pred_pp)
//   r[4b+2] = Re(V_obs_qq - pred_qq)
//   r[4b+3] = Im(V_obs_qq - pred_qq)

void build_residual_diagonal(
    const view_1d_real& r,
    const view_1d_complex& g_p,
    const view_1d_complex& g_q,
    const view_2d_complex& vis_obs,
    const view_2d_complex& vis_model,
    const view_1d_int& ant1,
    const view_1d_int& ant2,
    int n_bl)
{
    // Kokkos::parallel_for launches n_bl parallel work items.
    // On CPU (Serial/OpenMP): regular loop or threaded loop.
    // On GPU (CUDA/HIP): one thread per baseline.
    // The lambda captures views by value (Kokkos views are lightweight handles).
    Kokkos::parallel_for("residual_diagonal", n_bl,
        KOKKOS_LAMBDA(const int b) {
            const int ai = ant1(b);
            const int aj = ant2(b);

            // vis layout: column 0 = V(0,0), col 1 = V(0,1), col 2 = V(1,0), col 3 = V(1,1)
            const complex_type obs_pp = vis_obs(b, 0);
            const complex_type mod_pp = vis_model(b, 0);
            const complex_type obs_qq = vis_obs(b, 3);
            const complex_type mod_qq = vis_model(b, 3);

            // pred_pp = g_p[i] * model_pp * conj(g_p[j])
            const complex_type pred_pp = g_p(ai) * mod_pp * Kokkos::conj(g_p(aj));
            const complex_type pred_qq = g_q(ai) * mod_qq * Kokkos::conj(g_q(aj));

            const complex_type diff_pp = obs_pp - pred_pp;
            const complex_type diff_qq = obs_qq - pred_qq;

            r(4 * b + 0) = diff_pp.real();
            r(4 * b + 1) = diff_pp.imag();
            r(4 * b + 2) = diff_qq.real();
            r(4 * b + 3) = diff_qq.imag();
        });
    // Kokkos::parallel_for is asynchronous on GPU. A fence ensures completion
    // before the residual vector is used. For correctness we fence here.
    Kokkos::fence("residual_diagonal_fence");
}

// ---------------------------------------------------------------------------
// Diagonal residual (freq-dependent)
// ---------------------------------------------------------------------------
// Same as above but looped over frequencies.
// The parallel_for is over (baseline, freq) pairs, linearized as b*n_freq+f.

void build_residual_diagonal_freq(
    const view_1d_real& r,
    const view_2d_complex& g_p,
    const view_2d_complex& g_q,
    const view_2d_complex& vis_obs,
    const view_2d_complex& vis_model,
    const view_1d_int& ant1,
    const view_1d_int& ant2,
    int n_bl,
    int n_freq)
{
    const int n_total = n_bl * n_freq;

    Kokkos::parallel_for("residual_diagonal_freq", n_total,
        KOKKOS_LAMBDA(const int idx) {
            // Decompose linear index into baseline and frequency.
            const int b = idx / n_freq;
            const int f = idx % n_freq;
            const int ai = ant1(b);
            const int aj = ant2(b);

            // vis row index: baseline-major, freq-minor.
            const int vis_row = b * n_freq + f;
            const complex_type obs_pp = vis_obs(vis_row, 0);
            const complex_type mod_pp = vis_model(vis_row, 0);
            const complex_type obs_qq = vis_obs(vis_row, 3);
            const complex_type mod_qq = vis_model(vis_row, 3);

            // g_p, g_q are (n_ant, n_freq).
            const complex_type pred_pp = g_p(ai, f) * mod_pp * Kokkos::conj(g_p(aj, f));
            const complex_type pred_qq = g_q(ai, f) * mod_qq * Kokkos::conj(g_q(aj, f));

            const complex_type diff_pp = obs_pp - pred_pp;
            const complex_type diff_qq = obs_qq - pred_qq;

            const int base = (b * n_freq + f) * 4;
            r(base + 0) = diff_pp.real();
            r(base + 1) = diff_pp.imag();
            r(base + 2) = diff_qq.real();
            r(base + 3) = diff_qq.imag();
        });
    Kokkos::fence("residual_diagonal_freq_fence");
}

// ---------------------------------------------------------------------------
// Full 2x2 residual
// ---------------------------------------------------------------------------
// For each baseline b with antennas (i, j):
//   pred = J_i * V_M * J_j†
//   R = V_obs - pred  (2x2 complex matrix)
// Flatten R into 8 reals: Re/Im of R(0,0), R(0,1), R(1,0), R(1,1).

void build_residual_full_2x2(
    const view_1d_real& r,
    const view_1d_mat2& J_current,
    const view_2d_complex& vis_obs,
    const view_2d_complex& vis_model,
    const view_1d_int& ant1,
    const view_1d_int& ant2,
    int n_bl)
{
    Kokkos::parallel_for("residual_full_2x2", n_bl,
        KOKKOS_LAMBDA(const int b) {
            const int ai = ant1(b);
            const int aj = ant2(b);

            // Load V_model as Mat2.
            Mat2 VM;
            VM(0, 0) = vis_model(b, 0);
            VM(0, 1) = vis_model(b, 1);
            VM(1, 0) = vis_model(b, 2);
            VM(1, 1) = vis_model(b, 3);

            // Load V_obs as Mat2.
            Mat2 Vobs;
            Vobs(0, 0) = vis_obs(b, 0);
            Vobs(0, 1) = vis_obs(b, 1);
            Vobs(1, 0) = vis_obs(b, 2);
            Vobs(1, 1) = vis_obs(b, 3);

            // pred = J_i * V_M * J_j†
            const Mat2 Ji = J_current(ai);
            const Mat2 Jj = J_current(aj);
            const Mat2 JiVM = mat2_multiply(Ji, VM);
            const Mat2 pred = mat2_multiply_hermitian(JiVM, Jj);

            // R = V_obs - pred → 8 real residuals.
            const int base = 8 * b;
            for (int rc = 0; rc < 4; ++rc) {
                const complex_type diff = Vobs.m[rc] - pred.m[rc];
                r(base + 2 * rc + 0) = diff.real();
                r(base + 2 * rc + 1) = diff.imag();
            }
        });
    Kokkos::fence("residual_full_2x2_fence");
}

// ---------------------------------------------------------------------------
// Residual norm squared: ||r||^2
// ---------------------------------------------------------------------------

real_type residual_norm_sq(const view_1d_real& r) {
    // KokkosBlas::dot(r, r) computes the dot product r^T r = ||r||^2.
    // This uses the backend-optimal implementation (cuBLAS on CUDA, etc.).
    return KokkosBlas::dot(r, r);
}

}  // namespace boa
