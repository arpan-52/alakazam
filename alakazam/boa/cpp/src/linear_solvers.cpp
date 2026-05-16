// boa/linear_solvers.cpp — Dense Cholesky, CG, and LSQR implementations.
//
// All three solve the LM inner step: find delta such that
//   (J^T J + lambda I) delta ≈ -J^T r
//
// All three compute the predicted reduction for the LM gain ratio:
//   predicted_reduction = 0.5 * (||J delta||^2 + lambda * ||delta||^2)
// This equals L(0) - L(delta) for the quadratic LM surrogate model,
// which is always positive for positive lambda and nonzero delta.
//
// SpMV via KokkosSparse::spmv("N"/"T", alpha, J, x, beta, y)
//   "N" → y = alpha * J   * x + beta * y
//   "T" → y = alpha * J^T * x + beta * y
//
// BLAS via KokkosBlas:
//   dot(x, y)        → x^T y (returns scalar)
//   axpy(alpha, x, y)→ y += alpha * x
//   scal(y, alpha, x)→ y = alpha * x  (x == y allowed for in-place)

#include <boa/linear_solvers.hpp>

#include <KokkosBlas.hpp>
#include <KokkosSparse_spmv.hpp>

#include <cmath>
#include <vector>
#include <algorithm>

namespace boa {

// ---------------------------------------------------------------------------
// Shared: compute predicted_reduction = 0.5 * (||Jδ||^2 + λ||δ||^2)
// ---------------------------------------------------------------------------
// Called by all three solvers after computing delta.
// Requires one extra SpMV (J * delta).

static real_type compute_predicted_reduction(
    const crs_matrix_type& J,
    const view_1d_real& delta,
    real_type lambda)
{
    const int m = J.numRows();
    view_1d_real t("t_pred", m);
    KokkosSparse::spmv("N", real_type(1.0), J, delta, real_type(0.0), t);
    const real_type Jd_sq = KokkosBlas::dot(t, t);
    const real_type d_sq  = KokkosBlas::dot(delta, delta);
    return 0.5 * (Jd_sq + lambda * d_sq);
}

// ---------------------------------------------------------------------------
// Dense Cholesky — host-side, pure C++, no LAPACK
// ---------------------------------------------------------------------------
// Algorithm:
//   1. Mirror J and r to host.
//   2. Compute H = J^T J + lambda I and b = -J^T r by iterating over CSR rows.
//   3. Cholesky factorize H = L L^T (in-place, lower triangular).
//   4. Forward substitution: L y = b.
//   5. Back substitution: L^T x = y.
//   6. Copy result to device delta.

LinearResult solve_dense_cholesky(
    const crs_matrix_type& J,
    const view_1d_real& r,
    view_1d_real& delta,
    real_type lambda)
{
    const int m = J.numRows();
    const int n = J.numCols();

    // Mirror J structure and values to host.
    auto h_row_map = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), J.graph.row_map);
    auto h_entries = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), J.graph.entries);
    auto h_values  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), J.values);
    auto h_r       = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r);

    // H = J^T J + lambda I (dense, n x n, row-major), b = -J^T r.
    std::vector<real_type> H(n * n, 0.0);
    std::vector<real_type> b(n, 0.0);

    for (int row = 0; row < m; ++row) {
        const real_type ri = h_r(row);
        const int ks = h_row_map(row);
        const int ke = h_row_map(row + 1);

        for (int kk = ks; kk < ke; ++kk) {
            const int a = h_entries(kk);
            const real_type Jra = h_values(kk);
            b[a] -= Jra * ri;                     // b_a -= J[row,a] * r[row]
            for (int ll = ks; ll < ke; ++ll) {
                const int bb = h_entries(ll);
                H[a * n + bb] += Jra * h_values(ll);  // H[a,b] += J[row,a]*J[row,b]
            }
        }
    }

    // Add lambda to diagonal.
    for (int i = 0; i < n; ++i) H[i * n + i] += lambda;

    // Cholesky factorization: H = L L^T, stored in lower triangle of H.
    // L[j][j] = sqrt(H[j][j] - sum_{k<j} L[j][k]^2)
    // L[i][j] = (H[i][j] - sum_{k<j} L[i][k]*L[j][k]) / L[j][j]  for i > j
    for (int j = 0; j < n; ++j) {
        real_type diag = H[j * n + j];
        for (int k = 0; k < j; ++k) diag -= H[j * n + k] * H[j * n + k];
        if (diag <= 0.0) diag = 1.0e-30;  // guard against non-PD (shouldn't happen for lambda>0)
        H[j * n + j] = std::sqrt(diag);
        const real_type inv_diag = 1.0 / H[j * n + j];
        for (int i = j + 1; i < n; ++i) {
            real_type val = H[i * n + j];
            for (int k = 0; k < j; ++k) val -= H[i * n + k] * H[j * n + k];
            H[i * n + j] = val * inv_diag;
        }
    }

    // Forward substitution: L y = b.
    std::vector<real_type> y(n, 0.0);
    for (int i = 0; i < n; ++i) {
        real_type val = b[i];
        for (int k = 0; k < i; ++k) val -= H[i * n + k] * y[k];
        y[i] = val / H[i * n + i];
    }

    // Back substitution: L^T x = y. L^T is upper triangular: L^T[i][j] = L[j][i].
    std::vector<real_type> x(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        real_type val = y[i];
        for (int k = i + 1; k < n; ++k) val -= H[k * n + i] * x[k];
        x[i] = val / H[i * n + i];
    }

    // Copy result to device delta.
    auto h_delta = Kokkos::create_mirror_view(delta);
    for (int i = 0; i < n; ++i) h_delta(i) = x[i];
    Kokkos::deep_copy(delta, h_delta);

    const real_type pred = compute_predicted_reduction(J, delta, lambda);

    LinearResult res;
    res.iters              = 1;
    res.residual_norm      = 0.0;   // exact solve — no meaningful iterative residual
    res.predicted_reduction = pred;
    return res;
}

// ---------------------------------------------------------------------------
// CG — Conjugate Gradient on normal equations
// ---------------------------------------------------------------------------
// System: A x = b  where A = J^T J + lambda I, b = -J^T r, x = delta.
//
// At each CG iteration we compute A p via two SpMVs:
//   t  = J  * p       ("N")
//   Ap = J^T * t      ("T")  then  Ap += lambda * p

LinearResult solve_cg(
    const crs_matrix_type& J,
    const view_1d_real& r,
    view_1d_real& delta,
    real_type lambda,
    int max_iter,
    real_type tol)
{
    const int m = J.numRows();
    const int n = J.numCols();

    // Allocate workspace.
    view_1d_real b("cg_b", n);    // RHS = -J^T r
    view_1d_real p("cg_p", n);    // search direction
    view_1d_real Ap("cg_Ap", n);  // A * p
    view_1d_real t("cg_t", m);    // J * p (intermediate)

    // b = -J^T r
    KokkosSparse::spmv("T", real_type(-1.0), J, r, real_type(0.0), b);

    // x = 0, r_cg = b, p = b
    Kokkos::deep_copy(delta, real_type(0.0));
    Kokkos::deep_copy(p, b);
    view_1d_real r_cg("cg_r", n);
    Kokkos::deep_copy(r_cg, b);

    real_type rr = KokkosBlas::dot(r_cg, r_cg);
    const real_type b_norm_sq = rr;

    if (b_norm_sq < 1.0e-60) {
        // RHS is zero: delta = 0, no work needed.
        const real_type pred = compute_predicted_reduction(J, delta, lambda);
        LinearResult res;
        res.iters               = 0;
        res.residual_norm       = 0.0;
        res.predicted_reduction = pred;
        return res;
    }

    int iters = 0;
    real_type final_rel_res = 1.0;

    for (int k = 0; k < max_iter; ++k) {
        ++iters;

        // Ap = J^T(J p) + lambda p
        KokkosSparse::spmv("N", real_type(1.0), J, p, real_type(0.0), t);
        KokkosSparse::spmv("T", real_type(1.0), J, t, real_type(0.0), Ap);
        KokkosBlas::axpy(lambda, p, Ap);   // Ap += lambda * p

        const real_type pAp = KokkosBlas::dot(p, Ap);
        if (pAp < 1.0e-60) break;  // breakdown

        const real_type alpha = rr / pAp;

        KokkosBlas::axpy( alpha, p,  delta);  // x += alpha * p
        KokkosBlas::axpy(-alpha, Ap, r_cg);   // r_cg -= alpha * Ap

        const real_type rr_new = KokkosBlas::dot(r_cg, r_cg);
        final_rel_res = std::sqrt(rr_new / b_norm_sq);

        if (final_rel_res < tol) break;

        const real_type beta = rr_new / rr;
        KokkosBlas::scal(p, beta, p);        // p = beta * p
        KokkosBlas::axpy(1.0, r_cg, p);      // p += r_cg
        rr = rr_new;
    }

    const real_type pred = compute_predicted_reduction(J, delta, lambda);

    LinearResult res;
    res.iters               = iters;
    res.residual_norm       = final_rel_res;
    res.predicted_reduction = pred;
    return res;
}

// ---------------------------------------------------------------------------
// LSQR — Paige-Saunders 1982, with damping parameter damp = sqrt(lambda)
// ---------------------------------------------------------------------------
// Minimizes: ||J delta + r||^2 + lambda * ||delta||^2
// This is the augmented-system form of the LM step. LSQR is equivalent to CG
// on the normal equations but works with condition number sqrt of CG's, so it
// is much more numerically stable for ill-conditioned J.
//
// Algorithm follows the original Paige-Saunders (1982) paper, Algorithm LSQR,
// Section 3.5 "Practical implementation with damping".
//
// Notation:
//   beta, alpha — bidiagonalization scalars
//   phi_bar, rho_bar — running QR factors
//   cs, sn — Givens rotation cosine/sine
//   w — direction vector for updating delta

LinearResult solve_lsqr(
    const crs_matrix_type& J,
    const view_1d_real& r,
    view_1d_real& delta,
    real_type lambda,
    int max_iter,
    real_type tol)
{
    const int m = J.numRows();
    const int n = J.numCols();
    const real_type damp = std::sqrt(lambda);

    // Allocate bidiagonalization workspace.
    view_1d_real u("lsqr_u", m);   // left bidiag vector
    view_1d_real v("lsqr_v", n);   // right bidiag vector
    view_1d_real w("lsqr_w", n);   // accumulated direction for delta update

    // Initialize: u = -r (LSQR convention: b = -r, minimize ||J x - b||).
    KokkosBlas::scal(u, real_type(-1.0), r);

    real_type beta = std::sqrt(KokkosBlas::dot(u, u));
    if (beta < 1.0e-30) {
        Kokkos::deep_copy(delta, real_type(0.0));
        LinearResult res; res.iters = 0; res.residual_norm = 0.0;
        res.predicted_reduction = 0.0; return res;
    }
    KokkosBlas::scal(u, real_type(1.0 / beta), u);

    // v = J^T u
    KokkosSparse::spmv("T", real_type(1.0), J, u, real_type(0.0), v);
    real_type alpha = std::sqrt(KokkosBlas::dot(v, v));
    if (alpha < 1.0e-30) {
        Kokkos::deep_copy(delta, real_type(0.0));
        LinearResult res; res.iters = 0; res.residual_norm = 0.0;
        res.predicted_reduction = 0.0; return res;
    }
    KokkosBlas::scal(v, real_type(1.0 / alpha), v);

    // w = v, x = 0
    Kokkos::deep_copy(w, v);
    Kokkos::deep_copy(delta, real_type(0.0));

    real_type phi_bar = beta;
    real_type rho_bar = alpha;
    const real_type beta1 = beta;   // initial residual norm estimate

    int iters = 0;
    real_type phi_bar_final = phi_bar;

    for (int k = 0; k < max_iter; ++k) {
        ++iters;

        // --- Bidiagonalization ---
        // u_new = J*v - alpha*u
        KokkosSparse::spmv("N", real_type(1.0), J, v, real_type(-alpha), u);
        beta = std::sqrt(KokkosBlas::dot(u, u));
        if (beta < 1.0e-30) break;
        KokkosBlas::scal(u, real_type(1.0 / beta), u);

        // v_new = J^T*u - beta*v
        KokkosSparse::spmv("T", real_type(1.0), J, u, real_type(-beta), v);
        alpha = std::sqrt(KokkosBlas::dot(v, v));
        if (alpha < 1.0e-30) break;
        KokkosBlas::scal(v, real_type(1.0 / alpha), v);

        // --- Givens rotation for damping: zero the damp^2 term ---
        // Rotate [rho_bar; damp] → [rhobar1; 0]
        real_type rhobar1, cs1 = 1.0, sn1 = 0.0;
        if (damp > 0.0) {
            rhobar1  = std::sqrt(rho_bar * rho_bar + damp * damp);
            cs1      = rho_bar / rhobar1;
            sn1      = damp    / rhobar1;
            phi_bar *= cs1;   // rotate phi_bar through damping Givens
        } else {
            rhobar1 = rho_bar;
        }

        // --- Main Givens rotation: zero the subdiagonal beta ---
        // Rotate [rhobar1; beta] → [rho; 0]
        const real_type rho = std::sqrt(rhobar1 * rhobar1 + beta * beta);
        const real_type cs  = rhobar1 / rho;
        const real_type sn  = beta    / rho;
        const real_type phi = cs * phi_bar;
        phi_bar             = sn * phi_bar;

        // --- Update delta and w ---
        // delta += (phi/rho) * w
        KokkosBlas::axpy(phi / rho, w, delta);

        // w = v + (-theta/rho) * w_old
        // where theta = sn * alpha (off-diagonal of next bidiag step, after damping rotation)
        // and rho_bar update: rho_bar_new = -cs * alpha (modulated by damping cs1)
        const real_type theta    = sn  * alpha;
        rho_bar                  = -cs * alpha;   // for next iteration (will be damped then)

        // Incorporate damping into rho_bar for next step: the damping Givens rotation
        // applied to the bidiagonal [rho_bar; damp] at the next iteration uses the
        // rho_bar computed here (before damping). This is correct per Paige-Saunders.
        // For the w update, theta uses sn (not sn1) per the standard algorithm.
        KokkosBlas::scal(w, real_type(-theta / rho), w);   // w = -(theta/rho) * w
        KokkosBlas::axpy(real_type(1.0), v, w);            // w += v

        phi_bar_final = phi_bar;

        // Convergence: |phi_bar| is the residual norm of the augmented LS problem.
        // phi_bar can become negative when rho_bar < 0 after rho_bar = -cs*alpha;
        // use std::abs to avoid false early termination.
        if (std::abs(phi_bar) / beta1 < tol) break;
    }

    const real_type pred = compute_predicted_reduction(J, delta, lambda);

    LinearResult res;
    res.iters               = iters;
    res.residual_norm       = (beta1 > 0.0) ? phi_bar_final / beta1 : 0.0;
    res.predicted_reduction = pred;
    return res;
}

}  // namespace boa
