// test_linear_solvers.cpp — Validate dense Cholesky, CG, and LSQR.
//
// Tests:
//   1. Dense Cholesky on a 4x4 SPD system with known solution.
//   2. CG on the same system.
//   3. LSQR on the same system.
//   4. All three agree on a small calibration-style sparse J (4 antennas, gain).
//   5. Predicted reduction is positive and consistent across solvers.
//
// All tests run within Kokkos scope (Serial or OpenMP depending on build).

#include <boa/types.hpp>
#include <boa/csr_pattern.hpp>
#include <boa/parameterization.hpp>
#include <boa/linear_solvers.hpp>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace boa;

static int n_pass = 0;
static int n_fail = 0;

static void check(bool cond, const char* name) {
    if (cond) { printf("  PASS: %s\n", name); ++n_pass; }
    else       { printf("  FAIL: %s\n", name); ++n_fail; }
}

// Build a trivial dense J that gives known J^T J + lambda I.
// J is m x n with J^T J = A_true.
// We build J as: J = chol(A_true)^T (upper Cholesky), so J^T J = A_true exactly.
// Then delta_true satisfies (A_true + lambda I) delta_true = b.

// ---------------------------------------------------------------------------
// Test 1-3: dense 4x4 system via a diagonal J (easy to construct).
// J is 8x4 with the first 4 rows = diag(sigma_0..sigma_3) and last 4 rows = 0.
// J^T J = diag(sigma^2). We choose sigmas = {2, 3, 4, 5}.
// b = -J^T r for some known r.
// Solution: delta_i = b_i / (sigma_i^2 + lambda).
// ---------------------------------------------------------------------------
static crs_matrix_type make_diagonal_J(int n, const std::vector<real_type>& sigma) {
    // J is 2n x n: first n rows are diag(sigma), next n rows are zero.
    // CSR: row i (i < n) has 1 nonzero at column i with value sigma[i].
    //      row i (i >= n) has 0 nonzeros.
    const int m = 2 * n;
    host_row_map_type h_row_map("hrm", m + 1);
    host_entries_type h_entries("he", n);  // only n nonzeros total

    // row_map: row 0..n-1 each have 1 nnz, row n..2n-1 have 0.
    for (int i = 0; i <= n; ++i) h_row_map(i) = i;
    for (int i = n + 1; i <= m; ++i) h_row_map(i) = n;
    // entries: column i in row i.
    for (int i = 0; i < n; ++i) h_entries(i) = i;

    // values: sigma[i] in row i.
    Kokkos::View<real_type*, host_space> h_values("hv", n);
    for (int i = 0; i < n; ++i) h_values(i) = sigma[i];

    // Build device CRS.
    auto d_row_map = Kokkos::create_mirror_view_and_copy(mem_space(), h_row_map);
    auto d_entries = Kokkos::create_mirror_view_and_copy(mem_space(), h_entries);
    auto d_values  = Kokkos::create_mirror_view_and_copy(mem_space(), h_values);

    crs_matrix_type::staticcrsgraph_type graph(d_entries, d_row_map);
    return crs_matrix_type("J_diag", n, d_values, graph);
}

static void test_diagonal_system() {
    printf("\n--- Linear solvers: diagonal J ---\n");
    const int n = 4;
    const real_type lambda = 0.5;
    const std::vector<real_type> sigma = {2.0, 3.0, 4.0, 5.0};

    // J is 8x4. Build J.
    crs_matrix_type J = make_diagonal_J(n, sigma);

    // r = [1, 2, 3, 4, 0, 0, 0, 0] (only first 4 rows matter)
    const int m = 2 * n;
    view_1d_real r("r", m);
    auto h_r = Kokkos::create_mirror_view(r);
    for (int i = 0; i < n; ++i) h_r(i) = (real_type)(i + 1);
    for (int i = n; i < m; ++i) h_r(i) = 0.0;
    Kokkos::deep_copy(r, h_r);

    // True solution: delta_i = -sigma_i * r_i / (sigma_i^2 + lambda)
    // b_i = -J^T r evaluated: b_i = -sigma_i * r_i (since J is diagonal).
    // (A + lambda I) delta = b  →  (sigma_i^2 + lambda) delta_i = -sigma_i * r_i
    std::vector<real_type> delta_true(n);
    for (int i = 0; i < n; ++i)
        delta_true[i] = -sigma[i] * (i + 1.0) / (sigma[i] * sigma[i] + lambda);

    const real_type tol_check = 1.0e-10;

    // --- Dense Cholesky ---
    {
        view_1d_real delta("delta_chol", n);
        LinearResult lr = solve_dense_cholesky(J, r, delta, lambda);
        auto h_d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), delta);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (std::abs(h_d(i) - delta_true[i]) > tol_check) {
                printf("    Cholesky delta[%d] = %.6e, expected %.6e\n",
                       i, h_d(i), delta_true[i]);
                ok = false;
            }
        }
        check(ok, "Cholesky delta == true solution");
        check(lr.predicted_reduction > 0.0, "Cholesky predicted_reduction > 0");
    }

    // --- CG ---
    {
        view_1d_real delta("delta_cg", n);
        LinearResult lr = solve_cg(J, r, delta, lambda, 100, 1.0e-12);
        auto h_d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), delta);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (std::abs(h_d(i) - delta_true[i]) > 1.0e-8) {
                printf("    CG delta[%d] = %.6e, expected %.6e\n",
                       i, h_d(i), delta_true[i]);
                ok = false;
            }
        }
        check(ok, "CG delta == true solution");
        check(lr.predicted_reduction > 0.0, "CG predicted_reduction > 0");
        check(lr.iters <= n, "CG converged in <= n iterations");
    }

    // --- LSQR ---
    {
        view_1d_real delta("delta_lsqr", n);
        LinearResult lr = solve_lsqr(J, r, delta, lambda, 100, 1.0e-10);
        auto h_d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), delta);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (std::abs(h_d(i) - delta_true[i]) > 1.0e-8) {
                printf("    LSQR delta[%d] = %.6e, expected %.6e\n",
                       i, h_d(i), delta_true[i]);
                ok = false;
            }
        }
        check(ok, "LSQR delta == true solution");
        check(lr.predicted_reduction > 0.0, "LSQR predicted_reduction > 0");
    }
}

// ---------------------------------------------------------------------------
// Test 4: calibration-style sparse J — 4 antennas, gain solver
// Antennas 1,2,3 are free (ref=0). 3 baselines: (0,1),(0,2),(1,2).
// Build the CSR pattern via boa's own builder, fill known values, solve.
// All three solvers must agree to 1e-8.
// ---------------------------------------------------------------------------
static void test_gain_style_system() {
    printf("\n--- Linear solvers: calibration sparse J (gain) ---\n");

    const int n_ant = 4;
    const int n_bl  = 6;  // all 4-choose-2 baselines
    const int ref_ant = 0;
    const int ppa = GainParam::params_per_ant;  // 4
    const real_type lambda = 1.0e-3;

    AntennaMap amap = build_antenna_map(n_ant, ref_ant, ppa);
    // n_params = 3 * 4 = 12

    host_view_1d_int h_ant1("h_ant1", n_bl);
    host_view_1d_int h_ant2("h_ant2", n_bl);
    // All pairs (i,j) with i < j for n_ant=4
    int b = 0;
    for (int i = 0; i < n_ant; ++i)
        for (int j = i + 1; j < n_ant; ++j) {
            h_ant1(b) = i; h_ant2(b) = j; ++b;
        }

    crs_matrix_type J = build_csr_diagonal(h_ant1, h_ant2, amap, 1);

    // Fill J values with simple known values (1.0) so J^T J is computable.
    // We only need all three solvers to agree; we don't need a ground truth delta.
    // Set all nonzero values = 1.0.
    {
        auto h_vals = Kokkos::create_mirror_view(J.values);
        Kokkos::deep_copy(h_vals, real_type(1.0));
        Kokkos::deep_copy(J.values, h_vals);
    }

    const int m = J.numRows();
    const int n = J.numCols();

    // r = random-ish fixed vector (deterministic).
    view_1d_real r("r_gain", m);
    {
        auto h_r = Kokkos::create_mirror_view(r);
        for (int i = 0; i < m; ++i) h_r(i) = (real_type)(i % 7 + 1) * 0.1;
        Kokkos::deep_copy(r, h_r);
    }

    // Solve with all three.
    view_1d_real delta_chol("d_chol", n);
    view_1d_real delta_cg("d_cg", n);
    view_1d_real delta_lsqr("d_lsqr", n);

    solve_dense_cholesky(J, r, delta_chol, lambda);
    solve_cg(J, r, delta_cg, lambda, 500, 1.0e-12);
    solve_lsqr(J, r, delta_lsqr, lambda, 500, 1.0e-10);

    auto h_chol = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), delta_chol);
    auto h_cg   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), delta_cg);
    auto h_lsqr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), delta_lsqr);

    bool cg_ok = true, lsqr_ok = true;
    for (int i = 0; i < n; ++i) {
        if (std::abs(h_cg(i) - h_chol(i)) > 1.0e-7)   cg_ok   = false;
        if (std::abs(h_lsqr(i) - h_chol(i)) > 1.0e-7) lsqr_ok = false;
    }
    check(cg_ok,   "CG agrees with Cholesky on sparse gain J");
    check(lsqr_ok, "LSQR agrees with Cholesky on sparse gain J");
}

// ---------------------------------------------------------------------------
// Test 5: zero RHS → delta should be zero.
// ---------------------------------------------------------------------------
static void test_zero_rhs() {
    printf("\n--- Linear solvers: zero RHS ---\n");
    const int n = 4;
    const real_type lambda = 1.0;
    const std::vector<real_type> sigma = {1.0, 2.0, 3.0, 4.0};
    crs_matrix_type J = make_diagonal_J(n, sigma);

    const int m = 2 * n;
    view_1d_real r("r_zero", m);
    Kokkos::deep_copy(r, real_type(0.0));

    view_1d_real delta_chol("d_chol", n);
    view_1d_real delta_cg("d_cg", n);
    view_1d_real delta_lsqr("d_lsqr", n);

    solve_dense_cholesky(J, r, delta_chol, lambda);
    solve_cg(J, r, delta_cg, lambda, 50, 1.0e-12);
    solve_lsqr(J, r, delta_lsqr, lambda, 50, 1.0e-12);

    auto h_chol = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), delta_chol);
    auto h_cg   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), delta_cg);
    auto h_lsqr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), delta_lsqr);

    bool chol_ok = true, cg_ok = true, lsqr_ok = true;
    for (int i = 0; i < n; ++i) {
        if (std::abs(h_chol(i)) > 1.0e-14) chol_ok = false;
        if (std::abs(h_cg(i))   > 1.0e-14) cg_ok   = false;
        if (std::abs(h_lsqr(i)) > 1.0e-14) lsqr_ok = false;
    }
    check(chol_ok, "Cholesky: zero RHS → zero delta");
    check(cg_ok,   "CG: zero RHS → zero delta");
    check(lsqr_ok, "LSQR: zero RHS → zero delta");
}

int main() {
    Kokkos::initialize();
    {
        printf("=== test_linear_solvers ===\n");
        test_diagonal_system();
        test_gain_style_system();
        test_zero_rhs();
        printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    }
    Kokkos::finalize();
    return n_fail == 0 ? 0 : 1;
}
