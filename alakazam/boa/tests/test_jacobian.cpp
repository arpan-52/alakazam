// test_jacobian.cpp — Validate analytic Jacobian vs finite difference.
//
// This is the single most critical test. If the Jacobian is wrong,
// no solver will converge.
//
// For each solver type, we:
//   1. Set up a small problem (2-4 antennas).
//   2. Build the CSR pattern.
//   3. Fill the analytic Jacobian.
//   4. For each parameter, perturb by ±ε and compute (r(x+ε)-r(x-ε))/(2ε).
//   5. Compare against the analytic Jacobian column.
//
// All operations happen on the host (Serial execution within Kokkos scope).

#include <boa/types.hpp>
#include <boa/csr_pattern.hpp>
#include <boa/residual.hpp>
#include <boa/jacobian.hpp>
#include <boa/parameterization.hpp>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace boa;

static int n_pass = 0;
static int n_fail = 0;
static const double FD_EPS = 1e-7;
static const double FD_TOL = 1e-5;  // relative tolerance for fd vs analytic

static void check(bool cond, const char* name) {
    if (cond) {
        printf("  PASS: %s\n", name);
        ++n_pass;
    } else {
        printf("  FAIL: %s\n", name);
        ++n_fail;
    }
}

static bool fd_close(double analytic, double fd, double tol = FD_TOL) {
    double denom = std::max(std::abs(analytic), std::abs(fd));
    if (denom < 1e-14) return true;  // both effectively zero
    return std::abs(analytic - fd) / denom < tol;
}

// Helper: read CSR value for (row, col) from host mirror.
static double get_J_entry(
    const crs_matrix_type& J,
    int row, int col)
{
    auto h_row_map = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), J.graph.row_map);
    auto h_entries = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), J.graph.entries);
    auto h_values  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), J.values);

    for (int pos = h_row_map(row); pos < h_row_map(row + 1); ++pos) {
        if (h_entries(pos) == col) return h_values(pos);
    }
    return 0.0;  // not in sparsity pattern → structurally zero
}

// --------------------------------------------------------------------------
// Test 1: Gain solver Jacobian
// --------------------------------------------------------------------------
static void test_gain_jacobian() {
    printf("\n--- Jacobian finite diff: Gain (G) ---\n");

    const int n_ant = 3;
    const int n_bl = 3;  // (0,1), (0,2), (1,2)
    const int ref_ant = 0;
    const int ppa = GainParam::params_per_ant;  // 4

    // Build antenna map.
    AntennaMap amap = build_antenna_map(n_ant, ref_ant, ppa);

    // Antenna indices.
    host_view_1d_int h_ant1("h_ant1", n_bl);
    host_view_1d_int h_ant2("h_ant2", n_bl);
    h_ant1(0) = 0; h_ant2(0) = 1;
    h_ant1(1) = 0; h_ant2(1) = 2;
    h_ant1(2) = 1; h_ant2(2) = 2;

    view_1d_int ant1("ant1", n_bl);
    view_1d_int ant2("ant2", n_bl);
    Kokkos::deep_copy(ant1, h_ant1);
    Kokkos::deep_copy(ant2, h_ant2);

    // Model visibilities.
    view_2d_complex vis_model("vis_model", n_bl, 4);
    {
        auto h = Kokkos::create_mirror_view(vis_model);
        h(0, 0) = complex_type(1.0, 0.3);  h(0, 3) = complex_type(0.8, -0.2);
        h(1, 0) = complex_type(0.7, 0.1);  h(1, 3) = complex_type(0.9, 0.4);
        h(2, 0) = complex_type(0.5, -0.3); h(2, 3) = complex_type(1.1, 0.2);
        h(0, 1) = complex_type(0,0); h(0, 2) = complex_type(0,0);
        h(1, 1) = complex_type(0,0); h(1, 2) = complex_type(0,0);
        h(2, 1) = complex_type(0,0); h(2, 2) = complex_type(0,0);
        Kokkos::deep_copy(vis_model, h);
    }

    // Observed = model (for this test we just need residual at a point).
    view_2d_complex vis_obs("vis_obs", n_bl, 4);
    Kokkos::deep_copy(vis_obs, vis_model);

    // Parameter vector: near-identity gains for non-ref antennas.
    // ant0 is ref (excluded). ant1: [1.1, 0.2, 0.9, -0.1], ant2: [0.95, 0.15, 1.05, -0.05]
    const int n_params = amap.n_params;  // (n_ant-1)*4 = 8
    view_1d_real params("params", n_params);
    {
        auto h = Kokkos::create_mirror_view(params);
        // ant1 params at offset 0
        h(0) = 1.1; h(1) = 0.2; h(2) = 0.9; h(3) = -0.1;
        // ant2 params at offset 4
        h(4) = 0.95; h(5) = 0.15; h(6) = 1.05; h(7) = -0.05;
        Kokkos::deep_copy(params, h);
    }

    // Build CSR and fill analytic Jacobian.
    crs_matrix_type Jac = build_csr_diagonal(h_ant1, h_ant2, amap, 1);
    fill_jacobian_gain(Jac, params, vis_model, ant1, ant2, amap.ant_to_param, n_bl, ppa);

    // Finite difference: for each parameter p, perturb ±ε, compute residual.
    const int n_res = n_bl * 4;
    auto h_params = Kokkos::create_mirror_view(params);
    Kokkos::deep_copy(h_params, params);

    int n_checked = 0;
    for (int p = 0; p < n_params; ++p) {
        // Perturb +ε.
        double orig = h_params(p);
        h_params(p) = orig + FD_EPS;
        Kokkos::deep_copy(params, h_params);

        // Compute g_p, g_q from params for all antennas.
        // ref_ant gets identity gains.
        view_1d_complex gp_plus("gp+", n_ant);
        view_1d_complex gq_plus("gq+", n_ant);
        {
            auto hgp = Kokkos::create_mirror_view(gp_plus);
            auto hgq = Kokkos::create_mirror_view(gq_plus);
            // ref_ant = 0
            hgp(0) = complex_type(1.0, 0.0);
            hgq(0) = complex_type(1.0, 0.0);
            for (int a = 1; a < n_ant; ++a) {
                int off = amap.h_ant_to_param(a);
                complex_type g_p_val, g_q_val;
                double pp[4] = {h_params(off), h_params(off+1), h_params(off+2), h_params(off+3)};
                GainParam::params_to_diagonal(pp, g_p_val, g_q_val);
                hgp(a) = g_p_val;
                hgq(a) = g_q_val;
            }
            Kokkos::deep_copy(gp_plus, hgp);
            Kokkos::deep_copy(gq_plus, hgq);
        }

        view_1d_real r_plus("r+", n_res);
        build_residual_diagonal(r_plus, gp_plus, gq_plus, vis_obs, vis_model, ant1, ant2, n_bl);
        auto h_rp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r_plus);

        // Perturb -ε.
        h_params(p) = orig - FD_EPS;
        Kokkos::deep_copy(params, h_params);

        view_1d_complex gp_minus("gp-", n_ant);
        view_1d_complex gq_minus("gq-", n_ant);
        {
            auto hgp = Kokkos::create_mirror_view(gp_minus);
            auto hgq = Kokkos::create_mirror_view(gq_minus);
            hgp(0) = complex_type(1.0, 0.0);
            hgq(0) = complex_type(1.0, 0.0);
            for (int a = 1; a < n_ant; ++a) {
                int off = amap.h_ant_to_param(a);
                complex_type g_p_val, g_q_val;
                double pp[4] = {h_params(off), h_params(off+1), h_params(off+2), h_params(off+3)};
                GainParam::params_to_diagonal(pp, g_p_val, g_q_val);
                hgp(a) = g_p_val;
                hgq(a) = g_q_val;
            }
            Kokkos::deep_copy(gp_minus, hgp);
            Kokkos::deep_copy(gq_minus, hgq);
        }

        view_1d_real r_minus("r-", n_res);
        build_residual_diagonal(r_minus, gp_minus, gq_minus, vis_obs, vis_model, ant1, ant2, n_bl);
        auto h_rm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r_minus);

        // Restore param.
        h_params(p) = orig;
        Kokkos::deep_copy(params, h_params);

        // Compare fd vs analytic for each residual row.
        for (int row = 0; row < n_res; ++row) {
            double fd = (h_rp(row) - h_rm(row)) / (2.0 * FD_EPS);
            double analytic = get_J_entry(Jac, row, p);
            bool ok = fd_close(analytic, fd);
            if (!ok) {
                printf("  FAIL: G Jac[%d,%d] analytic=%.8e fd=%.8e\n", row, p, analytic, fd);
                ++n_fail;
            } else {
                ++n_checked;
            }
        }
    }
    char buf[128];
    snprintf(buf, sizeof(buf), "G Jacobian: %d entries match finite diff", n_checked);
    check(n_checked > 0, buf);
}

// --------------------------------------------------------------------------
// Test 2: Delay solver Jacobian
// --------------------------------------------------------------------------
static void test_delay_jacobian() {
    printf("\n--- Jacobian finite diff: Delay (K) ---\n");

    const int n_ant = 3;
    const int n_bl = 3;
    const int ref_ant = 0;
    const int n_freq = 2;
    const int ppa = DelayParam::params_per_ant;  // 2

    AntennaMap amap = build_antenna_map(n_ant, ref_ant, ppa);

    host_view_1d_int h_ant1("h_ant1", n_bl);
    host_view_1d_int h_ant2("h_ant2", n_bl);
    h_ant1(0) = 0; h_ant2(0) = 1;
    h_ant1(1) = 0; h_ant2(1) = 2;
    h_ant1(2) = 1; h_ant2(2) = 2;
    view_1d_int ant1("ant1", n_bl);
    view_1d_int ant2("ant2", n_bl);
    Kokkos::deep_copy(ant1, h_ant1);
    Kokkos::deep_copy(ant2, h_ant2);

    // Frequencies.
    view_1d_real freqs("freqs", n_freq);
    {
        auto h = Kokkos::create_mirror_view(freqs);
        h(0) = 1.0e9;  // 1 GHz
        h(1) = 1.5e9;  // 1.5 GHz
        Kokkos::deep_copy(freqs, h);
    }

    // Model vis: (n_bl*n_freq, 4).
    view_2d_complex vis_model("vis_model", n_bl * n_freq, 4);
    {
        auto h = Kokkos::create_mirror_view(vis_model);
        for (int i = 0; i < n_bl * n_freq; ++i) {
            h(i, 0) = complex_type(1.0, 0.2 * (i % 3));
            h(i, 1) = complex_type(0.0, 0.0);
            h(i, 2) = complex_type(0.0, 0.0);
            h(i, 3) = complex_type(0.8, -0.1 * (i % 2));
        }
        Kokkos::deep_copy(vis_model, h);
    }
    view_2d_complex vis_obs("vis_obs", n_bl * n_freq, 4);
    Kokkos::deep_copy(vis_obs, vis_model);

    // Params: delays in ns. ant1: [2.5, -1.0], ant2: [0.5, 1.5]
    const int n_params = amap.n_params;
    view_1d_real params("params", n_params);
    {
        auto h = Kokkos::create_mirror_view(params);
        h(0) = 2.5; h(1) = -1.0;
        h(2) = 0.5; h(3) = 1.5;
        Kokkos::deep_copy(params, h);
    }

    crs_matrix_type Jac = build_csr_diagonal(h_ant1, h_ant2, amap, n_freq);
    fill_jacobian_delay(Jac, params, vis_model, ant1, ant2, amap.ant_to_param, freqs, n_bl, n_freq, ppa);

    const int n_res = n_bl * n_freq * 4;
    auto h_params = Kokkos::create_mirror_view(params);
    Kokkos::deep_copy(h_params, params);
    auto h_freqs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), freqs);

    int n_checked = 0;
    for (int p = 0; p < n_params; ++p) {
        double orig = h_params(p);

        // +ε
        h_params(p) = orig + FD_EPS;
        Kokkos::deep_copy(params, h_params);
        view_2d_complex gp_plus("gp+", n_ant, n_freq);
        view_2d_complex gq_plus("gq+", n_ant, n_freq);
        {
            auto hgp = Kokkos::create_mirror_view(gp_plus);
            auto hgq = Kokkos::create_mirror_view(gq_plus);
            for (int a = 0; a < n_ant; ++a) {
                for (int f = 0; f < n_freq; ++f) {
                    if (a == ref_ant) {
                        hgp(a, f) = complex_type(1.0, 0.0);
                        hgq(a, f) = complex_type(1.0, 0.0);
                    } else {
                        int off = amap.h_ant_to_param(a);
                        double pp[2] = {h_params(off), h_params(off+1)};
                        complex_type gp_val, gq_val;
                        DelayParam::params_to_diagonal(pp, h_freqs(f), gp_val, gq_val);
                        hgp(a, f) = gp_val;
                        hgq(a, f) = gq_val;
                    }
                }
            }
            Kokkos::deep_copy(gp_plus, hgp);
            Kokkos::deep_copy(gq_plus, hgq);
        }
        view_1d_real r_plus("r+", n_res);
        build_residual_diagonal_freq(r_plus, gp_plus, gq_plus, vis_obs, vis_model, ant1, ant2, n_bl, n_freq);
        auto h_rp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r_plus);

        // -ε
        h_params(p) = orig - FD_EPS;
        Kokkos::deep_copy(params, h_params);
        view_2d_complex gp_minus("gp-", n_ant, n_freq);
        view_2d_complex gq_minus("gq-", n_ant, n_freq);
        {
            auto hgp = Kokkos::create_mirror_view(gp_minus);
            auto hgq = Kokkos::create_mirror_view(gq_minus);
            for (int a = 0; a < n_ant; ++a) {
                for (int f = 0; f < n_freq; ++f) {
                    if (a == ref_ant) {
                        hgp(a, f) = complex_type(1.0, 0.0);
                        hgq(a, f) = complex_type(1.0, 0.0);
                    } else {
                        int off = amap.h_ant_to_param(a);
                        double pp[2] = {h_params(off), h_params(off+1)};
                        complex_type gp_val, gq_val;
                        DelayParam::params_to_diagonal(pp, h_freqs(f), gp_val, gq_val);
                        hgp(a, f) = gp_val;
                        hgq(a, f) = gq_val;
                    }
                }
            }
            Kokkos::deep_copy(gp_minus, hgp);
            Kokkos::deep_copy(gq_minus, hgq);
        }
        view_1d_real r_minus("r-", n_res);
        build_residual_diagonal_freq(r_minus, gp_minus, gq_minus, vis_obs, vis_model, ant1, ant2, n_bl, n_freq);
        auto h_rm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r_minus);

        h_params(p) = orig;
        Kokkos::deep_copy(params, h_params);

        for (int row = 0; row < n_res; ++row) {
            double fd = (h_rp(row) - h_rm(row)) / (2.0 * FD_EPS);
            double analytic = get_J_entry(Jac, row, p);
            bool ok = fd_close(analytic, fd);
            if (!ok) {
                printf("  FAIL: K Jac[%d,%d] analytic=%.8e fd=%.8e\n", row, p, analytic, fd);
                ++n_fail;
            } else {
                ++n_checked;
            }
        }
    }
    char buf[128];
    snprintf(buf, sizeof(buf), "K Jacobian: %d entries match finite diff", n_checked);
    check(n_checked > 0, buf);
}

// --------------------------------------------------------------------------
// Test 3: Leakage solver Jacobian
// --------------------------------------------------------------------------
static void test_leakage_jacobian() {
    printf("\n--- Jacobian finite diff: Leakage (D) ---\n");

    const int n_ant = 3;
    const int n_bl = 3;
    const int ref_ant = 0;
    const int ppa = LeakageParam::params_per_ant;  // 4

    AntennaMap amap = build_antenna_map(n_ant, ref_ant, ppa);

    host_view_1d_int h_ant1("h_ant1", n_bl);
    host_view_1d_int h_ant2("h_ant2", n_bl);
    h_ant1(0) = 0; h_ant2(0) = 1;
    h_ant1(1) = 0; h_ant2(1) = 2;
    h_ant1(2) = 1; h_ant2(2) = 2;
    view_1d_int ant1("ant1", n_bl);
    view_1d_int ant2("ant2", n_bl);
    Kokkos::deep_copy(ant1, h_ant1);
    Kokkos::deep_copy(ant2, h_ant2);

    // Model vis: full 2x2 with cross-hand terms.
    view_2d_complex vis_model("vis_model", n_bl, 4);
    {
        auto h = Kokkos::create_mirror_view(vis_model);
        h(0, 0) = complex_type(1.0, 0.1);
        h(0, 1) = complex_type(0.05, -0.02);
        h(0, 2) = complex_type(-0.03, 0.04);
        h(0, 3) = complex_type(0.9, -0.1);
        h(1, 0) = complex_type(0.8, 0.2);
        h(1, 1) = complex_type(0.02, 0.01);
        h(1, 2) = complex_type(0.01, -0.03);
        h(1, 3) = complex_type(1.1, 0.15);
        h(2, 0) = complex_type(0.7, -0.15);
        h(2, 1) = complex_type(-0.01, 0.02);
        h(2, 2) = complex_type(0.03, 0.01);
        h(2, 3) = complex_type(0.95, 0.05);
        Kokkos::deep_copy(vis_model, h);
    }
    view_2d_complex vis_obs("vis_obs", n_bl, 4);
    Kokkos::deep_copy(vis_obs, vis_model);

    // Leakage params: small off-diagonal.
    // ant1: [Re(dpq), Im(dpq), Re(dqp), Im(dqp)] = [0.05, 0.02, -0.03, 0.01]
    // ant2: [0.04, -0.01, 0.02, 0.03]
    const int n_params = amap.n_params;
    view_1d_real params("params", n_params);
    {
        auto h = Kokkos::create_mirror_view(params);
        h(0) = 0.05; h(1) = 0.02; h(2) = -0.03; h(3) = 0.01;
        h(4) = 0.04; h(5) = -0.01; h(6) = 0.02; h(7) = 0.03;
        Kokkos::deep_copy(params, h);
    }

    crs_matrix_type Jac = build_csr_full_2x2(h_ant1, h_ant2, amap);
    fill_jacobian_leakage(Jac, params, vis_model, ant1, ant2, amap.ant_to_param, n_bl, ppa);

    const int n_res = n_bl * 8;
    auto h_params = Kokkos::create_mirror_view(params);
    Kokkos::deep_copy(h_params, params);

    int n_checked = 0;
    for (int p = 0; p < n_params; ++p) {
        double orig = h_params(p);

        // Helper lambda: build Jones from params and compute residual.
        auto compute_residual = [&](view_1d_real& r_out) {
            view_1d_mat2 J_current("J", n_ant);
            auto hJ = Kokkos::create_mirror_view(J_current);
            // ref_ant = 0 → identity
            hJ(0) = Mat2::identity();
            for (int a = 1; a < n_ant; ++a) {
                int off = amap.h_ant_to_param(a);
                double pp[4] = {h_params(off), h_params(off+1), h_params(off+2), h_params(off+3)};
                hJ(a) = LeakageParam::params_to_jones(pp);
            }
            Kokkos::deep_copy(J_current, hJ);
            build_residual_full_2x2(r_out, J_current, vis_obs, vis_model, ant1, ant2, n_bl);
        };

        h_params(p) = orig + FD_EPS;
        Kokkos::deep_copy(params, h_params);
        view_1d_real r_plus("r+", n_res);
        compute_residual(r_plus);
        auto h_rp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r_plus);

        h_params(p) = orig - FD_EPS;
        Kokkos::deep_copy(params, h_params);
        view_1d_real r_minus("r-", n_res);
        compute_residual(r_minus);
        auto h_rm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r_minus);

        h_params(p) = orig;
        Kokkos::deep_copy(params, h_params);

        for (int row = 0; row < n_res; ++row) {
            double fd = (h_rp(row) - h_rm(row)) / (2.0 * FD_EPS);
            double analytic = get_J_entry(Jac, row, p);
            bool ok = fd_close(analytic, fd);
            if (!ok) {
                printf("  FAIL: D Jac[%d,%d] analytic=%.8e fd=%.8e\n", row, p, analytic, fd);
                ++n_fail;
            } else {
                ++n_checked;
            }
        }
    }
    char buf[128];
    snprintf(buf, sizeof(buf), "D Jacobian: %d entries match finite diff", n_checked);
    check(n_checked > 0, buf);
}

// --------------------------------------------------------------------------
// Test 4: Cross-delay Jacobian
// --------------------------------------------------------------------------
static void test_cross_delay_jacobian() {
    printf("\n--- Jacobian finite diff: Cross-delay (KC) ---\n");

    const int n_bl = 2;
    const int n_freq = 2;

    view_1d_int ant1("ant1", n_bl);
    view_1d_int ant2("ant2", n_bl);
    {
        auto h1 = Kokkos::create_mirror_view(ant1);
        auto h2 = Kokkos::create_mirror_view(ant2);
        h1(0) = 0; h2(0) = 1;
        h1(1) = 0; h2(1) = 2;
        Kokkos::deep_copy(ant1, h1);
        Kokkos::deep_copy(ant2, h2);
    }

    view_1d_real freqs("freqs", n_freq);
    {
        auto h = Kokkos::create_mirror_view(freqs);
        h(0) = 1.0e9; h(1) = 1.5e9;
        Kokkos::deep_copy(freqs, h);
    }
    auto h_freqs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), freqs);

    // Model vis with cross-hand terms.
    view_2d_complex vis_model("vis_model", n_bl * n_freq, 4);
    {
        auto h = Kokkos::create_mirror_view(vis_model);
        for (int i = 0; i < n_bl * n_freq; ++i) {
            h(i, 0) = complex_type(1.0, 0.0);
            h(i, 1) = complex_type(0.3, 0.1);
            h(i, 2) = complex_type(0.2, -0.15);
            h(i, 3) = complex_type(0.9, 0.0);
        }
        Kokkos::deep_copy(vis_model, h);
    }
    view_2d_complex vis_obs("vis_obs", n_bl * n_freq, 4);
    Kokkos::deep_copy(vis_obs, vis_model);

    // Global param: tau = 1.5 ns.
    view_1d_real params("params", 1);
    {
        auto h = Kokkos::create_mirror_view(params);
        h(0) = 1.5;
        Kokkos::deep_copy(params, h);
    }

    const int n_res = n_bl * n_freq * 4;
    crs_matrix_type Jac = build_csr_global(n_res, 1);
    fill_jacobian_cross_delay(Jac, params, vis_model, ant1, ant2, freqs, n_bl, n_freq);

    auto h_params = Kokkos::create_mirror_view(params);
    Kokkos::deep_copy(h_params, params);

    // Compute FD for the single param.
    double orig = h_params(0);

    auto compute_cross_delay_residual = [&](view_1d_real& r_out) {
        // Build g_p, g_q per baseline*freq using cross-delay param.
        view_2d_complex gp("gp", n_bl, n_freq);
        view_2d_complex gq("gq", n_bl, n_freq);
        // For cross-delay, gains are NOT per-antenna — they apply uniformly.
        // We use a 3-antenna setup but the gains apply to every antenna identically.
        // Actually, cross-delay is a single global Jones applied to all baselines:
        // V_obs_pq = g_p * V_M_pq * conj(g_q) = exp(-2πi τ ν) * V_M_pq
        // V_obs_qp = g_q * V_M_qp * conj(g_p) = V_M_qp * conj(exp(-2πi τ ν))
        // So g_p = exp(-2πi τ ν), g_q = 1 for ALL antennas.
        // The residual uses pq and qp columns, producing 4 reals per bl*freq.

        // For the FD test, compute residuals directly.
        auto h_r = Kokkos::create_mirror_view(r_out);
        auto h_vis = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vis_model);
        auto h_vobs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vis_obs);
        double tau = h_params(0);

        for (int b = 0; b < n_bl; ++b) {
            for (int f = 0; f < n_freq; ++f) {
                int vis_row = b * n_freq + f;
                double freq = h_freqs(f);
                complex_type gp_val, gq_val;
                CrossDelayParam::param_to_diagonal(tau, freq, gp_val, gq_val);

                complex_type mpq = h_vis(vis_row, 1);
                complex_type mqp = h_vis(vis_row, 2);
                complex_type opq = h_vobs(vis_row, 1);
                complex_type oqp = h_vobs(vis_row, 2);

                complex_type pred_pq = gp_val * mpq * Kokkos::conj(gq_val);
                complex_type pred_qp = gq_val * mqp * Kokkos::conj(gp_val);

                complex_type diff_pq = opq - pred_pq;
                complex_type diff_qp = oqp - pred_qp;

                int base = (b * n_freq + f) * 4;
                h_r(base + 0) = diff_pq.real();
                h_r(base + 1) = diff_pq.imag();
                h_r(base + 2) = diff_qp.real();
                h_r(base + 3) = diff_qp.imag();
            }
        }
        Kokkos::deep_copy(r_out, h_r);
    };

    h_params(0) = orig + FD_EPS;
    Kokkos::deep_copy(params, h_params);
    view_1d_real r_plus("r+", n_res);
    compute_cross_delay_residual(r_plus);
    auto h_rp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r_plus);

    h_params(0) = orig - FD_EPS;
    Kokkos::deep_copy(params, h_params);
    view_1d_real r_minus("r-", n_res);
    compute_cross_delay_residual(r_minus);
    auto h_rm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r_minus);

    h_params(0) = orig;
    Kokkos::deep_copy(params, h_params);

    int n_checked = 0;
    for (int row = 0; row < n_res; ++row) {
        double fd = (h_rp(row) - h_rm(row)) / (2.0 * FD_EPS);
        double analytic = get_J_entry(Jac, row, 0);
        bool ok = fd_close(analytic, fd);
        if (!ok) {
            printf("  FAIL: KC Jac[%d,0] analytic=%.8e fd=%.8e\n", row, analytic, fd);
            ++n_fail;
        } else {
            ++n_checked;
        }
    }
    char buf[128];
    snprintf(buf, sizeof(buf), "KC Jacobian: %d entries match finite diff", n_checked);
    check(n_checked > 0, buf);
}

// --------------------------------------------------------------------------
// Test 5: Cross-phase Jacobian
// --------------------------------------------------------------------------
static void test_cross_phase_jacobian() {
    printf("\n--- Jacobian finite diff: Cross-phase (CP) ---\n");

    const int n_bl = 2;

    view_1d_int ant1("ant1", n_bl);
    view_1d_int ant2("ant2", n_bl);
    {
        auto h1 = Kokkos::create_mirror_view(ant1);
        auto h2 = Kokkos::create_mirror_view(ant2);
        h1(0) = 0; h2(0) = 1;
        h1(1) = 0; h2(1) = 2;
        Kokkos::deep_copy(ant1, h1);
        Kokkos::deep_copy(ant2, h2);
    }

    view_2d_complex vis_model("vis_model", n_bl, 4);
    {
        auto h = Kokkos::create_mirror_view(vis_model);
        for (int b = 0; b < n_bl; ++b) {
            h(b, 0) = complex_type(1.0, 0.0);
            h(b, 1) = complex_type(0.25, 0.1);
            h(b, 2) = complex_type(0.15, -0.2);
            h(b, 3) = complex_type(0.9, 0.0);
        }
        Kokkos::deep_copy(vis_model, h);
    }
    view_2d_complex vis_obs("vis_obs", n_bl, 4);
    Kokkos::deep_copy(vis_obs, vis_model);

    view_1d_real params("params", 1);
    {
        auto h = Kokkos::create_mirror_view(params);
        h(0) = 0.3;  // phi in radians
        Kokkos::deep_copy(params, h);
    }

    const int n_res = n_bl * 4;
    crs_matrix_type Jac = build_csr_global(n_res, 1);
    fill_jacobian_cross_phase(Jac, params, vis_model, ant1, ant2, n_bl);

    auto h_params = Kokkos::create_mirror_view(params);
    Kokkos::deep_copy(h_params, params);

    double orig = h_params(0);

    auto compute_cross_phase_residual = [&](view_1d_real& r_out) {
        auto h_r = Kokkos::create_mirror_view(r_out);
        auto h_vis = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vis_model);
        auto h_vobs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vis_obs);
        double phi = h_params(0);

        for (int b = 0; b < n_bl; ++b) {
            complex_type gp_val, gq_val;
            CrossPhaseParam::param_to_diagonal(phi, gp_val, gq_val);

            complex_type mpq = h_vis(b, 1);
            complex_type mqp = h_vis(b, 2);
            complex_type opq = h_vobs(b, 1);
            complex_type oqp = h_vobs(b, 2);

            complex_type pred_pq = gp_val * mpq * Kokkos::conj(gq_val);
            complex_type pred_qp = gq_val * mqp * Kokkos::conj(gp_val);

            complex_type diff_pq = opq - pred_pq;
            complex_type diff_qp = oqp - pred_qp;

            h_r(b * 4 + 0) = diff_pq.real();
            h_r(b * 4 + 1) = diff_pq.imag();
            h_r(b * 4 + 2) = diff_qp.real();
            h_r(b * 4 + 3) = diff_qp.imag();
        }
        Kokkos::deep_copy(r_out, h_r);
    };

    h_params(0) = orig + FD_EPS;
    Kokkos::deep_copy(params, h_params);
    view_1d_real r_plus("r+", n_res);
    compute_cross_phase_residual(r_plus);
    auto h_rp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r_plus);

    h_params(0) = orig - FD_EPS;
    Kokkos::deep_copy(params, h_params);
    view_1d_real r_minus("r-", n_res);
    compute_cross_phase_residual(r_minus);
    auto h_rm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r_minus);

    h_params(0) = orig;
    Kokkos::deep_copy(params, h_params);

    int n_checked = 0;
    for (int row = 0; row < n_res; ++row) {
        double fd = (h_rp(row) - h_rm(row)) / (2.0 * FD_EPS);
        double analytic = get_J_entry(Jac, row, 0);
        bool ok = fd_close(analytic, fd);
        if (!ok) {
            printf("  FAIL: CP Jac[%d,0] analytic=%.8e fd=%.8e\n", row, analytic, fd);
            ++n_fail;
        } else {
            ++n_checked;
        }
    }
    char buf[128];
    snprintf(buf, sizeof(buf), "CP Jacobian: %d entries match finite diff", n_checked);
    check(n_checked > 0, buf);
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        printf("=== test_jacobian (finite difference validation) ===\n");
        test_gain_jacobian();
        test_delay_jacobian();
        test_leakage_jacobian();
        test_cross_delay_jacobian();
        test_cross_phase_jacobian();
        printf("\n%d passed, %d failed\n", n_pass, n_fail);
    }
    Kokkos::finalize();
    return n_fail > 0 ? 1 : 0;
}
