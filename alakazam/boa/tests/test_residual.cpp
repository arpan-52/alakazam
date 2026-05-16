// test_residual.cpp — Verify residual kernels against hand-computed values.
//
// For a small problem (3 antennas, 3 baselines), we set known Jones gains
// and model visibilities, compute predicted visibilities by hand, and check
// that the residual vector matches.
//
// CUDA note: 2D views have different layouts on host (LayoutRight) vs device
// (LayoutLeft). We always create device views first, then create_mirror_view
// to get a host mirror with matching layout, fill it, and deep_copy back.

#include <boa/types.hpp>
#include <boa/residual.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace boa;

static int n_pass = 0;
static int n_fail = 0;

static void check(bool cond, const char* name) {
    if (cond) {
        printf("  PASS: %s\n", name);
        ++n_pass;
    } else {
        printf("  FAIL: %s\n", name);
        ++n_fail;
    }
}

static bool near(double a, double b, double tol = 1e-12) {
    return std::abs(a - b) < tol;
}

// --------------------------------------------------------------------------
// Test 1: Diagonal residual (freq-independent) — identity gains → zero residual
// --------------------------------------------------------------------------
static void test_diagonal_identity() {
    printf("\n--- diagonal residual: identity gains ---\n");

    const int n_ant = 3;
    const int n_bl = 3;

    view_1d_int ant1("ant1", n_bl);
    view_1d_int ant2("ant2", n_bl);
    {
        auto h1 = Kokkos::create_mirror_view(ant1);
        auto h2 = Kokkos::create_mirror_view(ant2);
        h1(0) = 0; h2(0) = 1;
        h1(1) = 0; h2(1) = 2;
        h1(2) = 1; h2(2) = 2;
        Kokkos::deep_copy(ant1, h1);
        Kokkos::deep_copy(ant2, h2);
    }

    view_2d_complex vis_obs("vis_obs", n_bl, 4);
    view_2d_complex vis_model("vis_model", n_bl, 4);
    {
        auto h = Kokkos::create_mirror_view(vis_obs);
        h(0, 0) = complex_type(1.0, 0.5);
        h(0, 1) = complex_type(0.0, 0.0);
        h(0, 2) = complex_type(0.0, 0.0);
        h(0, 3) = complex_type(0.8, -0.3);
        h(1, 0) = complex_type(0.7, 0.2);
        h(1, 1) = complex_type(0.0, 0.0);
        h(1, 2) = complex_type(0.0, 0.0);
        h(1, 3) = complex_type(0.9, 0.1);
        h(2, 0) = complex_type(0.5, -0.4);
        h(2, 1) = complex_type(0.0, 0.0);
        h(2, 2) = complex_type(0.0, 0.0);
        h(2, 3) = complex_type(0.6, 0.7);
        Kokkos::deep_copy(vis_obs, h);
        Kokkos::deep_copy(vis_model, h);
    }

    view_1d_complex g_p("g_p", n_ant);
    view_1d_complex g_q("g_q", n_ant);
    {
        auto hp = Kokkos::create_mirror_view(g_p);
        auto hq = Kokkos::create_mirror_view(g_q);
        for (int a = 0; a < n_ant; ++a) {
            hp(a) = complex_type(1.0, 0.0);
            hq(a) = complex_type(1.0, 0.0);
        }
        Kokkos::deep_copy(g_p, hp);
        Kokkos::deep_copy(g_q, hq);
    }

    view_1d_real r("r", n_bl * 4);
    build_residual_diagonal(r, g_p, g_q, vis_obs, vis_model, ant1, ant2, n_bl);

    auto h_r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r);
    for (int i = 0; i < n_bl * 4; ++i) {
        check(near(h_r(i), 0.0), "identity gains → zero residual");
    }
}

// --------------------------------------------------------------------------
// Test 2: Diagonal residual — known non-trivial gains
// --------------------------------------------------------------------------
static void test_diagonal_known_gains() {
    printf("\n--- diagonal residual: known gains ---\n");

    const int n_bl = 1;

    view_1d_int ant1("ant1", 1);
    view_1d_int ant2("ant2", 1);
    {
        auto h1 = Kokkos::create_mirror_view(ant1);
        auto h2 = Kokkos::create_mirror_view(ant2);
        h1(0) = 0; h2(0) = 1;
        Kokkos::deep_copy(ant1, h1);
        Kokkos::deep_copy(ant2, h2);
    }

    complex_type M_pp(2.0, 1.0), M_qq(1.0, -1.0);
    view_2d_complex vis_model("vis_model", 1, 4);
    {
        auto h = Kokkos::create_mirror_view(vis_model);
        h(0, 0) = M_pp;
        h(0, 1) = complex_type(0.0, 0.0);
        h(0, 2) = complex_type(0.0, 0.0);
        h(0, 3) = M_qq;
        Kokkos::deep_copy(vis_model, h);
    }

    complex_type g0p(1.5, 0.5), g1p(0.8, -0.2), g0q(1.0, 0.3), g1q(0.9, -0.1);
    view_1d_complex g_p("g_p", 2);
    view_1d_complex g_q("g_q", 2);
    {
        auto hp = Kokkos::create_mirror_view(g_p);
        auto hq = Kokkos::create_mirror_view(g_q);
        hp(0) = g0p; hp(1) = g1p;
        hq(0) = g0q; hq(1) = g1q;
        Kokkos::deep_copy(g_p, hp);
        Kokkos::deep_copy(g_q, hq);
    }

    complex_type pred_pp = g0p * M_pp * Kokkos::conj(g1p);
    complex_type pred_qq = g0q * M_qq * Kokkos::conj(g1q);

    complex_type obs_pp(3.0, 1.0), obs_qq(2.0, -0.5);
    view_2d_complex vis_obs("vis_obs", 1, 4);
    {
        auto h = Kokkos::create_mirror_view(vis_obs);
        h(0, 0) = obs_pp;
        h(0, 1) = complex_type(0.0, 0.0);
        h(0, 2) = complex_type(0.0, 0.0);
        h(0, 3) = obs_qq;
        Kokkos::deep_copy(vis_obs, h);
    }

    view_1d_real r("r", 4);
    build_residual_diagonal(r, g_p, g_q, vis_obs, vis_model, ant1, ant2, n_bl);

    auto h_r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r);

    complex_type diff_pp = obs_pp - pred_pp;
    complex_type diff_qq = obs_qq - pred_qq;

    check(near(h_r(0), diff_pp.real(), 1e-10), "known gains r[0] Re(pp)");
    check(near(h_r(1), diff_pp.imag(), 1e-10), "known gains r[1] Im(pp)");
    check(near(h_r(2), diff_qq.real(), 1e-10), "known gains r[2] Re(qq)");
    check(near(h_r(3), diff_qq.imag(), 1e-10), "known gains r[3] Im(qq)");
}

// --------------------------------------------------------------------------
// Test 3: Freq-dependent diagonal residual — identity → zero
// --------------------------------------------------------------------------
static void test_diagonal_freq() {
    printf("\n--- diagonal residual: freq-dependent ---\n");

    const int n_bl = 1;
    const int n_freq = 2;

    view_1d_int ant1("ant1", 1);
    view_1d_int ant2("ant2", 1);
    {
        auto h1 = Kokkos::create_mirror_view(ant1);
        auto h2 = Kokkos::create_mirror_view(ant2);
        h1(0) = 0; h2(0) = 1;
        Kokkos::deep_copy(ant1, h1);
        Kokkos::deep_copy(ant2, h2);
    }

    view_2d_complex vis_model("vis_model", 2, 4);
    {
        auto h = Kokkos::create_mirror_view(vis_model);
        h(0, 0) = complex_type(1.0, 0.0);
        h(0, 1) = complex_type(0.0, 0.0);
        h(0, 2) = complex_type(0.0, 0.0);
        h(0, 3) = complex_type(0.5, 0.0);
        h(1, 0) = complex_type(0.8, 0.2);
        h(1, 1) = complex_type(0.0, 0.0);
        h(1, 2) = complex_type(0.0, 0.0);
        h(1, 3) = complex_type(0.6, -0.1);
        Kokkos::deep_copy(vis_model, h);
    }

    view_2d_complex vis_obs("vis_obs", 2, 4);
    Kokkos::deep_copy(vis_obs, vis_model);

    view_2d_complex g_p("g_p", 2, 2);
    view_2d_complex g_q("g_q", 2, 2);
    {
        auto hp = Kokkos::create_mirror_view(g_p);
        auto hq = Kokkos::create_mirror_view(g_q);
        for (int a = 0; a < 2; ++a) {
            for (int f = 0; f < 2; ++f) {
                hp(a, f) = complex_type(1.0, 0.0);
                hq(a, f) = complex_type(1.0, 0.0);
            }
        }
        Kokkos::deep_copy(g_p, hp);
        Kokkos::deep_copy(g_q, hq);
    }

    view_1d_real r("r", n_bl * n_freq * 4);
    build_residual_diagonal_freq(r, g_p, g_q, vis_obs, vis_model, ant1, ant2, n_bl, n_freq);

    auto h_r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r);
    for (int i = 0; i < n_bl * n_freq * 4; ++i) {
        check(near(h_r(i), 0.0), "freq-dep identity → zero residual");
    }
}

// --------------------------------------------------------------------------
// Test 4: Full 2x2 residual — identity Jones → zero residual
// --------------------------------------------------------------------------
static void test_full_2x2_identity() {
    printf("\n--- full 2x2 residual: identity Jones ---\n");

    const int n_ant = 2;
    const int n_bl = 1;

    view_1d_int ant1("ant1", 1);
    view_1d_int ant2("ant2", 1);
    {
        auto h1 = Kokkos::create_mirror_view(ant1);
        auto h2 = Kokkos::create_mirror_view(ant2);
        h1(0) = 0; h2(0) = 1;
        Kokkos::deep_copy(ant1, h1);
        Kokkos::deep_copy(ant2, h2);
    }

    view_2d_complex vis_model("vis_model", 1, 4);
    view_2d_complex vis_obs("vis_obs", 1, 4);
    {
        auto h = Kokkos::create_mirror_view(vis_model);
        h(0, 0) = complex_type(1.0, 0.2);
        h(0, 1) = complex_type(0.1, -0.1);
        h(0, 2) = complex_type(-0.1, 0.05);
        h(0, 3) = complex_type(0.9, 0.3);
        Kokkos::deep_copy(vis_model, h);
        Kokkos::deep_copy(vis_obs, vis_model);
    }

    view_1d_mat2 J_current("J_current", n_ant);
    {
        auto hJ = Kokkos::create_mirror_view(J_current);
        hJ(0) = Mat2::identity();
        hJ(1) = Mat2::identity();
        Kokkos::deep_copy(J_current, hJ);
    }

    view_1d_real r("r", n_bl * 8);
    build_residual_full_2x2(r, J_current, vis_obs, vis_model, ant1, ant2, n_bl);

    auto h_r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r);
    for (int i = 0; i < 8; ++i) {
        check(near(h_r(i), 0.0), "full 2x2 identity → zero residual");
    }
}

// --------------------------------------------------------------------------
// Test 5: Full 2x2 residual — known non-identity Jones
// --------------------------------------------------------------------------
static void test_full_2x2_known() {
    printf("\n--- full 2x2 residual: known Jones ---\n");

    const int n_ant = 2;
    const int n_bl = 1;

    view_1d_int ant1("ant1", 1);
    view_1d_int ant2("ant2", 1);
    {
        auto h1 = Kokkos::create_mirror_view(ant1);
        auto h2 = Kokkos::create_mirror_view(ant2);
        h1(0) = 0; h2(0) = 1;
        Kokkos::deep_copy(ant1, h1);
        Kokkos::deep_copy(ant2, h2);
    }

    Mat2 VM;
    VM(0, 0) = complex_type(1.0, 0.0);
    VM(0, 1) = complex_type(0.0, 0.0);
    VM(1, 0) = complex_type(0.0, 0.0);
    VM(1, 1) = complex_type(1.0, 0.0);

    view_2d_complex vis_model("vis_model", 1, 4);
    {
        auto h = Kokkos::create_mirror_view(vis_model);
        for (int k = 0; k < 4; ++k) h(0, k) = VM.m[k];
        Kokkos::deep_copy(vis_model, h);
    }

    Mat2 J0, J1;
    J0(0, 0) = complex_type(1.0, 0.0);
    J0(0, 1) = complex_type(0.1, 0.05);
    J0(1, 0) = complex_type(0.02, -0.01);
    J0(1, 1) = complex_type(1.0, 0.0);
    J1 = Mat2::identity();

    Mat2 pred = mat2_multiply_hermitian(mat2_multiply(J0, VM), J1);

    view_2d_complex vis_obs("vis_obs", 1, 4);
    {
        auto h = Kokkos::create_mirror_view(vis_obs);
        for (int k = 0; k < 4; ++k) h(0, k) = complex_type(0.0, 0.0);
        Kokkos::deep_copy(vis_obs, h);
    }

    view_1d_mat2 J_current("J_current", n_ant);
    {
        auto hJ = Kokkos::create_mirror_view(J_current);
        hJ(0) = J0;
        hJ(1) = J1;
        Kokkos::deep_copy(J_current, hJ);
    }

    view_1d_real r("r", 8);
    build_residual_full_2x2(r, J_current, vis_obs, vis_model, ant1, ant2, n_bl);

    auto h_r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r);

    for (int rc = 0; rc < 4; ++rc) {
        complex_type neg_pred = complex_type(0.0, 0.0) - pred.m[rc];
        check(near(h_r(2 * rc + 0), neg_pred.real(), 1e-10),
              "full 2x2 known: Re component");
        check(near(h_r(2 * rc + 1), neg_pred.imag(), 1e-10),
              "full 2x2 known: Im component");
    }
}

// --------------------------------------------------------------------------
// Test 6: residual_norm_sq
// --------------------------------------------------------------------------
static void test_norm_sq() {
    printf("\n--- residual_norm_sq ---\n");

    const int n = 4;
    view_1d_real r("r", n);
    {
        auto h = Kokkos::create_mirror_view(r);
        h(0) = 1.0; h(1) = 2.0; h(2) = 3.0; h(3) = 4.0;
        Kokkos::deep_copy(r, h);
    }

    real_type norm_sq = residual_norm_sq(r);
    check(near(norm_sq, 30.0), "||r||^2 == 30");
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        printf("=== test_residual ===\n");
        test_diagonal_identity();
        test_diagonal_known_gains();
        test_diagonal_freq();
        test_full_2x2_identity();
        test_full_2x2_known();
        test_norm_sq();
        printf("\n%d passed, %d failed\n", n_pass, n_fail);
    }
    Kokkos::finalize();
    return n_fail > 0 ? 1 : 0;
}
