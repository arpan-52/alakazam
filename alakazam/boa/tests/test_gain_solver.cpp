// test_gain_solver.cpp — End-to-end G solver test.
//
// Synthesize observations from known true gains, verify recovery.
// 5 antennas, ref_ant=0, 10 baselines, noiseless.
// All 2D device views use create_mirror_view for host-side filling
// to avoid CUDA LayoutLeft vs LayoutRight mismatch.

#include <boa/types.hpp>
#include <boa/api.hpp>
#include <cstdio>
#include <cmath>

using namespace boa;

static int n_pass = 0, n_fail = 0;
static void check(bool c, const char* n) {
    if (c) { printf("  PASS: %s\n", n); ++n_pass; }
    else   { printf("  FAIL: %s\n", n); ++n_fail; }
}

int main() {
    Kokkos::initialize();
    {
        printf("=== test_gain_solver ===\n");

        const int n_ant = 5, ref_ant = 0;
        const int n_bl = n_ant * (n_ant - 1) / 2;  // 10

        view_1d_int ant1("ant1", n_bl), ant2("ant2", n_bl);
        {
            auto h1 = Kokkos::create_mirror_view(ant1);
            auto h2 = Kokkos::create_mirror_view(ant2);
            int b = 0;
            for (int i = 0; i < n_ant; ++i)
                for (int j = i+1; j < n_ant; ++j) { h1(b)=i; h2(b)=j; ++b; }
            Kokkos::deep_copy(ant1, h1);
            Kokkos::deep_copy(ant2, h2);
        }

        // True gains per antenna. ref_ant=0 fixed at identity.
        const real_type true_amp_p[5]   = {1.0, 1.10, 0.85, 1.20, 0.90};
        const real_type true_phase_p[5] = {0.0, 0.20, -0.15, 0.25, -0.10};
        const real_type true_amp_q[5]   = {1.0, 0.95, 1.15, 0.80, 1.05};
        const real_type true_phase_q[5] = {0.0, -0.10, 0.30, -0.20, 0.15};

        auto make_g = [](real_type amp, real_type phase) -> complex_type {
            return complex_type(amp * std::cos(phase), amp * std::sin(phase));
        };

        // Build device views and fill via host mirror.
        view_2d_complex vis_model("vm", n_bl, 4);
        view_2d_complex vis_obs("vo",   n_bl, 4);
        {
            auto h_vm = Kokkos::create_mirror_view(vis_model);
            auto h_vo = Kokkos::create_mirror_view(vis_obs);
            auto h1   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ant1);
            auto h2   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ant2);

            for (int b = 0; b < n_bl; ++b) {
                // Model: unit diagonal, no cross-hands.
                h_vm(b, 0) = complex_type(1.0, 0.0);
                h_vm(b, 1) = complex_type(0.0, 0.0);
                h_vm(b, 2) = complex_type(0.0, 0.0);
                h_vm(b, 3) = complex_type(1.0, 0.0);

                const int ai = h1(b), aj = h2(b);
                complex_type gp_i = make_g(true_amp_p[ai], true_phase_p[ai]);
                complex_type gp_j = make_g(true_amp_p[aj], true_phase_p[aj]);
                complex_type gq_i = make_g(true_amp_q[ai], true_phase_q[ai]);
                complex_type gq_j = make_g(true_amp_q[aj], true_phase_q[aj]);
                h_vo(b, 0) = gp_i * h_vm(b, 0) * Kokkos::conj(gp_j);
                h_vo(b, 3) = gq_i * h_vm(b, 3) * Kokkos::conj(gq_j);
                h_vo(b, 1) = h_vo(b, 2) = complex_type(0.0, 0.0);
            }
            Kokkos::deep_copy(vis_model, h_vm);
            Kokkos::deep_copy(vis_obs,   h_vo);
        }

        SolverInput inp;
        inp.vis_obs   = vis_obs;
        inp.vis_model = vis_model;
        inp.ant1      = ant1;
        inp.ant2      = ant2;
        inp.freqs     = view_1d_real("freqs", 0);
        inp.n_ant     = n_ant;
        inp.ref_ant   = ref_ant;

        SolverOptions opts;
        opts.max_iter      = 200;
        opts.tol           = 1.0e-12;
        opts.linear_solver = LinearSolverType::DENSE_CHOLESKY;

        SolverResult res = solve_G(inp, opts);

        printf("\n  G solver: cost=%.3e, iters=%d, converged=%s\n",
               res.cost, res.n_iter, res.converged ? "yes" : "no");

        check(res.converged, "G solver converged");
        check(res.cost < 1.0e-20, "G solver cost < 1e-20");

        auto h_jones = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res.jones);
        bool jones_ok = true;
        for (int a = 1; a < n_ant; ++a) {
            complex_type gp_true = make_g(true_amp_p[a], true_phase_p[a]);
            complex_type gq_true = make_g(true_amp_q[a], true_phase_q[a]);
            if (Kokkos::abs(h_jones(a, 0) - gp_true) > 1.0e-8) {
                printf("    ant %d: gp=(%.6f,%.6fi) true=(%.6f,%.6fi)\n",
                       a, h_jones(a,0).real(), h_jones(a,0).imag(),
                       gp_true.real(), gp_true.imag());
                jones_ok = false;
            }
            if (Kokkos::abs(h_jones(a, 3) - gq_true) > 1.0e-8) {
                printf("    ant %d: gq=(%.6f,%.6fi) true=(%.6f,%.6fi)\n",
                       a, h_jones(a,3).real(), h_jones(a,3).imag(),
                       gq_true.real(), gq_true.imag());
                jones_ok = false;
            }
        }
        check(jones_ok, "G solver recovers true gains for all antennas");
        check(Kokkos::abs(h_jones(ref_ant, 0) - complex_type(1.0, 0.0)) < 1.0e-14 &&
              Kokkos::abs(h_jones(ref_ant, 3) - complex_type(1.0, 0.0)) < 1.0e-14,
              "G solver: ref_ant jones is identity");

        printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    }
    Kokkos::finalize();
    return n_fail == 0 ? 0 : 1;
}
