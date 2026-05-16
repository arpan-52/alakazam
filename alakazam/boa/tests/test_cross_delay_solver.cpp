// test_cross_delay_solver.cpp — End-to-end KC (cross-delay) solver test.
//
// Synthesize cross-hand observations from a known tau_cross, verify recovery.
// 4 antennas, 6 baselines, 8 frequency channels.
// J = diag(exp(-2πi τ ν), 1) — one global parameter.
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
        printf("=== test_cross_delay_solver ===\n");

        const int n_ant = 4, ref_ant = 0;
        const int n_bl  = n_ant * (n_ant - 1) / 2;
        const int n_freq = 16;

        view_1d_int ant1("ant1", n_bl), ant2("ant2", n_bl);
        {
            auto h_a1 = Kokkos::create_mirror_view(ant1);
            auto h_a2 = Kokkos::create_mirror_view(ant2);
            int b = 0;
            for (int i = 0; i < n_ant; ++i)
                for (int j = i+1; j < n_ant; ++j) { h_a1(b)=i; h_a2(b)=j; ++b; }
            Kokkos::deep_copy(ant1, h_a1);
            Kokkos::deep_copy(ant2, h_a2);
        }

        // < 0.5 ns to stay within the 1/f_center = 1 ns ambiguity period at 1 GHz.
        const real_type true_tau_ns = 0.25;
        const real_type two_pi = 2.0 * 3.14159265358979323846;

        view_1d_real freqs("freqs", n_freq);
        {
            auto h_fr = Kokkos::create_mirror_view(freqs);
            for (int f = 0; f < n_freq; ++f)
                h_fr(f) = 1.0e9 + f * 2.0e6;  // 2 MHz spacing
            Kokkos::deep_copy(freqs, h_fr);
        }

        const int n_vis = n_bl * n_freq;
        view_2d_complex vis_model("vm", n_vis, 4);
        view_2d_complex vis_obs("vo",   n_vis, 4);
        {
            auto h_vm = Kokkos::create_mirror_view(vis_model);
            auto h_vo = Kokkos::create_mirror_view(vis_obs);
            auto h_fr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), freqs);

            for (int b2 = 0; b2 < n_bl; ++b2) {
                for (int f = 0; f < n_freq; ++f) {
                    const int row = b2 * n_freq + f;
                    h_vm(row, 0) = complex_type(1.0, 0.0);
                    h_vm(row, 1) = complex_type(1.0, 0.0);
                    h_vm(row, 2) = complex_type(1.0, 0.0);
                    h_vm(row, 3) = complex_type(1.0, 0.0);

                    real_type phase = -two_pi * true_tau_ns * 1.0e-9 * h_fr(f);
                    complex_type gp(std::cos(phase), std::sin(phase));

                    h_vo(row, 0) = h_vm(row, 0);
                    h_vo(row, 1) = gp * h_vm(row, 1);
                    h_vo(row, 2) = h_vm(row, 2) * Kokkos::conj(gp);
                    h_vo(row, 3) = h_vm(row, 3);
                }
            }
            Kokkos::deep_copy(vis_model, h_vm);
            Kokkos::deep_copy(vis_obs,   h_vo);
        }

        SolverInput inp;
        inp.vis_obs   = vis_obs;
        inp.vis_model = vis_model;
        inp.ant1      = ant1;
        inp.ant2      = ant2;
        inp.freqs     = freqs;
        inp.n_ant     = n_ant;
        inp.ref_ant   = ref_ant;

        SolverOptions opts;
        opts.max_iter      = 200;
        opts.tol           = 1.0e-12;
        opts.linear_solver = LinearSolverType::DENSE_CHOLESKY;

        SolverResult res = solve_KC(inp, opts);

        auto h_params = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res.params);
        printf("\n  KC solver: tau_got=%.6f ns, tau_true=%.6f ns\n",
               h_params(0), true_tau_ns);
        printf("  KC solver: cost=%.3e, iters=%d, converged=%s\n",
               res.cost, res.n_iter, res.converged ? "yes" : "no");

        check(res.converged, "KC solver converged");
        check(res.cost < 1.0e-18, "KC solver cost < 1e-18");
        check(std::abs(h_params(0) - true_tau_ns) < 1.0e-6,
              "KC solver recovers true cross-delay tau");

        printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    }
    Kokkos::finalize();
    return n_fail == 0 ? 0 : 1;
}
