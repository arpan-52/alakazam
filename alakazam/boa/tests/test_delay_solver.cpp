// test_delay_solver.cpp — End-to-end K (delay) solver test.
//
// Synthesize freq-dependent observations from known true delays,
// then verify recovery. Uses 4 antennas, 6 baselines, 8 frequency channels.
// True delays: tau_p, tau_q in ns per antenna (ref_ant=0 fixed at 0).
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
        printf("=== test_delay_solver ===\n");

        const int n_ant = 4, ref_ant = 0;
        const int n_bl  = n_ant * (n_ant - 1) / 2;  // 6
        const int n_freq = 8;

        view_1d_int ant1("ant1", n_bl), ant2("ant2", n_bl);
        {
            auto h_ant1 = Kokkos::create_mirror_view(ant1);
            auto h_ant2 = Kokkos::create_mirror_view(ant2);
            int b = 0;
            for (int i = 0; i < n_ant; ++i)
                for (int j = i+1; j < n_ant; ++j) { h_ant1(b)=i; h_ant2(b)=j; ++b; }
            Kokkos::deep_copy(ant1, h_ant1);
            Kokkos::deep_copy(ant2, h_ant2);
        }

        // True delays in ns: ref_ant=0 has 0 delay.
        // Values < 0.5 ns to stay within the 1/f_center = 1 ns ambiguity period.
        real_type true_tau_p[4] = {0.0, 0.30, -0.18, 0.42};
        real_type true_tau_q[4] = {0.0, -0.15, 0.22, -0.35};

        view_1d_real freqs("freqs", n_freq);
        {
            auto h_freqs = Kokkos::create_mirror_view(freqs);
            for (int f = 0; f < n_freq; ++f)
                h_freqs(f) = 1.0e9 + f * 1.0e6;
            Kokkos::deep_copy(freqs, h_freqs);
        }

        const real_type two_pi = 2.0 * 3.14159265358979323846;

        auto make_gdelay = [&](real_type tau_ns, real_type freq_hz) -> complex_type {
            real_type phase = -two_pi * tau_ns * 1.0e-9 * freq_hz;
            return complex_type(std::cos(phase), std::sin(phase));
        };

        const int n_vis = n_bl * n_freq;
        view_2d_complex vis_model("vm", n_vis, 4);
        view_2d_complex vis_obs("vo",   n_vis, 4);
        {
            auto h_vm  = Kokkos::create_mirror_view(vis_model);
            auto h_vo  = Kokkos::create_mirror_view(vis_obs);
            auto h_a1  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ant1);
            auto h_a2  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ant2);
            auto h_fr  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), freqs);

            for (int b2 = 0; b2 < n_bl; ++b2) {
                const int ai = h_a1(b2), aj = h_a2(b2);
                for (int f = 0; f < n_freq; ++f) {
                    const int row = b2 * n_freq + f;
                    h_vm(row, 0) = complex_type(1.0, 0.0);
                    h_vm(row, 1) = complex_type(0.0, 0.0);
                    h_vm(row, 2) = complex_type(0.0, 0.0);
                    h_vm(row, 3) = complex_type(1.0, 0.0);

                    complex_type gp_i = make_gdelay(true_tau_p[ai], h_fr(f));
                    complex_type gp_j = make_gdelay(true_tau_p[aj], h_fr(f));
                    complex_type gq_i = make_gdelay(true_tau_q[ai], h_fr(f));
                    complex_type gq_j = make_gdelay(true_tau_q[aj], h_fr(f));

                    h_vo(row, 0) = gp_i * h_vm(row, 0) * Kokkos::conj(gp_j);
                    h_vo(row, 3) = gq_i * h_vm(row, 3) * Kokkos::conj(gq_j);
                    h_vo(row, 1) = complex_type(0.0, 0.0);
                    h_vo(row, 2) = complex_type(0.0, 0.0);
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
        opts.max_iter      = 300;
        opts.tol           = 1.0e-12;
        opts.linear_solver = LinearSolverType::DENSE_CHOLESKY;

        SolverResult res = solve_K(inp, opts);

        printf("\n  K solver: cost=%.3e, iters=%d, converged=%s\n",
               res.cost, res.n_iter, res.converged ? "yes" : "no");

        check(res.converged, "K solver converged");
        check(res.cost < 1.0e-18, "K solver cost < 1e-18");

        auto h_params = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res.params);

        // Verify recovered delays (params layout: [tau_p, tau_q] per non-ref antenna).
        // Antenna order (excluding ref=0): 1, 2, 3.
        bool delays_ok = true;
        const int free_ants[3] = {1, 2, 3};
        for (int k = 0; k < 3; ++k) {
            int a = free_ants[k];
            real_type tau_p_got = h_params(k * 2 + 0);
            real_type tau_q_got = h_params(k * 2 + 1);
            if (std::abs(tau_p_got - true_tau_p[a]) > 1.0e-6) {
                printf("    ant %d: tau_p_solved=%.6f, tau_p_true=%.6f\n",
                       a, tau_p_got, true_tau_p[a]);
                delays_ok = false;
            }
            if (std::abs(tau_q_got - true_tau_q[a]) > 1.0e-6) {
                printf("    ant %d: tau_q_solved=%.6f, tau_q_true=%.6f\n",
                       a, tau_q_got, true_tau_q[a]);
                delays_ok = false;
            }
        }
        check(delays_ok, "K solver recovers true delays for all antennas");

        printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    }
    Kokkos::finalize();
    return n_fail == 0 ? 0 : 1;
}
