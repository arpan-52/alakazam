// test_cross_phase_solver.cpp — End-to-end CP (cross-phase) solver test.
//
// Synthesize cross-hand observations from a known phi_cross, verify recovery.
// J = diag(1, exp(i*phi)) — one global parameter.
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
        printf("=== test_cross_phase_solver ===\n");

        const int n_ant = 4, ref_ant = 0;
        const int n_bl  = n_ant * (n_ant - 1) / 2;

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

        const real_type true_phi = 0.7;  // 0.7 radians
        const complex_type gq(std::cos(true_phi), std::sin(true_phi));

        view_2d_complex vis_model("vm", n_bl, 4);
        view_2d_complex vis_obs("vo",   n_bl, 4);
        {
            auto h_vm = Kokkos::create_mirror_view(vis_model);
            auto h_vo = Kokkos::create_mirror_view(vis_obs);

            for (int b2 = 0; b2 < n_bl; ++b2) {
                h_vm(b2, 0) = complex_type(2.0, 0.0);
                h_vm(b2, 1) = complex_type(0.5, 0.3);
                h_vm(b2, 2) = complex_type(0.5, -0.3);
                h_vm(b2, 3) = complex_type(2.0, 0.0);

                h_vo(b2, 0) = h_vm(b2, 0);
                h_vo(b2, 1) = h_vm(b2, 1) * Kokkos::conj(gq);
                h_vo(b2, 2) = gq * h_vm(b2, 2);
                h_vo(b2, 3) = h_vm(b2, 3);
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
        opts.max_iter      = 100;
        opts.tol           = 1.0e-12;
        opts.linear_solver = LinearSolverType::DENSE_CHOLESKY;

        SolverResult res = solve_CP(inp, opts);

        auto h_params = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res.params);
        printf("\n  CP solver: phi_got=%.8f rad, phi_true=%.8f rad\n",
               h_params(0), true_phi);
        printf("  CP solver: cost=%.3e, iters=%d, converged=%s\n",
               res.cost, res.n_iter, res.converged ? "yes" : "no");

        check(res.converged, "CP solver converged");
        check(res.cost < 1.0e-22, "CP solver cost < 1e-22");
        check(std::abs(h_params(0) - true_phi) < 1.0e-8,
              "CP solver recovers true cross-phase phi");

        printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    }
    Kokkos::finalize();
    return n_fail == 0 ? 0 : 1;
}
