// test_leakage_solver.cpp — End-to-end D (leakage) solver test.
//
// Synthesize full-2x2 observations from known true leakage matrices,
// verify the solver recovers d_pq and d_qp per antenna.
// 4 antennas, ref_ant=0, 6 baselines.
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
        printf("=== test_leakage_solver ===\n");

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

        // True leakages (small values typical in radio astronomy).
        // ref_ant=0: d=0 (identity).
        complex_type true_dpq[4] = {
            complex_type(0.0, 0.0),
            complex_type(0.05, 0.02),
            complex_type(-0.03, 0.04),
            complex_type(0.07, -0.01)
        };
        complex_type true_dqp[4] = {
            complex_type(0.0, 0.0),
            complex_type(-0.02, 0.06),
            complex_type(0.04, 0.03),
            complex_type(-0.05, -0.02)
        };

        auto make_jones = [&](int a) -> Mat2 {
            Mat2 J;
            J(0,0) = complex_type(1.0, 0.0);
            J(0,1) = true_dpq[a];
            J(1,0) = true_dqp[a];
            J(1,1) = complex_type(1.0, 0.0);
            return J;
        };

        view_2d_complex vis_model("vm", n_bl, 4);
        view_2d_complex vis_obs("vo",   n_bl, 4);
        {
            auto h_vm = Kokkos::create_mirror_view(vis_model);
            auto h_vo = Kokkos::create_mirror_view(vis_obs);
            auto h_a1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ant1);
            auto h_a2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ant2);

            for (int b2 = 0; b2 < n_bl; ++b2) {
                h_vm(b2, 0) = complex_type(1.0, 0.0);
                h_vm(b2, 1) = complex_type(0.0, 0.0);
                h_vm(b2, 2) = complex_type(0.0, 0.0);
                h_vm(b2, 3) = complex_type(1.0, 0.0);
            }

            for (int b2 = 0; b2 < n_bl; ++b2) {
                const int ai = h_a1(b2), aj = h_a2(b2);
                Mat2 Ji = make_jones(ai), Jj = make_jones(aj);
                Mat2 VM;
                VM(0,0) = h_vm(b2, 0); VM(0,1) = h_vm(b2, 1);
                VM(1,0) = h_vm(b2, 2); VM(1,1) = h_vm(b2, 3);
                Mat2 pred = mat2_multiply_hermitian(mat2_multiply(Ji, VM), Jj);
                h_vo(b2, 0) = pred(0, 0);
                h_vo(b2, 1) = pred(0, 1);
                h_vo(b2, 2) = pred(1, 0);
                h_vo(b2, 3) = pred(1, 1);
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
        opts.max_iter      = 300;
        opts.tol           = 1.0e-12;
        opts.linear_solver = LinearSolverType::DENSE_CHOLESKY;

        SolverResult res = solve_D(inp, opts);

        printf("\n  D solver: cost=%.3e, iters=%d, converged=%s\n",
               res.cost, res.n_iter, res.converged ? "yes" : "no");

        check(res.converged, "D solver converged");
        check(res.cost < 1.0e-20, "D solver cost < 1e-20");

        auto h_jones = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res.jones);

        // Check recovered leakage for non-ref antennas.
        // Jones layout: col 0 = J(0,0), col 1 = J(0,1)=dpq, col 2 = J(1,0)=dqp, col 3 = J(1,1)
        bool leakage_ok = true;
        for (int a = 1; a < n_ant; ++a) {
            complex_type dpq_got = h_jones(a, 1);
            complex_type dqp_got = h_jones(a, 2);
            if (Kokkos::abs(dpq_got - true_dpq[a]) > 1.0e-8) {
                printf("    ant %d: dpq_got=(%.4f,%.4f) true=(%.4f,%.4f)\n",
                       a, dpq_got.real(), dpq_got.imag(),
                       true_dpq[a].real(), true_dpq[a].imag());
                leakage_ok = false;
            }
            if (Kokkos::abs(dqp_got - true_dqp[a]) > 1.0e-8) {
                printf("    ant %d: dqp_got=(%.4f,%.4f) true=(%.4f,%.4f)\n",
                       a, dqp_got.real(), dqp_got.imag(),
                       true_dqp[a].real(), true_dqp[a].imag());
                leakage_ok = false;
            }
        }
        check(leakage_ok, "D solver recovers true leakages for all antennas");

        // Ref antenna: diagonal = 1, off-diag = 0.
        bool ref_ok = (Kokkos::abs(h_jones(ref_ant, 0) - complex_type(1,0)) < 1e-14 &&
                       Kokkos::abs(h_jones(ref_ant, 1)) < 1e-14 &&
                       Kokkos::abs(h_jones(ref_ant, 2)) < 1e-14 &&
                       Kokkos::abs(h_jones(ref_ant, 3) - complex_type(1,0)) < 1e-14);
        check(ref_ok, "D solver: ref_ant jones is identity");

        printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    }
    Kokkos::finalize();
    return n_fail == 0 ? 0 : 1;
}
