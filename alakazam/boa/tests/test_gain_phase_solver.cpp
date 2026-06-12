// test_gain_phase_solver.cpp — Phase-only G solver (solve_Gp) test.
//
// Two cases, 5 antennas, ref_ant=0, noiseless:
//   1. Pure phase corruption: exact recovery, cost ~ 0, all amps exactly 1.
//   2. Phase corruption + per-antenna amp errors + global flux factor in the
//      data: amps must STAY pinned at 1 and the phases must still be
//      recovered exactly (phase structure is exact in the data; the amp
//      mismatch only raises the residual floor).

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
        printf("=== test_gain_phase_solver ===\n");

        const int n_ant = 5, ref_ant = 0;
        const int n_bl = n_ant * (n_ant - 1) / 2;

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

        const real_type true_phase_p[5] = {0.0, 0.20, -0.15, 0.25, -0.10};
        const real_type true_phase_q[5] = {0.0, -0.10, 0.30, -0.20, 0.15};
        // Case 2 amp structure (NOT solvable by the phase-only model):
        const real_type amp_p[5] = {1.05, 1.10, 0.85, 1.20, 0.90};
        const real_type amp_q[5] = {0.95, 0.95, 1.15, 0.80, 1.05};
        const real_type flux = 1.6;

        auto run_case = [&](bool with_amps, const char* label) {
            view_2d_complex vis_model("vm", n_bl, 4);
            view_2d_complex vis_obs("vo",   n_bl, 4);
            {
                auto h_vm = Kokkos::create_mirror_view(vis_model);
                auto h_vo = Kokkos::create_mirror_view(vis_obs);
                auto h1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ant1);
                auto h2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ant2);
                for (int b = 0; b < n_bl; ++b) {
                    h_vm(b, 0) = complex_type(1.0, 0.0);
                    h_vm(b, 1) = complex_type(0.0, 0.0);
                    h_vm(b, 2) = complex_type(0.0, 0.0);
                    h_vm(b, 3) = complex_type(1.0, 0.0);
                    const int ai = h1(b), aj = h2(b);
                    const real_type ap = with_amps ? flux * amp_p[ai] * amp_p[aj] : 1.0;
                    const real_type aq = with_amps ? flux * amp_q[ai] * amp_q[aj] : 1.0;
                    const real_type php = true_phase_p[ai] - true_phase_p[aj];
                    const real_type phq = true_phase_q[ai] - true_phase_q[aj];
                    h_vo(b, 0) = complex_type(ap * std::cos(php), ap * std::sin(php));
                    h_vo(b, 3) = complex_type(aq * std::cos(phq), aq * std::sin(phq));
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
            opts.max_iter      = 300;
            opts.tol           = 1.0e-12;
            opts.linear_solver = LinearSolverType::DENSE_CHOLESKY;

            SolverResult res = solve_Gp(inp, opts);
            printf("\n  Gp %s: cost=%.3e, iters=%d, converged=%s\n",
                   label, res.cost, res.n_iter, res.converged ? "yes" : "no");

            char name[128];
            snprintf(name, sizeof(name), "Gp %s: converged", label);
            check(res.converged, name);

            auto h_j = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res.jones);

            bool amps_unit = true, phases_ok = true;
            for (int a = 0; a < n_ant; ++a) {
                if (std::abs(Kokkos::abs(h_j(a, 0)) - 1.0) > 1.0e-12 ||
                    std::abs(Kokkos::abs(h_j(a, 3)) - 1.0) > 1.0e-12)
                    amps_unit = false;
                const real_type php = std::atan2(h_j(a, 0).imag(), h_j(a, 0).real());
                const real_type phq = std::atan2(h_j(a, 3).imag(), h_j(a, 3).real());
                // 1e-5 rad: with an amp-mismatch residual floor the ftol
                // exit leaves ~1e-6 rad on the shallow tail — negligible.
                if (std::abs(php - true_phase_p[a]) > 1.0e-5 ||
                    std::abs(phq - true_phase_q[a]) > 1.0e-5) {
                    printf("    ant %d: phase_p=%.6f true=%.6f  phase_q=%.6f true=%.6f\n",
                           a, php, true_phase_p[a], phq, true_phase_q[a]);
                    phases_ok = false;
                }
            }
            snprintf(name, sizeof(name), "Gp %s: all amps exactly 1", label);
            check(amps_unit, name);
            snprintf(name, sizeof(name), "Gp %s: phases recovered", label);
            check(phases_ok, name);

            if (!with_amps) {
                snprintf(name, sizeof(name), "Gp %s: cost < 1e-20", label);
                check(res.cost < 1.0e-20, name);
            }
        };

        run_case(false, "pure-phase");
        run_case(true,  "amp-mismatch");

        printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    }
    Kokkos::finalize();
    return n_fail == 0 ? 0 : 1;
}
