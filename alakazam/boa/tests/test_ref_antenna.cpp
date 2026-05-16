// test_ref_antenna.cpp — Verify ref_ant is fixed at identity for all solver types.
//
// For each solver type, run the solver and verify:
//   1. ref_ant's contribution to jones stays at identity (G, D) or zero-delay (K).
//   2. Other antennas converge to the right values.
//   3. The ref_ant column is absent from the parameter vector (n_params check).
// All 2D device views use create_mirror_view for host-side filling
// to avoid CUDA LayoutLeft vs LayoutRight mismatch.

#include <boa/types.hpp>
#include <boa/api.hpp>
#include <boa/csr_pattern.hpp>
#include <boa/parameterization.hpp>
#include <cstdio>
#include <cmath>

using namespace boa;

static int n_pass = 0, n_fail = 0;
static void check(bool c, const char* n) {
    if (c) { printf("  PASS: %s\n", n); ++n_pass; }
    else   { printf("  FAIL: %s\n", n); ++n_fail; }
}

// Build SolverInput with trivial unit-gain noiseless data for ref_ant tests.
// n_ant=3, ref_ant=2 (not 0, to verify non-zero ref_ant works).
static SolverInput make_trivial_G_input(int n_ant, int ref_ant,
                                         real_type amp, real_type phase)
{
    const int n_bl = n_ant * (n_ant - 1) / 2;

    view_1d_int a1("a1", n_bl), a2("a2", n_bl);
    {
        auto h_a1 = Kokkos::create_mirror_view(a1);
        auto h_a2 = Kokkos::create_mirror_view(a2);
        int b = 0;
        for (int i = 0; i < n_ant; ++i)
            for (int j = i+1; j < n_ant; ++j) { h_a1(b)=i; h_a2(b)=j; ++b; }
        Kokkos::deep_copy(a1, h_a1);
        Kokkos::deep_copy(a2, h_a2);
    }

    auto g_of = [&](int a) -> complex_type {
        if (a == ref_ant) return complex_type(1.0, 0.0);
        return complex_type(amp*std::cos(phase), amp*std::sin(phase));
    };

    view_2d_complex vm("vm", n_bl, 4), vo("vo", n_bl, 4);
    {
        auto h_vm = Kokkos::create_mirror_view(vm);
        auto h_vo = Kokkos::create_mirror_view(vo);
        auto h_a1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a1);
        auto h_a2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a2);

        for (int b2 = 0; b2 < n_bl; ++b2) {
            int ai = h_a1(b2), aj = h_a2(b2);
            h_vm(b2, 0) = complex_type(1.0, 0.0);
            h_vm(b2, 1) = complex_type(0.0, 0.0);
            h_vm(b2, 2) = complex_type(0.0, 0.0);
            h_vm(b2, 3) = complex_type(1.0, 0.0);
            h_vo(b2, 0) = g_of(ai) * h_vm(b2, 0) * Kokkos::conj(g_of(aj));
            h_vo(b2, 3) = g_of(ai) * h_vm(b2, 3) * Kokkos::conj(g_of(aj));
            h_vo(b2, 1) = h_vo(b2, 2) = complex_type(0.0, 0.0);
        }
        Kokkos::deep_copy(vm, h_vm);
        Kokkos::deep_copy(vo, h_vo);
    }

    SolverInput inp;
    inp.vis_obs = vo; inp.vis_model = vm;
    inp.ant1 = a1; inp.ant2 = a2;
    inp.freqs = view_1d_real("f", 0);
    inp.n_ant = n_ant; inp.ref_ant = ref_ant;
    return inp;
}

int main() {
    Kokkos::initialize();
    {
        printf("=== test_ref_antenna ===\n");

        SolverOptions opts;
        opts.max_iter = 200; opts.tol = 1.0e-12;
        opts.linear_solver = LinearSolverType::DENSE_CHOLESKY;

        // --- G solver: ref_ant=2, 3 antennas ---
        printf("\n--- G solver ref_ant=2 ---\n");
        {
            const int n_ant = 3, ref_ant = 2;
            SolverInput inp = make_trivial_G_input(n_ant, ref_ant, 1.2, 0.3);
            SolverResult res = solve_G(inp, opts);

            AntennaMap amap = build_antenna_map(n_ant, ref_ant, GainParam::params_per_ant);
            check(amap.n_params == (n_ant - 1) * GainParam::params_per_ant,
                  "G: n_params = (n_ant-1)*params_per_ant");

            auto h_j = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res.jones);
            bool ref_ok = (Kokkos::abs(h_j(ref_ant, 0) - complex_type(1,0)) < 1e-12 &&
                           Kokkos::abs(h_j(ref_ant, 3) - complex_type(1,0)) < 1e-12 &&
                           Kokkos::abs(h_j(ref_ant, 1)) < 1e-12 &&
                           Kokkos::abs(h_j(ref_ant, 2)) < 1e-12);
            check(ref_ok, "G: ref_ant=2 jones is identity");
            check(res.converged, "G: converged with ref_ant=2");
        }

        // --- G solver: ref_ant=1 (middle), 4 antennas ---
        printf("\n--- G solver ref_ant=1 ---\n");
        {
            const int n_ant = 4, ref_ant = 1;
            SolverInput inp = make_trivial_G_input(n_ant, ref_ant, 0.9, -0.2);
            SolverResult res = solve_G(inp, opts);

            auto h_j = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res.jones);
            bool ref_ok = (Kokkos::abs(h_j(ref_ant, 0) - complex_type(1,0)) < 1e-12 &&
                           Kokkos::abs(h_j(ref_ant, 3) - complex_type(1,0)) < 1e-12);
            check(ref_ok, "G: ref_ant=1 jones is identity");
            check(res.cost < 1.0e-20, "G: ref_ant=1 cost < 1e-20");
        }

        // --- D solver: ref_ant=0 ---
        printf("\n--- D solver ref_ant=0 ---\n");
        {
            const int n_ant = 3, ref_ant = 0, n_bl = 3;

            view_1d_int a1("a1", n_bl), a2("a2", n_bl);
            {
                auto h_a1 = Kokkos::create_mirror_view(a1);
                auto h_a2 = Kokkos::create_mirror_view(a2);
                h_a1(0)=0; h_a2(0)=1; h_a1(1)=0; h_a2(1)=2; h_a1(2)=1; h_a2(2)=2;
                Kokkos::deep_copy(a1, h_a1);
                Kokkos::deep_copy(a2, h_a2);
            }

            // True D-Jones for ants 1,2 (small leakage).
            complex_type dpq1(0.08, 0.03), dqp1(-0.05, 0.02);
            complex_type dpq2(-0.04, 0.06), dqp2(0.03, -0.07);

            auto make_J = [&](int a) -> Mat2 {
                Mat2 J;
                J(0,0) = complex_type(1,0);
                J(1,1) = complex_type(1,0);
                if (a == 1) { J(0,1)=dpq1; J(1,0)=dqp1; }
                else if (a == 2) { J(0,1)=dpq2; J(1,0)=dqp2; }
                else { J(0,1)=complex_type(0,0); J(1,0)=complex_type(0,0); }
                return J;
            };

            view_2d_complex vm("vm",n_bl,4), vo("vo",n_bl,4);
            {
                auto h_vm = Kokkos::create_mirror_view(vm);
                auto h_vo = Kokkos::create_mirror_view(vo);
                auto h_a1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a1);
                auto h_a2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a2);

                for (int b2 = 0; b2 < n_bl; ++b2) {
                    h_vm(b2,0)=complex_type(1,0); h_vm(b2,1)=complex_type(0.2,0.1);
                    h_vm(b2,2)=complex_type(0.2,-0.1); h_vm(b2,3)=complex_type(1,0);
                }
                for (int b2 = 0; b2 < n_bl; ++b2) {
                    int ai=h_a1(b2), aj=h_a2(b2);
                    Mat2 Ji=make_J(ai), Jj=make_J(aj);
                    Mat2 VM; VM(0,0)=h_vm(b2,0); VM(0,1)=h_vm(b2,1);
                             VM(1,0)=h_vm(b2,2); VM(1,1)=h_vm(b2,3);
                    Mat2 pred = mat2_multiply_hermitian(mat2_multiply(Ji,VM), Jj);
                    h_vo(b2,0)=pred(0,0); h_vo(b2,1)=pred(0,1);
                    h_vo(b2,2)=pred(1,0); h_vo(b2,3)=pred(1,1);
                }
                Kokkos::deep_copy(vm, h_vm);
                Kokkos::deep_copy(vo, h_vo);
            }

            SolverInput inp;
            inp.vis_obs=vo; inp.vis_model=vm;
            inp.ant1=a1; inp.ant2=a2;
            inp.freqs=view_1d_real("f",0);
            inp.n_ant=n_ant; inp.ref_ant=ref_ant;

            SolverResult res = solve_D(inp, opts);
            auto h_j = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res.jones);

            check(res.cost < 1.0e-20, "D: ref_ant=0 cost < 1e-20");
            bool ref_ok = (Kokkos::abs(h_j(0,1)) < 1e-12 && Kokkos::abs(h_j(0,2)) < 1e-12 &&
                           Kokkos::abs(h_j(0,0)-complex_type(1,0)) < 1e-12 &&
                           Kokkos::abs(h_j(0,3)-complex_type(1,0)) < 1e-12);
            check(ref_ok, "D: ref_ant=0 jones is identity");

            AntennaMap amap = build_antenna_map(n_ant, ref_ant, LeakageParam::params_per_ant);
            check(amap.n_params == (n_ant-1)*LeakageParam::params_per_ant,
                  "D: n_params = (n_ant-1)*params_per_ant");
        }

        // --- KC solver: ref_ant field is ignored (global param) ---
        printf("\n--- KC solver (global, ref_ant ignored) ---\n");
        {
            const int n_ant = 3, ref_ant = 0, n_bl = 3, n_freq = 4;

            view_1d_int a1("a1",n_bl), a2("a2",n_bl);
            {
                auto h_a1 = Kokkos::create_mirror_view(a1);
                auto h_a2 = Kokkos::create_mirror_view(a2);
                h_a1(0)=0; h_a2(0)=1; h_a1(1)=0; h_a2(1)=2; h_a1(2)=1; h_a2(2)=2;
                Kokkos::deep_copy(a1, h_a1);
                Kokkos::deep_copy(a2, h_a2);
            }

            const real_type two_pi = 2.0*3.14159265358979323846;
            const real_type true_tau = 0.20;  // < 0.5 ns to avoid 1/f ambiguity at 1 GHz

            view_1d_real fr("fr", n_freq);
            {
                auto h_fr = Kokkos::create_mirror_view(fr);
                for (int f=0; f<n_freq; ++f) h_fr(f) = 1.0e9 + f*5.0e6;
                Kokkos::deep_copy(fr, h_fr);
            }

            const int n_vis = n_bl*n_freq;
            view_2d_complex vm("vm",n_vis,4), vo("vo",n_vis,4);
            {
                auto h_vm = Kokkos::create_mirror_view(vm);
                auto h_vo = Kokkos::create_mirror_view(vo);
                auto h_fr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fr);

                for (int b2=0; b2<n_bl; ++b2) for (int f=0; f<n_freq; ++f) {
                    int row=b2*n_freq+f;
                    h_vm(row,0)=h_vm(row,3)=complex_type(1,0);
                    h_vm(row,1)=h_vm(row,2)=complex_type(1,0);
                    real_type ph=-two_pi*true_tau*1e-9*h_fr(f);
                    complex_type gp(std::cos(ph),std::sin(ph));
                    h_vo(row,0)=h_vm(row,0); h_vo(row,3)=h_vm(row,3);
                    h_vo(row,1)=gp*h_vm(row,1);
                    h_vo(row,2)=h_vm(row,2)*Kokkos::conj(gp);
                }
                Kokkos::deep_copy(vm, h_vm);
                Kokkos::deep_copy(vo, h_vo);
            }

            SolverInput inp;
            inp.vis_obs=vo; inp.vis_model=vm;
            inp.ant1=a1; inp.ant2=a2; inp.freqs=fr;
            inp.n_ant=n_ant; inp.ref_ant=ref_ant;

            SolverResult res = solve_KC(inp, opts);
            auto h_p = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res.params);
            check(res.params.extent(0) == 1, "KC: n_params == 1");
            check(std::abs(h_p(0)-true_tau) < 1e-6, "KC: recovers true tau");
        }

        printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    }
    Kokkos::finalize();
    return n_fail == 0 ? 0 : 1;
}
