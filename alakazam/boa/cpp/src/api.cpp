// boa/api.cpp — Problem structs and top-level solve functions.
//
// Each Problem struct encapsulates one solver type. It holds references to
// the data (antenna maps, visibilities, frequencies) and implements the four
// methods that solve_lm<Problem> calls:
//   n_residuals(), n_params(), build_residual(), fill_jacobian(),
//   build_csr_pattern(), initial_params(), params_to_jones().
//
// The solver functions (solve_G etc.) build the AntennaMap, create a Problem,
// and call solve_lm.

#include <boa/api.hpp>
#include <boa/lm.hpp>
#include <boa/residual.hpp>
#include <boa/jacobian.hpp>
#include <boa/parameterization.hpp>
#include <KokkosBlas.hpp>
#include <cmath>

namespace boa {

// ===========================================================================
// GainProblem — diagonal RIME, freq-independent
// ===========================================================================

struct GainProblem {
    // Data references
    view_2d_complex vis_obs;
    view_2d_complex vis_model;
    view_1d_int     ant1;
    view_1d_int     ant2;
    view_1d_int     ant_to_param;
    host_view_1d_int h_ant1;
    host_view_1d_int h_ant2;
    AntennaMap amap;
    int n_bl;
    int n_ant;

    int n_residuals() const { return n_bl * GainParam::residuals_per_bl; }
    int n_params()    const { return amap.n_params; }

    crs_matrix_type build_csr_pattern() const {
        return build_csr_diagonal(h_ant1, h_ant2, amap, 1);
    }

    // Initial params: identity gain for all non-ref antennas: amp=1, phase=0.
    view_1d_real initial_params() const {
        const int n = amap.n_params;
        view_1d_real p("g_init", n);
        auto h = Kokkos::create_mirror_view(p);
        for (int a = 0; a < n_ant; ++a) {
            const int off = amap.h_ant_to_param(a);
            if (off < 0) continue;
            h(off + 0) = 1.0;  // amp_p
            h(off + 1) = 0.0;  // phase_p
            h(off + 2) = 1.0;  // amp_q
            h(off + 3) = 0.0;  // phase_q
        }
        Kokkos::deep_copy(p, h);
        return p;
    }

    void build_residual(const view_1d_real& params, view_1d_real& r) const {
        // Convert params to per-antenna Jones diagonal.
        view_1d_complex g_p("g_p", n_ant);
        view_1d_complex g_q("g_q", n_ant);
        auto h_gp = Kokkos::create_mirror_view(g_p);
        auto h_gq = Kokkos::create_mirror_view(g_q);
        auto h_p  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), params);

        for (int a = 0; a < n_ant; ++a) {
            real_type pi[4] = {1.0, 0.0, 1.0, 0.0};
            const int off = amap.h_ant_to_param(a);
            if (off >= 0) for (int k = 0; k < 4; ++k) pi[k] = h_p(off + k);
            complex_type gpi, gqi;
            GainParam::params_to_diagonal(pi, gpi, gqi);
            h_gp(a) = gpi;
            h_gq(a) = gqi;
        }
        Kokkos::deep_copy(g_p, h_gp);
        Kokkos::deep_copy(g_q, h_gq);
        build_residual_diagonal(r, g_p, g_q, vis_obs, vis_model, ant1, ant2, n_bl);
    }

    void fill_jacobian(const view_1d_real& params, crs_matrix_type& J) const {
        fill_jacobian_gain(J, params, vis_model, ant1, ant2,
                           ant_to_param, n_bl, GainParam::params_per_ant);
    }

    // Convert params to Jones matrix (n_ant x 4 complex, flattened 2x2).
    view_2d_complex params_to_jones(const view_1d_real& params) const {
        view_2d_complex jones("jones", n_ant, 4);
        auto h_j = Kokkos::create_mirror_view(jones);
        auto h_p = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), params);
        for (int a = 0; a < n_ant; ++a) {
            real_type pi[4] = {1.0, 0.0, 1.0, 0.0};
            const int off = amap.h_ant_to_param(a);
            if (off >= 0) for (int k = 0; k < 4; ++k) pi[k] = h_p(off + k);
            complex_type gp, gq;
            GainParam::params_to_diagonal(pi, gp, gq);
            // Diagonal Jones: [[gp, 0], [0, gq]] → flattened [gp, 0, 0, gq]
            h_j(a, 0) = gp;
            h_j(a, 1) = complex_type(0.0, 0.0);
            h_j(a, 2) = complex_type(0.0, 0.0);
            h_j(a, 3) = gq;
        }
        Kokkos::deep_copy(jones, h_j);
        return jones;
    }
};

SolverResult solve_G(const SolverInput& inp, const SolverOptions& opts)
{
    const int n_bl = inp.ant1.extent(0);
    AntennaMap amap = build_antenna_map(inp.n_ant, inp.ref_ant, GainParam::params_per_ant);

    auto h_ant1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), inp.ant1);
    auto h_ant2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), inp.ant2);

    GainProblem prob;
    prob.vis_obs      = inp.vis_obs;
    prob.vis_model    = inp.vis_model;
    prob.ant1         = inp.ant1;
    prob.ant2         = inp.ant2;
    prob.ant_to_param = amap.ant_to_param;
    prob.h_ant1       = h_ant1;
    prob.h_ant2       = h_ant2;
    prob.amap         = amap;
    prob.n_bl         = n_bl;
    prob.n_ant        = inp.n_ant;

    return solve_lm(prob, opts, inp.init_params);
}

// ===========================================================================
// DelayProblem — diagonal RIME, freq-dependent
// ===========================================================================

struct DelayProblem {
    view_2d_complex vis_obs;
    view_2d_complex vis_model;
    view_1d_int     ant1;
    view_1d_int     ant2;
    view_1d_int     ant_to_param;
    view_1d_real    freqs;
    host_view_1d_int h_ant1;
    host_view_1d_int h_ant2;
    AntennaMap amap;
    int n_bl;
    int n_freq;
    int n_ant;

    int n_residuals() const { return n_bl * n_freq * DelayParam::residuals_per_bl_per_freq; }
    int n_params()    const { return amap.n_params; }

    crs_matrix_type build_csr_pattern() const {
        return build_csr_diagonal(h_ant1, h_ant2, amap, n_freq);
    }

    view_1d_real initial_params() const {
        const int n = amap.n_params;
        view_1d_real p("k_init", n);
        Kokkos::deep_copy(p, real_type(0.0));  // tau=0 for all antennas
        return p;
    }

    void build_residual(const view_1d_real& params, view_1d_real& r) const {
        // g_p, g_q: (n_ant, n_freq) complex
        view_2d_complex g_p("g_p", n_ant, n_freq);
        view_2d_complex g_q("g_q", n_ant, n_freq);
        auto h_gp = Kokkos::create_mirror_view(g_p);
        auto h_gq = Kokkos::create_mirror_view(g_q);
        auto h_p  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), params);
        auto h_fr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), freqs);

        for (int a = 0; a < n_ant; ++a) {
            real_type pi[2] = {0.0, 0.0};
            const int off = amap.h_ant_to_param(a);
            if (off >= 0) for (int k = 0; k < 2; ++k) pi[k] = h_p(off + k);
            for (int f = 0; f < n_freq; ++f) {
                complex_type gpi, gqi;
                DelayParam::params_to_diagonal(pi, h_fr(f), gpi, gqi);
                h_gp(a, f) = gpi;
                h_gq(a, f) = gqi;
            }
        }
        Kokkos::deep_copy(g_p, h_gp);
        Kokkos::deep_copy(g_q, h_gq);
        build_residual_diagonal_freq(r, g_p, g_q, vis_obs, vis_model,
                                     ant1, ant2, n_bl, n_freq);
    }

    void fill_jacobian(const view_1d_real& params, crs_matrix_type& J) const {
        fill_jacobian_delay(J, params, vis_model, ant1, ant2, ant_to_param,
                            freqs, n_bl, n_freq, DelayParam::params_per_ant);
    }

    view_2d_complex params_to_jones(const view_1d_real& params) const {
        // For delay, jones is diagonal at zero frequency (tau=0 → identity).
        // Return identity matrix for all antennas — caller uses params directly.
        view_2d_complex jones("jones_K", n_ant, 4);
        auto h_j = Kokkos::create_mirror_view(jones);
        for (int a = 0; a < n_ant; ++a) {
            h_j(a, 0) = complex_type(1.0, 0.0);
            h_j(a, 1) = complex_type(0.0, 0.0);
            h_j(a, 2) = complex_type(0.0, 0.0);
            h_j(a, 3) = complex_type(1.0, 0.0);
        }
        Kokkos::deep_copy(jones, h_j);
        return jones;
    }
};

SolverResult solve_K(const SolverInput& inp, const SolverOptions& opts)
{
    const int n_bl   = inp.ant1.extent(0);
    const int n_freq = inp.freqs.extent(0);
    AntennaMap amap  = build_antenna_map(inp.n_ant, inp.ref_ant, DelayParam::params_per_ant);

    auto h_ant1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), inp.ant1);
    auto h_ant2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), inp.ant2);

    DelayProblem prob;
    prob.vis_obs      = inp.vis_obs;
    prob.vis_model    = inp.vis_model;
    prob.ant1         = inp.ant1;
    prob.ant2         = inp.ant2;
    prob.ant_to_param = amap.ant_to_param;
    prob.freqs        = inp.freqs;
    prob.h_ant1       = h_ant1;
    prob.h_ant2       = h_ant2;
    prob.amap         = amap;
    prob.n_bl         = n_bl;
    prob.n_freq       = n_freq;
    prob.n_ant        = inp.n_ant;

    return solve_lm(prob, opts, inp.init_params);
}

// ===========================================================================
// LeakageProblem — full 2x2 RIME, freq-independent
// ===========================================================================

struct LeakageProblem {
    view_2d_complex vis_obs;
    view_2d_complex vis_model;
    view_1d_int     ant1;
    view_1d_int     ant2;
    view_1d_int     ant_to_param;
    host_view_1d_int h_ant1;
    host_view_1d_int h_ant2;
    AntennaMap amap;
    int n_bl;
    int n_ant;
    // Ceres convention: d_pq[ref]=0 (excluded), d_qp[ref] free at this offset.
    // ref_dqp_off = amap.n_params = (n_ant-1)*4
    int ref_dqp_off;

    int n_residuals() const { return n_bl * LeakageParam::residuals_per_bl; }
    // Total params: (n_ant-1)*4 + 2  (non-ref ants: 4 each, ref d_qp: 2)
    int n_params()    const { return ref_dqp_off + 2; }

    crs_matrix_type build_csr_pattern() const {
        return build_csr_leakage_ceres(h_ant1, h_ant2, amap);
    }

    view_1d_real initial_params() const {
        view_1d_real p("d_init", n_params());
        Kokkos::deep_copy(p, real_type(0.0));
        return p;
    }

    void build_residual(const view_1d_real& params, view_1d_real& r) const {
        view_1d_mat2 J_cur("J_cur", n_ant);
        auto h_J = Kokkos::create_mirror_view(J_cur);
        auto h_p = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), params);

        for (int a = 0; a < n_ant; ++a) {
            real_type pi[4] = {0.0, 0.0, 0.0, 0.0};
            const int off = amap.h_ant_to_param(a);
            if (off >= 0) {
                for (int k = 0; k < 4; ++k) pi[k] = h_p(off + k);
            } else {
                // ref: d_pq=0, d_qp from ref_dqp_off
                pi[2] = h_p(ref_dqp_off);
                pi[3] = h_p(ref_dqp_off + 1);
            }
            h_J(a) = LeakageParam::params_to_jones(pi);
        }
        Kokkos::deep_copy(J_cur, h_J);
        build_residual_full_2x2(r, J_cur, vis_obs, vis_model, ant1, ant2, n_bl);
    }

    void fill_jacobian(const view_1d_real& params, crs_matrix_type& J) const {
        fill_jacobian_leakage_ceres(J, params, vis_model, ant1, ant2,
                                    ant_to_param, amap.ref_ant, ref_dqp_off, n_bl);
    }

    view_2d_complex params_to_jones(const view_1d_real& params) const {
        view_2d_complex jones("jones_D", n_ant, 4);
        auto h_j = Kokkos::create_mirror_view(jones);
        auto h_p = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), params);
        for (int a = 0; a < n_ant; ++a) {
            real_type pi[4] = {0.0, 0.0, 0.0, 0.0};
            const int off = amap.h_ant_to_param(a);
            if (off >= 0) {
                for (int k = 0; k < 4; ++k) pi[k] = h_p(off + k);
            } else {
                // ref: d_pq=0, d_qp from ref_dqp_off
                pi[2] = h_p(ref_dqp_off);
                pi[3] = h_p(ref_dqp_off + 1);
            }
            Mat2 Ja = LeakageParam::params_to_jones(pi);
            h_j(a, 0) = Ja(0, 0);
            h_j(a, 1) = Ja(0, 1);
            h_j(a, 2) = Ja(1, 0);
            h_j(a, 3) = Ja(1, 1);
        }
        Kokkos::deep_copy(jones, h_j);
        return jones;
    }
};

SolverResult solve_D(const SolverInput& inp, const SolverOptions& opts)
{
    const int n_bl = inp.ant1.extent(0);
    AntennaMap amap = build_antenna_map(inp.n_ant, inp.ref_ant, LeakageParam::params_per_ant);

    auto h_ant1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), inp.ant1);
    auto h_ant2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), inp.ant2);

    LeakageProblem prob;
    prob.vis_obs      = inp.vis_obs;
    prob.vis_model    = inp.vis_model;
    prob.ant1         = inp.ant1;
    prob.ant2         = inp.ant2;
    prob.ant_to_param = amap.ant_to_param;
    prob.h_ant1       = h_ant1;
    prob.h_ant2       = h_ant2;
    prob.amap         = amap;
    prob.n_bl         = n_bl;
    prob.n_ant        = inp.n_ant;
    prob.ref_dqp_off  = amap.n_params;   // = (n_ant-1)*4

    return solve_lm(prob, opts, inp.init_params);
}

// ===========================================================================
// CrossDelayProblem — 1 global delay, freq-dependent, cross-hands only
// ===========================================================================

struct CrossDelayProblem {
    view_2d_complex vis_obs;
    view_2d_complex vis_model;
    view_1d_int     ant1;
    view_1d_int     ant2;
    view_1d_real    freqs;
    int n_bl;
    int n_freq;
    int n_ant;

    int n_residuals() const { return n_bl * n_freq * 4; }  // Re/Im pq + Re/Im qp per (bl,freq)
    int n_params()    const { return CrossDelayParam::n_global_params; }

    crs_matrix_type build_csr_pattern() const {
        return build_csr_global(n_residuals(), CrossDelayParam::n_global_params);
    }

    view_1d_real initial_params() const {
        view_1d_real p("kc_init", 1);
        Kokkos::deep_copy(p, real_type(0.0));
        return p;
    }

    void build_residual(const view_1d_real& params, view_1d_real& r) const {
        // Cross-delay: only pq and qp correlations, freq-dependent.
        // g_p = exp(-2πi τ ν), g_q = 1 (global for all antennas)
        auto h_p  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), params);
        auto h_fr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), freqs);
        const real_type tau_ns = h_p(0);

        // Build freq-dependent g_p (n_ant=1 effectively, global).
        // Residual: r_pq = V_obs_pq - g_p * V_M_pq, r_qp = V_obs_qp - V_M_qp * conj(g_p)
        // Layout: for each (bl, freq): [Re(r_pq), Im(r_pq), Re(r_qp), Im(r_qp)]
        // Build on host then upload (small number of channels in practice).
        auto h_r = Kokkos::create_mirror_view(r);

        // Get vis on host for cross-hand residuals.
        auto h_vobs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vis_obs);
        auto h_vmod = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vis_model);

        const real_type two_pi = 2.0 * 3.14159265358979323846;
        for (int b = 0; b < n_bl; ++b) {
            for (int f = 0; f < n_freq; ++f) {
                const int vis_row = b * n_freq + f;
                const real_type phase = -two_pi * tau_ns * 1.0e-9 * h_fr(f);
                const complex_type gp(std::cos(phase), std::sin(phase));

                const complex_type obs_pq = h_vobs(vis_row, 1);
                const complex_type obs_qp = h_vobs(vis_row, 2);
                const complex_type mod_pq = h_vmod(vis_row, 1);
                const complex_type mod_qp = h_vmod(vis_row, 2);

                const complex_type pred_pq = gp * mod_pq;
                const complex_type pred_qp = mod_qp * Kokkos::conj(gp);

                const complex_type diff_pq = obs_pq - pred_pq;
                const complex_type diff_qp = obs_qp - pred_qp;

                const int base = (b * n_freq + f) * 4;
                h_r(base + 0) = diff_pq.real();
                h_r(base + 1) = diff_pq.imag();
                h_r(base + 2) = diff_qp.real();
                h_r(base + 3) = diff_qp.imag();
            }
        }
        Kokkos::deep_copy(r, h_r);
    }

    void fill_jacobian(const view_1d_real& params, crs_matrix_type& J) const {
        fill_jacobian_cross_delay(J, params, vis_model, ant1, ant2,
                                  freqs, n_bl, n_freq);
    }

    view_2d_complex params_to_jones(const view_1d_real& /*params*/) const {
        view_2d_complex jones("jones_KC", n_ant, 4);
        auto h_j = Kokkos::create_mirror_view(jones);
        for (int a = 0; a < n_ant; ++a) {
            h_j(a, 0) = complex_type(1.0, 0.0);
            h_j(a, 1) = complex_type(0.0, 0.0);
            h_j(a, 2) = complex_type(0.0, 0.0);
            h_j(a, 3) = complex_type(1.0, 0.0);
        }
        Kokkos::deep_copy(jones, h_j);
        return jones;
    }
};

SolverResult solve_KC(const SolverInput& inp, const SolverOptions& opts)
{
    const int n_bl   = inp.ant1.extent(0);
    const int n_freq = inp.freqs.extent(0);

    CrossDelayProblem prob;
    prob.vis_obs  = inp.vis_obs;
    prob.vis_model = inp.vis_model;
    prob.ant1     = inp.ant1;
    prob.ant2     = inp.ant2;
    prob.freqs    = inp.freqs;
    prob.n_bl     = n_bl;
    prob.n_freq   = n_freq;
    prob.n_ant    = inp.n_ant;

    return solve_lm(prob, opts, inp.init_params);
}

// ===========================================================================
// CrossPhaseProblem — 1 global phase, freq-independent, cross-hands only
// ===========================================================================

struct CrossPhaseProblem {
    view_2d_complex vis_obs;
    view_2d_complex vis_model;
    view_1d_int     ant1;
    view_1d_int     ant2;
    int n_bl;
    int n_ant;

    int n_residuals() const { return n_bl * 4; }  // Re/Im pq + Re/Im qp per bl
    int n_params()    const { return CrossPhaseParam::n_global_params; }

    crs_matrix_type build_csr_pattern() const {
        return build_csr_global(n_residuals(), CrossPhaseParam::n_global_params);
    }

    view_1d_real initial_params() const {
        view_1d_real p("cp_init", 1);
        Kokkos::deep_copy(p, real_type(0.0));
        return p;
    }

    void build_residual(const view_1d_real& params, view_1d_real& r) const {
        auto h_p    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), params);
        auto h_vobs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vis_obs);
        auto h_vmod = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vis_model);
        auto h_r    = Kokkos::create_mirror_view(r);

        const real_type phi = h_p(0);
        const complex_type gq(std::cos(phi), std::sin(phi));

        for (int b = 0; b < n_bl; ++b) {
            const complex_type obs_pq = h_vobs(b, 1);
            const complex_type obs_qp = h_vobs(b, 2);
            const complex_type mod_pq = h_vmod(b, 1);
            const complex_type mod_qp = h_vmod(b, 2);

            // pred_pq = 1 * mod_pq * conj(gq) = mod_pq * conj(gq)
            // pred_qp = gq * mod_qp * conj(1) = gq * mod_qp
            const complex_type pred_pq = mod_pq * Kokkos::conj(gq);
            const complex_type pred_qp = gq * mod_qp;

            const complex_type diff_pq = obs_pq - pred_pq;
            const complex_type diff_qp = obs_qp - pred_qp;

            const int base = b * 4;
            h_r(base + 0) = diff_pq.real();
            h_r(base + 1) = diff_pq.imag();
            h_r(base + 2) = diff_qp.real();
            h_r(base + 3) = diff_qp.imag();
        }
        Kokkos::deep_copy(r, h_r);
    }

    void fill_jacobian(const view_1d_real& params, crs_matrix_type& J) const {
        fill_jacobian_cross_phase(J, params, vis_model, ant1, ant2, n_bl);
    }

    view_2d_complex params_to_jones(const view_1d_real& /*params*/) const {
        view_2d_complex jones("jones_CP", n_ant, 4);
        auto h_j = Kokkos::create_mirror_view(jones);
        for (int a = 0; a < n_ant; ++a) {
            h_j(a, 0) = complex_type(1.0, 0.0);
            h_j(a, 1) = complex_type(0.0, 0.0);
            h_j(a, 2) = complex_type(0.0, 0.0);
            h_j(a, 3) = complex_type(1.0, 0.0);
        }
        Kokkos::deep_copy(jones, h_j);
        return jones;
    }
};

SolverResult solve_CP(const SolverInput& inp, const SolverOptions& opts)
{
    const int n_bl = inp.ant1.extent(0);

    CrossPhaseProblem prob;
    prob.vis_obs   = inp.vis_obs;
    prob.vis_model = inp.vis_model;
    prob.ant1      = inp.ant1;
    prob.ant2      = inp.ant2;
    prob.n_bl      = n_bl;
    prob.n_ant     = inp.n_ant;

    return solve_lm(prob, opts, inp.init_params);
}

}  // namespace boa
