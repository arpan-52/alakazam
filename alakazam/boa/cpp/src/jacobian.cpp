// boa/jacobian.cpp — Fill Jacobian CSR values for each solver type.
//
// Each function is a Kokkos::parallel_for over baselines.
// Each baseline fills its block of rows in the Jacobian using the
// chain rule: ∂r/∂params = ∂r/∂g * ∂g/∂params.
//
// The CSR pattern (row_ptr, col_ind) is already fixed. We only write to
// the values array. Each baseline writes to disjoint rows, so no atomics.

#include <boa/jacobian.hpp>

namespace boa {

// ---------------------------------------------------------------------------
// Gain solver: diagonal RIME, freq-independent
// ---------------------------------------------------------------------------
// Residual layout per baseline: [Re(r_pp), Im(r_pp), Re(r_qq), Im(r_qq)]
// Each row has up to 2*params_per_ant nonzeros (ant_i cols + ant_j cols).

void fill_jacobian_gain(
    crs_matrix_type& J,
    const view_1d_real& params,
    const view_2d_complex& vis_model,
    const view_1d_int& ant1,
    const view_1d_int& ant2,
    const view_1d_int& ant_to_param,
    int n_bl,
    int params_per_ant)
{
    auto row_map = J.graph.row_map;
    auto entries = J.graph.entries;
    auto values  = J.values;

    Kokkos::parallel_for("fill_jacobian_gain", n_bl,
        KOKKOS_LAMBDA(const int b) {
            const int ai = ant1(b);
            const int aj = ant2(b);
            const int off_i = ant_to_param(ai);
            const int off_j = ant_to_param(aj);

            // Identity gain: amp=1, phase=0. Overridden for non-ref antennas below.
            real_type pi[4]; pi[0] = 1.0; pi[1] = 0.0; pi[2] = 1.0; pi[3] = 0.0;
            real_type pj[4]; pj[0] = 1.0; pj[1] = 0.0; pj[2] = 1.0; pj[3] = 0.0;
            if (off_i >= 0) for (int k = 0; k < 4; ++k) pi[k] = params(off_i + k);
            if (off_j >= 0) for (int k = 0; k < 4; ++k) pj[k] = params(off_j + k);

            // 4 residual rows per baseline.
            for (int r = 0; r < 4; ++r) {
                const int row = b * 4 + r;
                const int pol = r / 2;       // 0=pp, 1=qq
                const bool is_real = (r % 2 == 0);

                // Model value for this pol.
                const complex_type model_val = vis_model(b, pol * 3);  // col 0=pp, col 3=qq

                real_type vals_i[4], vals_j[4];
                GainParam::jacobian_one_row(vals_i, vals_j, pi, pj,
                                            model_val, pol, is_real);

                // Write to CSR values at the positions for this row.
                int pos = row_map(row);
                if (off_i >= 0) {
                    for (int k = 0; k < params_per_ant; ++k)
                        values(pos++) = vals_i[k];
                }
                if (off_j >= 0) {
                    for (int k = 0; k < params_per_ant; ++k)
                        values(pos++) = vals_j[k];
                }
            }
        });
    Kokkos::fence("fill_jacobian_gain_fence");
}

// ---------------------------------------------------------------------------
// Delay solver: diagonal RIME, freq-dependent
// ---------------------------------------------------------------------------
// Residual layout: baseline-major, freq within baseline, 4 reals per freq.
// Row index = b * (n_freq * 4) + f * 4 + r.

void fill_jacobian_delay(
    crs_matrix_type& J,
    const view_1d_real& params,
    const view_2d_complex& vis_model,
    const view_1d_int& ant1,
    const view_1d_int& ant2,
    const view_1d_int& ant_to_param,
    const view_1d_real& freqs,
    int n_bl,
    int n_freq,
    int params_per_ant)
{
    auto row_map = J.graph.row_map;
    auto values  = J.values;

    const int res_per_bl = 4 * n_freq;

    Kokkos::parallel_for("fill_jacobian_delay", n_bl * n_freq,
        KOKKOS_LAMBDA(const int idx) {
            const int b = idx / n_freq;
            const int f = idx % n_freq;
            const int ai = ant1(b);
            const int aj = ant2(b);
            const int off_i = ant_to_param(ai);
            const int off_j = ant_to_param(aj);

            // Identity delay: tau=0. Overridden for non-ref antennas below.
            real_type pi[2]; pi[0] = 0.0; pi[1] = 0.0;
            real_type pj[2]; pj[0] = 0.0; pj[1] = 0.0;
            if (off_i >= 0) for (int k = 0; k < 2; ++k) pi[k] = params(off_i + k);
            if (off_j >= 0) for (int k = 0; k < 2; ++k) pj[k] = params(off_j + k);

            const real_type freq_hz = freqs(f);
            const int vis_row = b * n_freq + f;

            for (int r = 0; r < 4; ++r) {
                const int row = b * res_per_bl + f * 4 + r;
                const int pol = r / 2;
                const bool is_real = (r % 2 == 0);
                const complex_type model_val = vis_model(vis_row, pol * 3);

                real_type vals_i[2], vals_j[2];
                DelayParam::jacobian_one_row(vals_i, vals_j, pi, pj,
                                             model_val, freq_hz, pol, is_real);

                int pos = row_map(row);
                if (off_i >= 0) {
                    for (int k = 0; k < params_per_ant; ++k)
                        values(pos++) = vals_i[k];
                }
                if (off_j >= 0) {
                    for (int k = 0; k < params_per_ant; ++k)
                        values(pos++) = vals_j[k];
                }
            }
        });
    Kokkos::fence("fill_jacobian_delay_fence");
}

// ---------------------------------------------------------------------------
// Leakage solver: full 2x2, freq-independent
// ---------------------------------------------------------------------------
// 8 residual rows per baseline.
// Row ordering: [Re(R00), Im(R00), Re(R01), Im(R01), Re(R10), Im(R10), Re(R11), Im(R11)]

void fill_jacobian_leakage(
    crs_matrix_type& J,
    const view_1d_real& params,
    const view_2d_complex& vis_model,
    const view_1d_int& ant1,
    const view_1d_int& ant2,
    const view_1d_int& ant_to_param,
    int n_bl,
    int params_per_ant)
{
    auto row_map = J.graph.row_map;
    auto values  = J.values;

    Kokkos::parallel_for("fill_jacobian_leakage", n_bl,
        KOKKOS_LAMBDA(const int b) {
            const int ai = ant1(b);
            const int aj = ant2(b);
            const int off_i = ant_to_param(ai);
            const int off_j = ant_to_param(aj);

            // Identity leakage: d_pq=0, d_qp=0. Overridden for non-ref antennas below.
            real_type pi[4]; pi[0] = 0.0; pi[1] = 0.0; pi[2] = 0.0; pi[3] = 0.0;
            real_type pj[4]; pj[0] = 0.0; pj[1] = 0.0; pj[2] = 0.0; pj[3] = 0.0;
            if (off_i >= 0) for (int k = 0; k < 4; ++k) pi[k] = params(off_i + k);
            if (off_j >= 0) for (int k = 0; k < 4; ++k) pj[k] = params(off_j + k);

            // Build VM from vis_model.
            Mat2 VM;
            VM(0, 0) = vis_model(b, 0);
            VM(0, 1) = vis_model(b, 1);
            VM(1, 0) = vis_model(b, 2);
            VM(1, 1) = vis_model(b, 3);

            for (int r = 0; r < 8; ++r) {
                const int row = b * 8 + r;

                real_type vals_i[4], vals_j[4];
                LeakageParam::jacobian_one_row(vals_i, vals_j, pi, pj, VM, r);

                int pos = row_map(row);
                if (off_i >= 0) {
                    for (int k = 0; k < params_per_ant; ++k)
                        values(pos++) = vals_i[k];
                }
                if (off_j >= 0) {
                    for (int k = 0; k < params_per_ant; ++k)
                        values(pos++) = vals_j[k];
                }
            }
        });
    Kokkos::fence("fill_jacobian_leakage_fence");
}

// ---------------------------------------------------------------------------
// Leakage solver — Ceres ref convention (full 2x2, freq-independent)
// ---------------------------------------------------------------------------
// d_pq[ref] is fixed at 0 (no columns in J for those params).
// d_qp[ref] is free — stored at params[ref_dqp_off] and params[ref_dqp_off+1].
// CSR layout per baseline involving ref: 2 cols for ref d_qp + 4 for non-ref ant.
// For non-ref baselines: 8 cols (4+4), same as fill_jacobian_leakage.

void fill_jacobian_leakage_ceres(
    crs_matrix_type& J,
    const view_1d_real& params,
    const view_2d_complex& vis_model,
    const view_1d_int& ant1,
    const view_1d_int& ant2,
    const view_1d_int& ant_to_param,
    int ref_ant,
    int ref_dqp_off,
    int n_bl)
{
    auto row_map = J.graph.row_map;
    auto values  = J.values;

    Kokkos::parallel_for("fill_jacobian_leakage_ceres", n_bl,
        KOKKOS_LAMBDA(const int b) {
            const int ai    = ant1(b);
            const int aj    = ant2(b);
            const int off_i = ant_to_param(ai);
            const int off_j = ant_to_param(aj);

            // d_pq=0 for ref, d_qp from ref_dqp_off for ref.
            real_type pi[4] = {0.0, 0.0, 0.0, 0.0};
            real_type pj[4] = {0.0, 0.0, 0.0, 0.0};
            if (off_i >= 0) {
                for (int k = 0; k < 4; ++k) pi[k] = params(off_i + k);
            } else {
                pi[2] = params(ref_dqp_off);
                pi[3] = params(ref_dqp_off + 1);
            }
            if (off_j >= 0) {
                for (int k = 0; k < 4; ++k) pj[k] = params(off_j + k);
            } else {
                pj[2] = params(ref_dqp_off);
                pj[3] = params(ref_dqp_off + 1);
            }

            Mat2 VM;
            VM(0, 0) = vis_model(b, 0);
            VM(0, 1) = vis_model(b, 1);
            VM(1, 0) = vis_model(b, 2);
            VM(1, 1) = vis_model(b, 3);

            for (int r = 0; r < 8; ++r) {
                const int row = b * 8 + r;
                real_type vals_i[4], vals_j[4];
                LeakageParam::jacobian_one_row(vals_i, vals_j, pi, pj, VM, r);

                int pos = row_map(row);
                if (off_i >= 0) {
                    for (int k = 0; k < 4; ++k) values(pos++) = vals_i[k];
                } else {
                    // ref: only d_qp columns (indices 2,3)
                    values(pos++) = vals_i[2];
                    values(pos++) = vals_i[3];
                }
                if (off_j >= 0) {
                    for (int k = 0; k < 4; ++k) values(pos++) = vals_j[k];
                } else {
                    values(pos++) = vals_j[2];
                    values(pos++) = vals_j[3];
                }
            }
        });
    Kokkos::fence("fill_jacobian_leakage_ceres_fence");
}

// ---------------------------------------------------------------------------
// Cross-delay: 1 global param, freq-dependent
// ---------------------------------------------------------------------------
// Cross-hand residuals: pq and qp, each giving 2 reals per freq.
// Total residuals per baseline per freq = 4 (Re/Im pq + Re/Im qp).
// Total = n_bl * n_freq * 4. Each row has 1 nonzero (the global param).

void fill_jacobian_cross_delay(
    crs_matrix_type& J,
    const view_1d_real& params,
    const view_2d_complex& vis_model,
    const view_1d_int& ant1,
    const view_1d_int& ant2,
    const view_1d_real& freqs,
    int n_bl,
    int n_freq)
{
    auto row_map = J.graph.row_map;
    auto values  = J.values;

    Kokkos::parallel_for("fill_jacobian_cross_delay", n_bl * n_freq,
        KOKKOS_LAMBDA(const int idx) {
            const int b = idx / n_freq;
            const int f = idx % n_freq;
            const real_type tau_ns = params(0);
            const real_type freq_hz = freqs(f);
            const int vis_row = b * n_freq + f;

            // pq model = vis_model col 1, qp model = vis_model col 2
            const complex_type model_pq = vis_model(vis_row, 1);
            const complex_type model_qp = vis_model(vis_row, 2);

            const int base_row = (b * n_freq + f) * 4;

            // Row 0: Re(r_pq), Row 1: Im(r_pq)
            real_type val;
            CrossDelayParam::jacobian_cross(val, tau_ns, freq_hz, model_pq, true, true);
            values(row_map(base_row + 0)) = val;
            CrossDelayParam::jacobian_cross(val, tau_ns, freq_hz, model_pq, true, false);
            values(row_map(base_row + 1)) = val;
            // Row 2: Re(r_qp), Row 3: Im(r_qp)
            CrossDelayParam::jacobian_cross(val, tau_ns, freq_hz, model_qp, false, true);
            values(row_map(base_row + 2)) = val;
            CrossDelayParam::jacobian_cross(val, tau_ns, freq_hz, model_qp, false, false);
            values(row_map(base_row + 3)) = val;
        });
    Kokkos::fence("fill_jacobian_cross_delay_fence");
}

// ---------------------------------------------------------------------------
// Cross-phase: 1 global param, freq-independent
// ---------------------------------------------------------------------------
// Same structure as cross-delay but no freq dimension.
// 4 residual rows per baseline: Re/Im pq + Re/Im qp.

void fill_jacobian_cross_phase(
    crs_matrix_type& J,
    const view_1d_real& params,
    const view_2d_complex& vis_model,
    const view_1d_int& ant1,
    const view_1d_int& ant2,
    int n_bl)
{
    auto row_map = J.graph.row_map;
    auto values  = J.values;

    Kokkos::parallel_for("fill_jacobian_cross_phase", n_bl,
        KOKKOS_LAMBDA(const int b) {
            const real_type phi = params(0);
            const complex_type model_pq = vis_model(b, 1);
            const complex_type model_qp = vis_model(b, 2);

            const int base_row = b * 4;

            real_type val;
            CrossPhaseParam::jacobian_cross(val, phi, model_pq, true, true);
            values(row_map(base_row + 0)) = val;
            CrossPhaseParam::jacobian_cross(val, phi, model_pq, true, false);
            values(row_map(base_row + 1)) = val;
            CrossPhaseParam::jacobian_cross(val, phi, model_qp, false, true);
            values(row_map(base_row + 2)) = val;
            CrossPhaseParam::jacobian_cross(val, phi, model_qp, false, false);
            values(row_map(base_row + 3)) = val;
        });
    Kokkos::fence("fill_jacobian_cross_phase_fence");
}

}  // namespace boa
