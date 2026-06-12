// boa/jacobian.hpp — Fill Jacobian CSR values using parameterization functors.
//
// The CSR pattern (row_ptr, col_ind) is built once by csr_pattern.cpp.
// This module fills the *values* array each LM iteration.
//
// For per-antenna solvers (G, K, D), each baseline row depends on
// antenna i's params and antenna j's params. The parameterization functor
// provides jacobian_one_row() to compute ∂residual/∂param.
//
// For global solvers (KC, CP), every row depends on the same global params.

#ifndef BOA_JACOBIAN_HPP
#define BOA_JACOBIAN_HPP

#include <boa/types.hpp>
#include <boa/csr_pattern.hpp>
#include <boa/parameterization.hpp>

namespace boa {

// ---------------------------------------------------------------------------
// Gain solver Jacobian fill
// ---------------------------------------------------------------------------
void fill_jacobian_gain(
    crs_matrix_type& J,
    const view_1d_real& params,
    const view_2d_complex& vis_model,
    const view_1d_int& ant1,
    const view_1d_int& ant2,
    const view_1d_int& ant_to_param,
    int n_bl,
    int params_per_ant);

// Phase-only gain Jacobian: amps fixed at 1 (not parameters),
// 2 columns per non-ref antenna: [phase_p, phase_q]. Ref excluded entirely.
void fill_jacobian_gain_phase(
    crs_matrix_type& J,
    const view_1d_real& params,
    const view_2d_complex& vis_model,
    const view_1d_int& ant1,
    const view_1d_int& ant2,
    const view_1d_int& ant_to_param,
    int n_bl);

// Gain Jacobian with free ref-antenna amplitudes:
//   phase[ref] fixed at 0 (no columns), amp_p/amp_q[ref] free
//   (2 cols at ref_amp_off / ref_amp_off+1).
void fill_jacobian_gain_ref_amp(
    crs_matrix_type& J,
    const view_1d_real& params,
    const view_2d_complex& vis_model,
    const view_1d_int& ant1,
    const view_1d_int& ant2,
    const view_1d_int& ant_to_param,
    int ref_amp_off,
    int n_bl);

// ---------------------------------------------------------------------------
// Delay solver Jacobian fill (freq-dependent)
// ---------------------------------------------------------------------------
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
    int params_per_ant);

// ---------------------------------------------------------------------------
// Leakage solver Jacobian fill (full 2x2)
// ---------------------------------------------------------------------------
void fill_jacobian_leakage(
    crs_matrix_type& J,
    const view_1d_real& params,
    const view_2d_complex& vis_model,
    const view_1d_int& ant1,
    const view_1d_int& ant2,
    const view_1d_int& ant_to_param,
    int n_bl,
    int params_per_ant);

// Leakage Jacobian — Ceres ref convention:
//   d_pq[ref] fixed at 0 (no columns), d_qp[ref] free (2 cols at ref_dqp_off).
void fill_jacobian_leakage_ceres(
    crs_matrix_type& J,
    const view_1d_real& params,
    const view_2d_complex& vis_model,
    const view_1d_int& ant1,
    const view_1d_int& ant2,
    const view_1d_int& ant_to_param,
    int ref_ant,
    int ref_dqp_off,
    int n_bl);

// ---------------------------------------------------------------------------
// Cross-delay Jacobian fill (global, freq-dependent)
// ---------------------------------------------------------------------------
void fill_jacobian_cross_delay(
    crs_matrix_type& J,
    const view_1d_real& params,
    const view_2d_complex& vis_model,
    const view_1d_int& ant1,
    const view_1d_int& ant2,
    const view_1d_real& freqs,
    int n_bl,
    int n_freq);

// ---------------------------------------------------------------------------
// Cross-phase Jacobian fill (global, freq-independent)
// ---------------------------------------------------------------------------
void fill_jacobian_cross_phase(
    crs_matrix_type& J,
    const view_1d_real& params,
    const view_2d_complex& vis_model,
    const view_1d_int& ant1,
    const view_1d_int& ant2,
    int n_bl);

}  // namespace boa

#endif  // BOA_JACOBIAN_HPP
