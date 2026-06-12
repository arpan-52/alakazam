// boa/api.hpp — Top-level solve functions for all five calibration types.
//
// Each function assembles the correct Problem struct for its solver type,
// then calls solve_lm<Problem>. The problem structs are defined in api.cpp
// and declared here so that tests and bindings can call these functions
// without re-implementing the LM wiring.
//
// Solver types and their parameterizations:
//   solve_G  — gain (diagonal RIME, freq-independent)
//              params: [amp_p, phase_p, amp_q, phase_q] per non-ref antenna
//              + [amp_p_ref, amp_q_ref] (ref phases fixed, amps free)
//   solve_Gp — phase-only gain (diagonal RIME, freq-independent)
//              params: [phase_p, phase_q] per non-ref antenna; ALL amps
//              fixed at 1 (they are not parameters), phase[ref]=0
//   solve_K  — delay (diagonal RIME, freq-dependent)
//              params: [tau_p_ns, tau_q_ns] per non-ref antenna
//   solve_D  — leakage (full 2x2 RIME, freq-independent)
//              params: [Re(d_pq), Im(d_pq), Re(d_qp), Im(d_qp)] per non-ref antenna
//   solve_KC — cross-delay (global, freq-dependent)
//              params: [tau_ns] (1 global parameter)
//   solve_CP — cross-phase (global, freq-independent)
//              params: [phi_rad] (1 global parameter)
//
// All functions take a SolverInput (device-side Kokkos Views) and SolverOptions.
// They return SolverResult with jones (n_ant x 4), params, cost, n_iter, converged.

#ifndef BOA_API_HPP
#define BOA_API_HPP

#include <boa/types.hpp>
#include <boa/csr_pattern.hpp>

namespace boa {

// ---------------------------------------------------------------------------
// solve_G — gain calibration
// ---------------------------------------------------------------------------
// Solves for diagonal Jones J = diag(g_p, g_q), g_p = amp_p * exp(i*phase_p).
// vis_obs, vis_model: (n_bl, 4) complex.
// freqs: ignored (pass empty view).
SolverResult solve_G(const SolverInput& inp, const SolverOptions& opts);

// ---------------------------------------------------------------------------
// solve_Gp — phase-only gain calibration
// ---------------------------------------------------------------------------
// Solves for diagonal Jones J = diag(exp(i*phase_p), exp(i*phase_q)) with
// all amplitudes fixed at 1 (amps are not parameters). phase[ref]=0.
// vis_obs, vis_model: (n_bl, 4) complex.
// freqs: ignored (pass empty view).
SolverResult solve_Gp(const SolverInput& inp, const SolverOptions& opts);

// ---------------------------------------------------------------------------
// solve_K — delay calibration
// ---------------------------------------------------------------------------
// Solves for diagonal Jones J = diag(exp(-2πi τ_p ν), exp(-2πi τ_q ν)).
// vis_obs, vis_model: (n_bl * n_freq, 4) complex (baseline-major, freq-minor).
// freqs: (n_freq,) Hz.
SolverResult solve_K(const SolverInput& inp, const SolverOptions& opts);

// ---------------------------------------------------------------------------
// solve_D — leakage calibration
// ---------------------------------------------------------------------------
// Solves for full 2x2 Jones J = [[1, d_pq], [d_qp, 1]].
// vis_obs, vis_model: (n_bl, 4) complex.
// freqs: ignored (pass empty view).
SolverResult solve_D(const SolverInput& inp, const SolverOptions& opts);

// ---------------------------------------------------------------------------
// solve_KC — cross-delay calibration
// ---------------------------------------------------------------------------
// Solves for 1 global delay tau. J = diag(exp(-2πi τ ν), 1).
// Only cross-hand correlations (pq, qp) are used.
// vis_obs, vis_model: (n_bl * n_freq, 4) complex.
// freqs: (n_freq,) Hz.
SolverResult solve_KC(const SolverInput& inp, const SolverOptions& opts);

// ---------------------------------------------------------------------------
// solve_CP — cross-phase calibration
// ---------------------------------------------------------------------------
// Solves for 1 global phase phi. J = diag(1, exp(i*phi)).
// Only cross-hand correlations (pq, qp) are used.
// vis_obs, vis_model: (n_bl, 4) complex.
// freqs: ignored.
SolverResult solve_CP(const SolverInput& inp, const SolverOptions& opts);

}  // namespace boa

#endif  // BOA_API_HPP
