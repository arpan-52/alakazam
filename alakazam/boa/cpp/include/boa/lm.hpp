// boa/lm.hpp — Levenberg-Marquardt outer loop, templated on Problem type.
//
// The LM algorithm minimizes the nonlinear least-squares objective
//   F(x) = 0.5 * ||r(x)||^2
// by iterating:
//   1. Build residual r at current params.
//   2. Build Jacobian J at current params.
//   3. Solve (J^T J + lambda I) delta = -J^T r  (the LM inner step).
//   4. Compute gain ratio: actual_reduction / predicted_reduction.
//   5. Accept or reject the step, update lambda accordingly.
//   6. Check convergence.
//
// The Problem type must provide:
//   int n_residuals() const
//   int n_params() const
//   void build_residual(const view_1d_real& params, view_1d_real& r) const
//   void fill_jacobian(const view_1d_real& params, crs_matrix_type& J) const
//   crs_matrix_type build_csr_pattern() const
//   view_1d_real initial_params() const
//   view_2d_complex params_to_jones(const view_1d_real& params) const
//
// The template is header-only because the Kokkos device kernels called inside
// Problem::build_residual and Problem::fill_jacobian must be compiled with
// the same CUDA/OpenMP toolchain as the rest of the translation unit.

#ifndef BOA_LM_HPP
#define BOA_LM_HPP

#include <boa/types.hpp>
#include <boa/csr_pattern.hpp>
#include <boa/residual.hpp>
#include <boa/linear_solvers.hpp>
#include <KokkosBlas.hpp>
#include <cmath>

namespace boa {

// ---------------------------------------------------------------------------
// solve_lm — generic Levenberg-Marquardt solver
// ---------------------------------------------------------------------------
// Problem: a struct implementing the interface described above.
// Returns SolverResult with jones (for G/D), params (always), cost, n_iter.

template <typename Problem>
SolverResult solve_lm(Problem& prob, const SolverOptions& opts,
                      view_1d_real init_override = view_1d_real())
{
    const int n_res   = prob.n_residuals();
    const int n_param = prob.n_params();

    // Build the CSR sparsity pattern once. Only values change across iterations.
    crs_matrix_type J = prob.build_csr_pattern();

    // Initial parameter vector — caller override takes precedence.
    view_1d_real params = (init_override.extent(0) > 0)
                          ? init_override
                          : prob.initial_params();

    // Allocate working vectors.
    view_1d_real r        ("r",        n_res);
    view_1d_real r_trial  ("r_trial",  n_res);
    view_1d_real delta    ("delta",    n_param);
    view_1d_real p_trial  ("p_trial",  n_param);

    // Evaluate the initial residual.
    prob.build_residual(params, r);
    real_type cost = 0.5 * residual_norm_sq(r);

    real_type lambda  = opts.lm_lambda_init;
    bool converged    = false;
    int  n_iter       = 0;

    // Check if initial guess already satisfies tolerance (avoids false non-convergence
    // when the caller supplies a near-perfect init_params and the Jacobian is tiny).
    if (std::sqrt(2.0 * cost) < opts.tol) {
        converged = true;
    }

    for (int iter = 0; iter < opts.max_iter && !converged; ++iter) {
        ++n_iter;

        // Fill Jacobian values (pattern was built once above).
        prob.fill_jacobian(params, J);

        // Solve linear system for the LM step delta.
        LinearResult lin;
        switch (opts.linear_solver) {
            case LinearSolverType::CG:
                lin = solve_cg(J, r, delta, lambda,
                               opts.linear_max_iter, opts.linear_tol);
                break;
            case LinearSolverType::DENSE_CHOLESKY:
                lin = solve_dense_cholesky(J, r, delta, lambda);
                break;
            default:  // LSQR
                lin = solve_lsqr(J, r, delta, lambda,
                                 opts.linear_max_iter, opts.linear_tol);
                break;
        }

        // Guard: if predicted reduction is too small, increase lambda and retry.
        if (lin.predicted_reduction < 1.0e-30) {
            lambda *= 10.0;
            continue;
        }

        // Parameter-step (xtol) convergence: at an optimum with a nonzero
        // residual floor (unmodelled terms in the data), the gradient is
        // zero, delta ~ 0, and every trial step is rejected — without this
        // check the loop would burn max_iter and report false non-convergence.
        // Kept deliberately tight (1e-12): weakly-constrained directions
        // (e.g. the free ref-antenna d_qp in the D solver) legitimately
        // converge through many small steps and must not be cut short.
        const real_type delta_norm  = std::sqrt(KokkosBlas::dot(delta, delta));
        const real_type params_norm = std::sqrt(KokkosBlas::dot(params, params));
        if (delta_norm <= 1.0e-12 * (params_norm + 1.0e-12)) {
            converged = true;
            break;
        }

        // Trial step: p_trial = params + delta.
        Kokkos::deep_copy(p_trial, params);
        KokkosBlas::axpy(real_type(1.0), delta, p_trial);

        // Evaluate trial residual.
        prob.build_residual(p_trial, r_trial);
        const real_type cost_trial = 0.5 * residual_norm_sq(r_trial);

        // Gain ratio: actual / predicted reduction.
        const real_type actual_reduction = cost - cost_trial;
        const real_type gain_ratio = actual_reduction / lin.predicted_reduction;

        if (gain_ratio > 0.0) {
            // Accept step.
            Kokkos::deep_copy(params, p_trial);
            Kokkos::deep_copy(r,      r_trial);
            cost = cost_trial;

            // Reduce lambda for a good step (gain_ratio > 0.75 → decrease faster).
            if (gain_ratio > 0.75)
                lambda = std::max(lambda / 3.0, 1.0e-16);

            // Convergence: residual norm below tolerance (absolute),
            // or relative cost reduction is negligible.
            if (std::sqrt(2.0 * cost) < opts.tol ||
                actual_reduction < opts.tol * (cost + 1.0e-30)) {
                converged = true;
                break;
            }
        } else {
            // Reject step, increase lambda (more gradient-descent-like).
            lambda = std::min(lambda * 10.0, 1.0e16);
        }
    }

    SolverResult result;
    result.jones     = prob.params_to_jones(params);
    result.params    = params;
    result.cost      = cost;
    result.n_iter    = n_iter;
    result.converged = converged;
    return result;
}

}  // namespace boa

#endif  // BOA_LM_HPP
