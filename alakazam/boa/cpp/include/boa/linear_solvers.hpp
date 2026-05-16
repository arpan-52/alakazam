// boa/linear_solvers.hpp — Three inner linear solvers for the LM step.
//
// Each solver computes the parameter update delta that (approximately)
// solves the damped normal equations:
//
//   (J^T J + lambda * I) delta = -J^T r
//
// This is the LM inner step. Three implementations are provided:
//
//   solve_dense_cholesky — form J^T J explicitly on host, Cholesky factorize.
//     Pure C++, no LAPACK. Use for small problems (n_params < ~100) and
//     as the reference oracle when validating CG and LSQR.
//
//   solve_cg — Conjugate Gradient on the normal equations.
//     Uses KokkosSparse::spmv + KokkosBlas. Runs on device.
//     Good for medium-sized problems.
//
//   solve_lsqr — Paige-Saunders LSQR on the augmented system.
//     min ||J delta + r||^2 + lambda * ||delta||^2
//     Equivalent to CG on normal equations but numerically superior
//     because it avoids squaring the condition number.
//
// All three solvers write the result into the pre-allocated `delta` view
// and return a LinearResult with diagnostics.

#ifndef BOA_LINEAR_SOLVERS_HPP
#define BOA_LINEAR_SOLVERS_HPP

#include <boa/types.hpp>
#include <boa/csr_pattern.hpp>

namespace boa {

// Solve (J^T J + lambda I) delta = -J^T r using dense Cholesky (host-side).
// Forms J^T J in O(nnz * n_params) time, factorizes in O(n_params^3).
// Only practical for n_params < ~100, but numerically exact and always converges.
LinearResult solve_dense_cholesky(
    const crs_matrix_type& J,
    const view_1d_real& r,
    view_1d_real& delta,
    real_type lambda);

// Solve (J^T J + lambda I) delta = -J^T r using Conjugate Gradient.
// Iterative: each iteration does two SpMVs (J*p and J^T*(J*p)).
// Converges in at most n_params iterations for exact arithmetic.
LinearResult solve_cg(
    const crs_matrix_type& J,
    const view_1d_real& r,
    view_1d_real& delta,
    real_type lambda,
    int max_iter,
    real_type tol);

// Solve min ||J delta + r||^2 + lambda ||delta||^2 using LSQR (Paige-Saunders 1982).
// Works directly on J (not J^T J), so condition number is sqrt of CG's.
// Each iteration does two SpMVs: J*v and J^T*u.
LinearResult solve_lsqr(
    const crs_matrix_type& J,
    const view_1d_real& r,
    view_1d_real& delta,
    real_type lambda,
    int max_iter,
    real_type tol);

}  // namespace boa

#endif  // BOA_LINEAR_SOLVERS_HPP
