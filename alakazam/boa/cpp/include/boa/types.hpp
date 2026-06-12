// boa/types.hpp — Foundation types for the boa calibration solver.
//
// This header defines:
//   - Scalar type aliases (real_type, complex_type)
//   - Mat2: a 2x2 complex matrix for Jones algebra
//   - SolverInput: everything the solver receives from the pipeline
//   - SolverResult: everything the solver returns
//   - SolverOptions: user-configurable solver knobs
//   - LinearResult: what an inner linear solver returns
//
// All Kokkos::View types use the default memory space, which is determined
// at compile time by how Kokkos was built. If Kokkos was built with CUDA,
// the default space is GPU memory. If built with OpenMP or Serial, it's
// host memory. The same source code works in all cases.

#ifndef BOA_TYPES_HPP
#define BOA_TYPES_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

namespace boa {

// ---------------------------------------------------------------------------
// Scalar aliases
// ---------------------------------------------------------------------------

// All numerical work in double precision.
using real_type    = double;
using complex_type = Kokkos::complex<double>;

// Shorthand for the default execution and memory spaces.
// These are set by the Kokkos build configuration — not by us.
using exec_space   = Kokkos::DefaultExecutionSpace;
using mem_space    = exec_space::memory_space;
using host_space   = Kokkos::HostSpace;

// ---------------------------------------------------------------------------
// Mat2 — 2x2 complex matrix in row-major order
// ---------------------------------------------------------------------------
// Stored as 4 complex values: m[0]=J(0,0), m[1]=J(0,1), m[2]=J(1,0), m[3]=J(1,1).
//
// KOKKOS_INLINE_FUNCTION means this method can run on both host (CPU)
// and device (GPU). It's the Kokkos equivalent of __host__ __device__.

struct Mat2 {
    complex_type m[4];

    // Element access: M(row, col)
    KOKKOS_INLINE_FUNCTION
    complex_type& operator()(int r, int c) { return m[r * 2 + c]; }

    KOKKOS_INLINE_FUNCTION
    const complex_type& operator()(int r, int c) const { return m[r * 2 + c]; }

    // Return the 2x2 identity matrix.
    KOKKOS_INLINE_FUNCTION
    static Mat2 identity() {
        Mat2 I;
        I.m[0] = complex_type(1.0, 0.0);
        I.m[1] = complex_type(0.0, 0.0);
        I.m[2] = complex_type(0.0, 0.0);
        I.m[3] = complex_type(1.0, 0.0);
        return I;
    }

    // Return the zero matrix.
    KOKKOS_INLINE_FUNCTION
    static Mat2 zero() {
        Mat2 Z;
        Z.m[0] = complex_type(0.0, 0.0);
        Z.m[1] = complex_type(0.0, 0.0);
        Z.m[2] = complex_type(0.0, 0.0);
        Z.m[3] = complex_type(0.0, 0.0);
        return Z;
    }
};

// C = A * B  (2x2 complex matrix multiply)
KOKKOS_INLINE_FUNCTION
Mat2 mat2_multiply(const Mat2& A, const Mat2& B) {
    Mat2 C;
    C(0, 0) = A(0, 0) * B(0, 0) + A(0, 1) * B(1, 0);
    C(0, 1) = A(0, 0) * B(0, 1) + A(0, 1) * B(1, 1);
    C(1, 0) = A(1, 0) * B(0, 0) + A(1, 1) * B(1, 0);
    C(1, 1) = A(1, 0) * B(0, 1) + A(1, 1) * B(1, 1);
    return C;
}

// B = A†  (conjugate transpose)
KOKKOS_INLINE_FUNCTION
Mat2 mat2_hermitian(const Mat2& A) {
    Mat2 B;
    B(0, 0) = Kokkos::conj(A(0, 0));
    B(0, 1) = Kokkos::conj(A(1, 0));
    B(1, 0) = Kokkos::conj(A(0, 1));
    B(1, 1) = Kokkos::conj(A(1, 1));
    return B;
}

// C = A * B†  (multiply A by conjugate transpose of B)
// This is the most common operation in the RIME: J_i * V_M * J_j†
KOKKOS_INLINE_FUNCTION
Mat2 mat2_multiply_hermitian(const Mat2& A, const Mat2& B) {
    return mat2_multiply(A, mat2_hermitian(B));
}

// B = A^{-1}  (2x2 inverse via Cramer's rule)
// Returns zero matrix if singular.
KOKKOS_INLINE_FUNCTION
Mat2 mat2_inverse(const Mat2& A) {
    complex_type det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    real_type abs_det = Kokkos::abs(det);

    if (abs_det < 1.0e-30) {
        return Mat2::zero();
    }

    complex_type inv_det = complex_type(1.0, 0.0) / det;
    Mat2 B;
    B(0, 0) =  A(1, 1) * inv_det;
    B(0, 1) = -A(0, 1) * inv_det;
    B(1, 0) = -A(1, 0) * inv_det;
    B(1, 1) =  A(0, 0) * inv_det;
    return B;
}

// Frobenius norm squared: ||A||_F^2 = sum |A(r,c)|^2
KOKKOS_INLINE_FUNCTION
real_type mat2_frob_norm_sq(const Mat2& A) {
    real_type s = 0.0;
    for (int i = 0; i < 4; ++i) {
        s += A.m[i].real() * A.m[i].real() + A.m[i].imag() * A.m[i].imag();
    }
    return s;
}

// ---------------------------------------------------------------------------
// View type aliases
// ---------------------------------------------------------------------------
// Kokkos::View is the fundamental array type. The template parameters are:
//   1. Data type and layout (e.g. real_type* = 1D array of doubles)
//   2. Memory space (where the data lives: host RAM or GPU memory)
//
// Using the default memory space means the same code compiles for CPU or GPU.

using view_1d_real    = Kokkos::View<real_type*,    mem_space>;
using view_1d_complex = Kokkos::View<complex_type*, mem_space>;
using view_1d_int     = Kokkos::View<int*,          mem_space>;
using view_2d_real    = Kokkos::View<real_type**,    mem_space>;
using view_2d_complex = Kokkos::View<complex_type**, mem_space>;
using view_1d_mat2    = Kokkos::View<Mat2*,          mem_space>;

// Host-side mirrors for copying data between host and device.
// Kokkos::create_mirror_view(device_view) gives a host-accessible copy.
using host_view_1d_real    = Kokkos::View<real_type*,    host_space>;
using host_view_1d_complex = Kokkos::View<complex_type*, host_space>;
using host_view_1d_int     = Kokkos::View<int*,          host_space>;
using host_view_2d_real    = Kokkos::View<real_type**,    host_space>;
using host_view_2d_complex = Kokkos::View<complex_type**, host_space>;

// ---------------------------------------------------------------------------
// SolverInput — everything the solver receives from the pipeline
// ---------------------------------------------------------------------------
// The pipeline (Python side) packs observed/model visibilities, antenna
// indices, and frequencies into numpy arrays. The pybind11 layer converts
// them to Kokkos::Views and fills this struct.

struct SolverInput {
    // Observed visibilities: (n_bl, 4) complex — flattened 2x2 per baseline.
    // Layout: [V(0,0), V(0,1), V(1,0), V(1,1)] for each baseline.
    view_2d_complex vis_obs;

    // Model visibilities: same shape as vis_obs.
    view_2d_complex vis_model;

    // Antenna indices: (n_bl,) — which two antennas form each baseline.
    // ant1[b] < ant2[b] by convention.
    view_1d_int ant1;
    view_1d_int ant2;

    // Frequencies in Hz: (n_freq,).
    // Non-empty only for freq-dependent solvers (K, KC).
    // Empty (extent 0) for freq-independent solvers (G, D, CP).
    view_1d_real freqs;

    // Optional initial parameter guess. If extent(0) > 0, solve_lm uses it
    // instead of Problem::initial_params(). Layout is solver-specific (see api.cpp).
    view_1d_real init_params;

    // Number of antennas (including ref_ant).
    int n_ant;

    // Reference antenna index (0-based).
    // This antenna's parameters are excluded from optimization entirely.
    int ref_ant;
};

// ---------------------------------------------------------------------------
// SolverResult — everything the solver returns to the pipeline
// ---------------------------------------------------------------------------

struct SolverResult {
    // Solved Jones matrices: (n_ant, 4) complex — flattened 2x2 per antenna.
    // Valid for G and D solvers. Set to identity for K, KC, CP.
    view_2d_complex jones;

    // Raw parameter vector — canonical output for all solver types.
    // G: [amp_p, phase_p, amp_q, phase_q] * (n_ant-1) + [amp_p_ref, amp_q_ref]
    // K: [tau_p, tau_q] * (n_ant-1)
    // D: [Re(d_pq), Im(d_pq), Re(d_qp), Im(d_qp)] * (n_ant-1) + [Re, Im of d_qp_ref]
    // KC: [tau_ns]   CP: [phi_rad]
    view_1d_real params;

    // Final residual cost: 0.5 * ||r||^2
    real_type cost;

    // Number of LM outer iterations taken.
    int n_iter;

    // Whether the solver converged (cost or parameter tolerance met).
    bool converged;
};

// ---------------------------------------------------------------------------
// SolverOptions — user-configurable knobs
// ---------------------------------------------------------------------------

// Which inner linear solver to use for the LM update step.
enum class LinearSolverType {
    CG,              // Conjugate gradient on normal equations J^T J δ = -J^T r
    LSQR,            // Paige-Saunders LSQR — works directly on J, better conditioned
    DENSE_CHOLESKY   // Form J^T J explicitly, Cholesky factorize — reference/validation path
};

struct SolverOptions {
    // Maximum number of LM outer iterations.
    int max_iter = 100;

    // Convergence tolerance on the relative residual reduction.
    real_type tol = 1.0e-10;

    // Which inner linear solver to use.
    LinearSolverType linear_solver = LinearSolverType::LSQR;

    // Maximum iterations for the inner linear solver (CG or LSQR).
    int linear_max_iter = 200;

    // Convergence tolerance for the inner linear solver.
    real_type linear_tol = 1.0e-8;

    // Initial LM damping parameter λ.
    // Larger λ → more like gradient descent (safe but slow).
    // Smaller λ → more like Gauss-Newton (fast but may diverge).
    real_type lm_lambda_init = 1.0e-3;
};

// ---------------------------------------------------------------------------
// LinearResult — what an inner linear solver returns
// ---------------------------------------------------------------------------

struct LinearResult {
    // Number of iterations taken (1 for dense Cholesky).
    int iters;

    // Relative residual norm at convergence.
    real_type residual_norm;

    // 0.5 * (||J delta||^2 + lambda * ||delta||^2) — used by LM gain ratio.
    real_type predicted_reduction;
};

}  // namespace boa

#endif  // BOA_TYPES_HPP
