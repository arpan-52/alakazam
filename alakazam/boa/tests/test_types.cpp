// test_types.cpp — Verify Mat2 algebra: multiply, hermitian, inverse, identity.
//
// All tests run on the host. We construct known 2x2 matrices and check
// the results against hand-computed values.

#include <boa/types.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace boa;

// Check that two complex values are close.
static bool close(complex_type a, complex_type b, double tol = 1e-12) {
    return Kokkos::abs(a - b) < tol;
}

// Check that two Mat2 matrices are element-wise close.
static bool mat2_close(const Mat2& A, const Mat2& B, double tol = 1e-12) {
    for (int i = 0; i < 4; ++i) {
        if (!close(A.m[i], B.m[i], tol)) return false;
    }
    return true;
}

static int n_pass = 0;
static int n_fail = 0;

static void check(bool cond, const char* name) {
    if (cond) {
        printf("  PASS: %s\n", name);
        ++n_pass;
    } else {
        printf("  FAIL: %s\n", name);
        ++n_fail;
    }
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        printf("=== test_types ===\n");

        // --- Identity ---
        Mat2 I = Mat2::identity();
        check(close(I(0, 0), complex_type(1, 0)), "identity (0,0)");
        check(close(I(0, 1), complex_type(0, 0)), "identity (0,1)");
        check(close(I(1, 0), complex_type(0, 0)), "identity (1,0)");
        check(close(I(1, 1), complex_type(1, 0)), "identity (1,1)");

        // --- Multiply by identity ---
        Mat2 A;
        A(0, 0) = complex_type(1.0, 2.0);
        A(0, 1) = complex_type(3.0, 4.0);
        A(1, 0) = complex_type(5.0, 6.0);
        A(1, 1) = complex_type(7.0, 8.0);

        Mat2 AI = mat2_multiply(A, I);
        check(mat2_close(AI, A), "A * I == A");

        Mat2 IA = mat2_multiply(I, A);
        check(mat2_close(IA, A), "I * A == A");

        // --- Known multiply ---
        // A = [[1+2i, 3+4i], [5+6i, 7+8i]]
        // B = [[1, 0], [0, 1]] = I → already tested
        // Use a non-trivial B:
        // B = [[1+0i, 1+0i], [0+0i, 1+0i]]
        Mat2 B;
        B(0, 0) = complex_type(1, 0);
        B(0, 1) = complex_type(1, 0);
        B(1, 0) = complex_type(0, 0);
        B(1, 1) = complex_type(1, 0);

        Mat2 AB = mat2_multiply(A, B);
        // AB(0,0) = A(0,0)*1 + A(0,1)*0 = 1+2i
        // AB(0,1) = A(0,0)*1 + A(0,1)*1 = (1+2i)+(3+4i) = 4+6i
        // AB(1,0) = A(1,0)*1 + A(1,1)*0 = 5+6i
        // AB(1,1) = A(1,0)*1 + A(1,1)*1 = (5+6i)+(7+8i) = 12+14i
        check(close(AB(0, 0), complex_type(1, 2)), "multiply (0,0)");
        check(close(AB(0, 1), complex_type(4, 6)), "multiply (0,1)");
        check(close(AB(1, 0), complex_type(5, 6)), "multiply (1,0)");
        check(close(AB(1, 1), complex_type(12, 14)), "multiply (1,1)");

        // --- Hermitian (conjugate transpose) ---
        // A† = [[conj(A(0,0)), conj(A(1,0))], [conj(A(0,1)), conj(A(1,1))]]
        Mat2 Ah = mat2_hermitian(A);
        check(close(Ah(0, 0), Kokkos::conj(A(0, 0))), "hermitian (0,0)");
        check(close(Ah(0, 1), Kokkos::conj(A(1, 0))), "hermitian (0,1)");
        check(close(Ah(1, 0), Kokkos::conj(A(0, 1))), "hermitian (1,0)");
        check(close(Ah(1, 1), Kokkos::conj(A(1, 1))), "hermitian (1,1)");

        // --- Inverse ---
        // Use a well-conditioned matrix.
        // C = [[2+0i, 1+0i], [0+0i, 3+0i]]
        // det = 6, C^{-1} = [[3/6, -1/6], [0, 2/6]] = [[0.5, -1/6], [0, 1/3]]
        Mat2 C;
        C(0, 0) = complex_type(2, 0);
        C(0, 1) = complex_type(1, 0);
        C(1, 0) = complex_type(0, 0);
        C(1, 1) = complex_type(3, 0);

        Mat2 Cinv = mat2_inverse(C);
        check(close(Cinv(0, 0), complex_type(0.5, 0)), "inverse (0,0)");
        check(close(Cinv(0, 1), complex_type(-1.0 / 6.0, 0)), "inverse (0,1)");
        check(close(Cinv(1, 0), complex_type(0, 0)), "inverse (1,0)");
        check(close(Cinv(1, 1), complex_type(1.0 / 3.0, 0)), "inverse (1,1)");

        // Check C * C^{-1} = I.
        Mat2 CCinv = mat2_multiply(C, Cinv);
        check(mat2_close(CCinv, Mat2::identity()), "C * C^{-1} == I");

        // --- Frobenius norm ---
        // ||I||_F^2 = 1+0+0+1 = 2
        check(std::abs(mat2_frob_norm_sq(I) - 2.0) < 1e-12, "frob_norm_sq(I) == 2");

        printf("\n%d passed, %d failed\n", n_pass, n_fail);
    }
    Kokkos::finalize();
    return n_fail > 0 ? 1 : 0;
}
