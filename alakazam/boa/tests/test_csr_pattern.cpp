// test_csr_pattern.cpp — Verify CSR sparsity patterns for all problem types.
//
// Uses a small 4-antenna problem with ref_ant=0 to check:
//   - Correct number of rows and columns
//   - Correct nnz per row (accounting for ref_ant exclusion)
//   - Column indices point to the right antenna parameter blocks
//
// Antenna pairs for N=4: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3) → 6 baselines.
// With ref_ant=0: antennas 1,2,3 have parameters. Antenna 0 is excluded.
// For baselines involving antenna 0, only the other antenna's columns appear.

#include <boa/types.hpp>
#include <boa/csr_pattern.hpp>
#include <cstdio>
#include <cstdlib>

using namespace boa;

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
        printf("=== test_csr_pattern ===\n");

        const int n_ant = 4;
        const int ref_ant = 0;
        const int n_bl = n_ant * (n_ant - 1) / 2;  // 6 baselines

        // Build antenna pairs on host.
        host_view_1d_int h_ant1("h_ant1", n_bl);
        host_view_1d_int h_ant2("h_ant2", n_bl);
        int b = 0;
        for (int i = 0; i < n_ant; ++i) {
            for (int j = i + 1; j < n_ant; ++j) {
                h_ant1(b) = i;
                h_ant2(b) = j;
                b++;
            }
        }
        // Baselines: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)

        // ---------------------------------------------------------------
        // Test 1: AntennaMap for G solver (params_per_ant = 4)
        // ---------------------------------------------------------------
        printf("\n--- AntennaMap (G solver, ppa=4) ---\n");
        {
            AntennaMap amap = build_antenna_map(n_ant, ref_ant, 4);

            // ref_ant=0 excluded → 3 antennas × 4 params = 12 total params
            check(amap.n_params == 12, "n_params == 12");
            check(amap.h_ant_to_param(0) == -1, "ant 0 excluded");
            check(amap.h_ant_to_param(1) == 0, "ant 1 → offset 0");
            check(amap.h_ant_to_param(2) == 4, "ant 2 → offset 4");
            check(amap.h_ant_to_param(3) == 8, "ant 3 → offset 8");
        }

        // ---------------------------------------------------------------
        // Test 2: CSR pattern for G solver (diagonal, freq-independent)
        // ---------------------------------------------------------------
        printf("\n--- CSR diagonal (G solver, ppa=4, n_freq=1) ---\n");
        {
            AntennaMap amap = build_antenna_map(n_ant, ref_ant, 4);
            crs_matrix_type J = build_csr_diagonal(h_ant1, h_ant2, amap, 1);

            // 6 baselines × 4 residuals/bl = 24 rows
            check(J.numRows() == 24, "numRows == 24");
            // 12 columns (3 active antennas × 4 params)
            check(J.numCols() == 12, "numCols == 12");

            // Copy row_ptr to host to inspect.
            auto h_row_ptr = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), J.graph.row_map);
            auto h_col_ind = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), J.graph.entries);

            // Baseline (0,1): ant 0 is ref → only ant 1 cols → 4 nnz/row
            int nnz_row0 = h_row_ptr(1) - h_row_ptr(0);
            check(nnz_row0 == 4, "bl (0,1): nnz/row == 4 (ref excluded)");

            // Baseline (1,2): neither is ref → 8 nnz/row
            // Baseline (1,2) is baseline index 3, first row = 3*4 = 12
            int nnz_row12 = h_row_ptr(13) - h_row_ptr(12);
            check(nnz_row12 == 8, "bl (1,2): nnz/row == 8 (both active)");

            // Check col_ind for baseline (1,2), row 12:
            // Should have cols for ant 1 (offset 0: cols 0,1,2,3)
            // then ant 2 (offset 4: cols 4,5,6,7)
            int start = h_row_ptr(12);
            bool cols_correct = true;
            for (int k = 0; k < 4; ++k) {
                if (h_col_ind(start + k) != k) cols_correct = false;        // ant 1: 0,1,2,3
                if (h_col_ind(start + 4 + k) != 4 + k) cols_correct = false; // ant 2: 4,5,6,7
            }
            check(cols_correct, "bl (1,2): col_ind correct");
        }

        // ---------------------------------------------------------------
        // Test 3: CSR pattern for K solver (diagonal, freq-dependent)
        // ---------------------------------------------------------------
        printf("\n--- CSR diagonal (K solver, ppa=2, n_freq=8) ---\n");
        {
            int n_freq = 8;
            AntennaMap amap = build_antenna_map(n_ant, ref_ant, 2);
            crs_matrix_type J = build_csr_diagonal(h_ant1, h_ant2, amap, n_freq);

            // 6 baselines × 4 residuals/freq × 8 freq = 192 rows
            check(J.numRows() == 192, "numRows == 192");
            // 3 active antennas × 2 params = 6 columns
            check(J.numCols() == 6, "numCols == 6");

            auto h_row_ptr = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), J.graph.row_map);

            // Baseline (1,3): bl index 4, first row = 4 * 32 = 128
            // Neither is ref → 4 nnz/row (2 for ant_i + 2 for ant_j)
            int nnz = h_row_ptr(129) - h_row_ptr(128);
            check(nnz == 4, "bl (1,3) freq 0: nnz/row == 4");
        }

        // ---------------------------------------------------------------
        // Test 4: CSR pattern for D solver (full 2x2)
        // ---------------------------------------------------------------
        printf("\n--- CSR full 2x2 (D solver, ppa=4) ---\n");
        {
            AntennaMap amap = build_antenna_map(n_ant, ref_ant, 4);
            crs_matrix_type J = build_csr_full_2x2(h_ant1, h_ant2, amap);

            // 6 baselines × 8 residuals/bl = 48 rows
            check(J.numRows() == 48, "numRows == 48");
            check(J.numCols() == 12, "numCols == 12");

            auto h_row_ptr = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), J.graph.row_map);

            // Baseline (0,3): ant 0 is ref → 4 nnz/row
            // bl index 2, first row = 2*8 = 16
            int nnz = h_row_ptr(17) - h_row_ptr(16);
            check(nnz == 4, "bl (0,3): nnz/row == 4 (ref excluded)");

            // Baseline (2,3): bl index 5, first row = 5*8 = 40
            int nnz2 = h_row_ptr(41) - h_row_ptr(40);
            check(nnz2 == 8, "bl (2,3): nnz/row == 8 (both active)");
        }

        // ---------------------------------------------------------------
        // Test 5: CSR pattern for global solver (KC/CP)
        // ---------------------------------------------------------------
        printf("\n--- CSR global (KC/CP, 1 param) ---\n");
        {
            int n_residuals = 48;  // e.g. 6 bl × 8 freq
            crs_matrix_type J = build_csr_global(n_residuals, 1);

            check(J.numRows() == 48, "numRows == 48");
            check(J.numCols() == 1, "numCols == 1");

            auto h_row_ptr = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), J.graph.row_map);
            auto h_col_ind = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), J.graph.entries);

            // Every row has 1 nnz, column 0.
            bool all_correct = true;
            for (int i = 0; i < 48; ++i) {
                if (h_row_ptr(i + 1) - h_row_ptr(i) != 1) all_correct = false;
                if (h_col_ind(h_row_ptr(i)) != 0) all_correct = false;
            }
            check(all_correct, "every row: 1 nnz at col 0");
        }

        printf("\n%d passed, %d failed\n", n_pass, n_fail);
    }
    Kokkos::finalize();
    return n_fail > 0 ? 1 : 0;
}
