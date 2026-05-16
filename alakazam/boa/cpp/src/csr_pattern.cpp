// boa/csr_pattern.cpp — Build the fixed CSR sparsity patterns.
//
// All pattern construction happens on the host (CPU), because it's a one-time
// setup cost and involves irregular indexing. The resulting CrsMatrix is then
// usable on whatever execution space Kokkos was built with.

#include <boa/csr_pattern.hpp>
#include <vector>

namespace boa {

// ---------------------------------------------------------------------------
// build_antenna_map
// ---------------------------------------------------------------------------
// Creates the mapping from antenna index to parameter-vector offset.
// ref_ant is excluded — its parameters don't appear in the Jacobian.
//
// Example with n_ant=4, ref_ant=1, params_per_ant=4:
//   ant 0 → offset 0  (params 0,1,2,3)
//   ant 1 → -1         (excluded)
//   ant 2 → offset 4  (params 4,5,6,7)
//   ant 3 → offset 8  (params 8,9,10,11)
//   n_params = 12

AntennaMap build_antenna_map(int n_ant, int ref_ant, int params_per_ant) {
    AntennaMap amap;
    amap.n_ant = n_ant;
    amap.ref_ant = ref_ant;
    amap.params_per_ant = params_per_ant;

    // Build on host first.
    amap.h_ant_to_param = host_view_1d_int("h_ant_to_param", n_ant);
    int offset = 0;
    for (int a = 0; a < n_ant; ++a) {
        if (a == ref_ant) {
            amap.h_ant_to_param(a) = -1;  // excluded
        } else {
            amap.h_ant_to_param(a) = offset;
            offset += params_per_ant;
        }
    }
    amap.n_params = offset;

    // Copy to device.
    amap.ant_to_param = view_1d_int("ant_to_param", n_ant);
    Kokkos::deep_copy(amap.ant_to_param, amap.h_ant_to_param);

    return amap;
}

// ---------------------------------------------------------------------------
// build_csr_diagonal
// ---------------------------------------------------------------------------
// For diagonal solvers (G, K), the RIME is:
//   r_pp = V_obs_pp - g_i_p * V_M_pp * conj(g_j_p)
//   r_qq = V_obs_qq - g_i_q * V_M_qq * conj(g_j_q)
//
// Each baseline produces residuals_per_bl_per_freq = 4 residual rows per freq:
//   row 0: Re(r_pp)
//   row 1: Im(r_pp)
//   row 2: Re(r_qq)
//   row 3: Im(r_qq)
//
// Each row depends on ant_i's params and ant_j's params.
// If ant_i == ref_ant, only ant_j's columns appear (and vice versa).
//
// For G: params_per_ant=4 → max nnz_per_row = 8
// For K: params_per_ant=2 → max nnz_per_row = 4

crs_matrix_type build_csr_diagonal(
    const host_view_1d_int& h_ant1,
    const host_view_1d_int& h_ant2,
    const AntennaMap& amap,
    int n_freq)
{
    const int n_bl = h_ant1.extent(0);
    const int ppa = amap.params_per_ant;
    const int res_per_bl = 4 * n_freq;  // 4 real residuals per freq per baseline
    const int n_rows = n_bl * res_per_bl;
    const int n_cols = amap.n_params;

    // First pass: count nnz per row to build row_ptr.
    // Each row has ppa columns for ant_i (if not ref) + ppa for ant_j (if not ref).
    std::vector<int> row_ptr(n_rows + 1, 0);
    int total_nnz = 0;

    for (int b = 0; b < n_bl; ++b) {
        int ai = h_ant1(b);
        int aj = h_ant2(b);
        // How many nonzeros per row for this baseline?
        int nnz_row = 0;
        if (amap.h_ant_to_param(ai) >= 0) nnz_row += ppa;
        if (amap.h_ant_to_param(aj) >= 0) nnz_row += ppa;

        for (int f = 0; f < n_freq; ++f) {
            for (int r = 0; r < 4; ++r) {
                int row = b * res_per_bl + f * 4 + r;
                row_ptr[row + 1] = nnz_row;
            }
        }
        total_nnz += res_per_bl * nnz_row;
    }

    // Prefix sum to get actual row_ptr offsets.
    for (int i = 0; i < n_rows; ++i) {
        row_ptr[i + 1] += row_ptr[i];
    }

    // Second pass: fill col_ind.
    std::vector<int> col_ind(total_nnz);

    for (int b = 0; b < n_bl; ++b) {
        int ai = h_ant1(b);
        int aj = h_ant2(b);
        int off_i = amap.h_ant_to_param(ai);  // -1 if ref
        int off_j = amap.h_ant_to_param(aj);

        for (int f = 0; f < n_freq; ++f) {
            for (int r = 0; r < 4; ++r) {
                int row = b * res_per_bl + f * 4 + r;
                int pos = row_ptr[row];

                // Columns for ant_i (if not ref_ant).
                if (off_i >= 0) {
                    for (int k = 0; k < ppa; ++k) {
                        col_ind[pos++] = off_i + k;
                    }
                }
                // Columns for ant_j (if not ref_ant).
                if (off_j >= 0) {
                    for (int k = 0; k < ppa; ++k) {
                        col_ind[pos++] = off_j + k;
                    }
                }
            }
        }
    }

    // Copy row_ptr and col_ind into Kokkos host views.
    host_row_map_type h_row_ptr("h_row_ptr", n_rows + 1);
    host_entries_type h_col_ind("h_col_ind", total_nnz);
    for (int i = 0; i <= n_rows; ++i) h_row_ptr(i) = row_ptr[i];
    for (int i = 0; i < total_nnz; ++i) h_col_ind(i) = col_ind[i];

    // Create device-side views and copy.
    Kokkos::View<int*, mem_space> d_row_ptr("d_row_ptr", n_rows + 1);
    Kokkos::View<int*, mem_space> d_col_ind("d_col_ind", total_nnz);
    view_1d_real d_values("d_values", total_nnz);  // initialized to zero
    Kokkos::deep_copy(d_row_ptr, h_row_ptr);
    Kokkos::deep_copy(d_col_ind, h_col_ind);

    // Construct CrsMatrix from the three arrays.
    // The "J" string is just a label for debugging.
    crs_matrix_type J("J", n_rows, n_cols, total_nnz,
                      d_values, d_row_ptr, d_col_ind);
    return J;
}

// ---------------------------------------------------------------------------
// build_csr_full_2x2
// ---------------------------------------------------------------------------
// For the full 2x2 solver (D - leakage), the RIME is:
//   R = V_obs - J_i * V_M * J_j†   (full 2x2 matrix residual)
//
// Each baseline produces 8 residual rows:
//   Re/Im of R(0,0), R(0,1), R(1,0), R(1,1)
//
// Each row depends on ant_i's params and ant_j's params.
// params_per_ant = 4 for D: Re(d_pq), Im(d_pq), Re(d_qp), Im(d_qp)
// nnz_per_row = up to 8 (4 for ant_i + 4 for ant_j)

crs_matrix_type build_csr_full_2x2(
    const host_view_1d_int& h_ant1,
    const host_view_1d_int& h_ant2,
    const AntennaMap& amap)
{
    const int n_bl = h_ant1.extent(0);
    const int ppa = amap.params_per_ant;
    const int res_per_bl = 8;  // 8 real residuals per baseline (full 2x2)
    const int n_rows = n_bl * res_per_bl;
    const int n_cols = amap.n_params;

    // Count nnz per row.
    std::vector<int> row_ptr(n_rows + 1, 0);
    int total_nnz = 0;

    for (int b = 0; b < n_bl; ++b) {
        int ai = h_ant1(b);
        int aj = h_ant2(b);
        int nnz_row = 0;
        if (amap.h_ant_to_param(ai) >= 0) nnz_row += ppa;
        if (amap.h_ant_to_param(aj) >= 0) nnz_row += ppa;

        for (int r = 0; r < res_per_bl; ++r) {
            int row = b * res_per_bl + r;
            row_ptr[row + 1] = nnz_row;
        }
        total_nnz += res_per_bl * nnz_row;
    }

    // Prefix sum.
    for (int i = 0; i < n_rows; ++i) {
        row_ptr[i + 1] += row_ptr[i];
    }

    // Fill col_ind.
    std::vector<int> col_ind(total_nnz);

    for (int b = 0; b < n_bl; ++b) {
        int ai = h_ant1(b);
        int aj = h_ant2(b);
        int off_i = amap.h_ant_to_param(ai);
        int off_j = amap.h_ant_to_param(aj);

        for (int r = 0; r < res_per_bl; ++r) {
            int row = b * res_per_bl + r;
            int pos = row_ptr[row];

            if (off_i >= 0) {
                for (int k = 0; k < ppa; ++k) {
                    col_ind[pos++] = off_i + k;
                }
            }
            if (off_j >= 0) {
                for (int k = 0; k < ppa; ++k) {
                    col_ind[pos++] = off_j + k;
                }
            }
        }
    }

    // Copy to Kokkos views and construct CrsMatrix.
    host_row_map_type h_row_ptr("h_row_ptr", n_rows + 1);
    host_entries_type h_col_ind("h_col_ind", total_nnz);
    for (int i = 0; i <= n_rows; ++i) h_row_ptr(i) = row_ptr[i];
    for (int i = 0; i < total_nnz; ++i) h_col_ind(i) = col_ind[i];

    Kokkos::View<int*, mem_space> d_row_ptr("d_row_ptr", n_rows + 1);
    Kokkos::View<int*, mem_space> d_col_ind("d_col_ind", total_nnz);
    view_1d_real d_values("d_values", total_nnz);
    Kokkos::deep_copy(d_row_ptr, h_row_ptr);
    Kokkos::deep_copy(d_col_ind, h_col_ind);

    crs_matrix_type J("J", n_rows, n_cols, total_nnz,
                      d_values, d_row_ptr, d_col_ind);
    return J;
}

// ---------------------------------------------------------------------------
// build_csr_global
// ---------------------------------------------------------------------------
// For global-parameter solvers (KC, CP), every residual row depends on the
// same 1 (or few) global parameters. So every row has the same column(s).
// This is a dense column — every row touches column 0.

crs_matrix_type build_csr_global(
    int n_residuals,
    int n_global_params)
{
    const int n_rows = n_residuals;
    const int n_cols = n_global_params;
    const int nnz_per_row = n_global_params;
    const int total_nnz = n_rows * nnz_per_row;

    host_row_map_type h_row_ptr("h_row_ptr", n_rows + 1);
    host_entries_type h_col_ind("h_col_ind", total_nnz);

    for (int i = 0; i <= n_rows; ++i) {
        h_row_ptr(i) = i * nnz_per_row;
    }
    for (int i = 0; i < n_rows; ++i) {
        for (int k = 0; k < nnz_per_row; ++k) {
            h_col_ind(i * nnz_per_row + k) = k;
        }
    }

    Kokkos::View<int*, mem_space> d_row_ptr("d_row_ptr", n_rows + 1);
    Kokkos::View<int*, mem_space> d_col_ind("d_col_ind", total_nnz);
    view_1d_real d_values("d_values", total_nnz);
    Kokkos::deep_copy(d_row_ptr, h_row_ptr);
    Kokkos::deep_copy(d_col_ind, h_col_ind);

    crs_matrix_type J("J", n_rows, n_cols, total_nnz,
                      d_values, d_row_ptr, d_col_ind);
    return J;
}

// ---------------------------------------------------------------------------
// build_csr_leakage_ceres
// ---------------------------------------------------------------------------
// Ceres ref convention: d_pq[ref]=0 (excluded), d_qp[ref] free (2 params at end).
// Param layout: [(n_ant-1)*4 params for non-ref ants] + [2 params for ref d_qp]
// Total columns: amap.n_params + 2
//
// nnz per row:
//   neither ant is ref → 8 (4+4, same as build_csr_full_2x2)
//   one ant is ref     → 6 (2 for ref d_qp + 4 for non-ref ant)

crs_matrix_type build_csr_leakage_ceres(
    const host_view_1d_int& h_ant1,
    const host_view_1d_int& h_ant2,
    const AntennaMap& amap)
{
    const int n_bl        = static_cast<int>(h_ant1.extent(0));
    const int ref_dqp_off = amap.n_params;          // (n_ant-1)*4
    const int n_cols      = amap.n_params + 2;
    const int res_per_bl  = 8;
    const int n_rows      = n_bl * res_per_bl;

    std::vector<int> row_ptr(n_rows + 1, 0);
    int total_nnz = 0;

    for (int b = 0; b < n_bl; ++b) {
        const int off_i = amap.h_ant_to_param(h_ant1(b));
        const int off_j = amap.h_ant_to_param(h_ant2(b));
        // Non-ref ant: 4 cols; ref ant: 2 cols (d_qp only).
        const int nnz_row = (off_i >= 0 ? 4 : 2) + (off_j >= 0 ? 4 : 2);
        for (int r = 0; r < res_per_bl; ++r)
            row_ptr[b * res_per_bl + r + 1] = nnz_row;
        total_nnz += res_per_bl * nnz_row;
    }

    for (int i = 0; i < n_rows; ++i) row_ptr[i + 1] += row_ptr[i];

    std::vector<int> col_ind(total_nnz);
    for (int b = 0; b < n_bl; ++b) {
        const int off_i = amap.h_ant_to_param(h_ant1(b));
        const int off_j = amap.h_ant_to_param(h_ant2(b));
        for (int r = 0; r < res_per_bl; ++r) {
            int pos = row_ptr[b * res_per_bl + r];
            // ant_i columns
            if (off_i >= 0) {
                for (int k = 0; k < 4; ++k) col_ind[pos++] = off_i + k;
            } else {
                col_ind[pos++] = ref_dqp_off;
                col_ind[pos++] = ref_dqp_off + 1;
            }
            // ant_j columns
            if (off_j >= 0) {
                for (int k = 0; k < 4; ++k) col_ind[pos++] = off_j + k;
            } else {
                col_ind[pos++] = ref_dqp_off;
                col_ind[pos++] = ref_dqp_off + 1;
            }
        }
    }

    host_row_map_type h_row_ptr("h_row_ptr", n_rows + 1);
    host_entries_type h_col_ind("h_col_ind", total_nnz);
    for (int i = 0; i <= n_rows; ++i) h_row_ptr(i) = row_ptr[i];
    for (int i = 0; i < total_nnz; ++i) h_col_ind(i) = col_ind[i];

    Kokkos::View<int*, mem_space> d_row_ptr("d_row_ptr", n_rows + 1);
    Kokkos::View<int*, mem_space> d_col_ind("d_col_ind", total_nnz);
    view_1d_real d_values("d_values", total_nnz);
    Kokkos::deep_copy(d_row_ptr, h_row_ptr);
    Kokkos::deep_copy(d_col_ind, h_col_ind);

    crs_matrix_type J("J", n_rows, n_cols, total_nnz,
                      d_values, d_row_ptr, d_col_ind);
    return J;
}

}  // namespace boa
