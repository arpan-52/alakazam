// boa/csr_pattern.hpp — Build the fixed CSR sparsity pattern for the Jacobian.
//
// In calibration, the Jacobian J has a very specific sparsity structure:
// each baseline (i,j) produces residual rows that depend ONLY on antenna i's
// parameters and antenna j's parameters. All other columns are zero.
//
// This means:
//   - Each row has a fixed, small number of nonzeros (nnz_per_row).
//   - The column indices are determined entirely by the antenna pairs.
//   - The pattern never changes across LM iterations — only the values change.
//
// So we build the CSR pattern (row_ptr, col_ind) once at the start,
// then reuse it every iteration, only refilling the values array.
//
// The ref_ant is excluded: its parameters have no columns in J.
// A mapping array (ant_to_param) translates antenna index → parameter offset,
// with ref_ant mapping to -1 (excluded).
//
// KokkosSparse::CrsMatrix is the Kokkos sparse matrix type.
// It stores (row_ptr, col_ind, values) in CSR format and supports
// efficient SpMV via KokkosSparse::spmv().

#ifndef BOA_CSR_PATTERN_HPP
#define BOA_CSR_PATTERN_HPP

#include <boa/types.hpp>
#include <KokkosSparse_CrsMatrix.hpp>

namespace boa {

// The CrsMatrix type we use throughout boa.
// Template parameters:
//   real_type  — the scalar type of the matrix values (double)
//   int        — the ordinal type for indices
//   exec_space — where the matrix lives (CPU or GPU)
//   mem_space  — memory space (matches exec_space)
using crs_matrix_type = KokkosSparse::CrsMatrix<
    real_type,       // scalar type for values
    int,             // ordinal type for row/col indices
    exec_space,      // execution space
    void,            // memory traits (default)
    int              // size type
>;

// Host-side row map and entries types (for building the pattern on the host
// before copying to device).
using host_row_map_type = Kokkos::View<int*, host_space>;
using host_entries_type = Kokkos::View<int*, host_space>;

// ---------------------------------------------------------------------------
// Antenna-to-parameter mapping
// ---------------------------------------------------------------------------
// For per-antenna solvers (G, K, D), each antenna except ref_ant contributes
// params_per_ant real parameters. This mapping tells us where each antenna's
// block starts in the parameter vector.
//
// ant_to_param[a] = starting column index in J for antenna a's parameters.
// ant_to_param[ref_ant] = -1 (excluded, no columns).

struct AntennaMap {
    view_1d_int ant_to_param;      // (n_ant,) device view
    host_view_1d_int h_ant_to_param; // host mirror
    int n_params;                  // total number of parameters (columns in J)
    int n_ant;
    int ref_ant;
    int params_per_ant;
};

// Build the antenna-to-parameter mapping.
// n_ant: total antennas, ref_ant: excluded antenna, params_per_ant: real params per antenna.
// Returns: AntennaMap with n_params = (n_ant - 1) * params_per_ant.
AntennaMap build_antenna_map(int n_ant, int ref_ant, int params_per_ant);

// ---------------------------------------------------------------------------
// CSR pattern builders
// ---------------------------------------------------------------------------

// Build CSR pattern for per-antenna diagonal solvers (G, K).
//
// For G (freq-independent):
//   params_per_ant = 4 (amp_p, phase_p, amp_q, phase_q)
//   residuals_per_bl = 4 (re/im for pp, re/im for qq)
//   nnz_per_row = 2 * params_per_ant = 8 (4 cols for ant_i + 4 for ant_j)
//   BUT if ant_i or ant_j is ref_ant, those columns are absent.
//
// For K (freq-dependent):
//   params_per_ant = 2 (tau_p, tau_q)
//   residuals_per_bl = 4 * n_freq
//   nnz_per_row = 2 * params_per_ant = 4
//
// ant1, ant2: host int arrays (n_bl,)
// n_freq: 1 for freq-independent, actual count for freq-dependent.
crs_matrix_type build_csr_diagonal(
    const host_view_1d_int& h_ant1,
    const host_view_1d_int& h_ant2,
    const AntennaMap& amap,
    int n_freq);

// Build CSR pattern for the full 2x2 solver (D - leakage).
//
// params_per_ant = 4 (Re(d_pq), Im(d_pq), Re(d_qp), Im(d_qp))
// residuals_per_bl = 8 (re/im for all 4 elements of 2x2)
// nnz_per_row = 2 * params_per_ant = 8
crs_matrix_type build_csr_full_2x2(
    const host_view_1d_int& h_ant1,
    const host_view_1d_int& h_ant2,
    const AntennaMap& amap);

// Build CSR pattern for leakage with Ceres ref convention:
//   d_pq[ref] fixed at 0 (excluded from params)
//   d_qp[ref] free — 2 params appended at end of vector
// Total params: (n_ant-1)*4 + 2  (= amap.n_params + 2)
// nnz_per_row: 8 for non-ref baselines, 6 for baselines involving ref.
crs_matrix_type build_csr_leakage_ceres(
    const host_view_1d_int& h_ant1,
    const host_view_1d_int& h_ant2,
    const AntennaMap& amap);

// Build CSR pattern for the gain solver with free ref-antenna amplitudes:
//   phase[ref] fixed at 0 (excluded), amp_p/amp_q[ref] free — 2 params
//   appended at end of the vector (the documented G constraint; pinning
//   the ref amp makes the system inconsistent when the data carries a
//   global flux factor relative to the model).
// Total params: (n_ant-1)*4 + 2  (= amap.n_params + 2)
// nnz_per_row: 8 for non-ref baselines, 6 for baselines involving ref.
crs_matrix_type build_csr_gain_ref_amp(
    const host_view_1d_int& h_ant1,
    const host_view_1d_int& h_ant2,
    const AntennaMap& amap);

// Build CSR pattern for global-parameter solvers (KC, CP).
//
// n_global_params: 1 for both KC and CP.
// Every residual row depends on the same global parameter(s).
// n_residuals: total residual rows.
// nnz_per_row = n_global_params (every row touches the same column(s)).
crs_matrix_type build_csr_global(
    int n_residuals,
    int n_global_params);

}  // namespace boa

#endif  // BOA_CSR_PATTERN_HPP
