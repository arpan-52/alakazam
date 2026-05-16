// boa/parameterization.hpp — Five parameterization functors.
//
// Each functor knows how to:
//   1. Convert real parameters → Jones diagonal (g_p, g_q) or full Mat2
//   2. Compute the Jacobian chain rule: ∂residual/∂params for one baseline
//
// The functors are used by the Jacobian fill kernel and by the LM loop
// to convert between the parameter vector and Jones matrices.
//
// KOKKOS_INLINE_FUNCTION on every method means it runs on host or device.
//
// Parameterizations:
//   GainParam:       amp_p, phase_p, amp_q, phase_q  (4 reals/ant)
//   DelayParam:      tau_p, tau_q in nanoseconds      (2 reals/ant, freq-dependent)
//   LeakageParam:    Re(d_pq), Im(d_pq), Re(d_qp), Im(d_qp) (4 reals/ant)
//   CrossDelayParam: single global tau                (1 real total, freq-dependent)
//   CrossPhaseParam: single global phi                (1 real total)

#ifndef BOA_PARAMETERIZATION_HPP
#define BOA_PARAMETERIZATION_HPP

#include <boa/types.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace boa {

// ---------------------------------------------------------------------------
// GainParam — diagonal gains: J = diag(g_p, g_q)
// ---------------------------------------------------------------------------
// Parameters per antenna: [amp_p, phase_p, amp_q, phase_q]
//   g_p = amp_p * exp(i * phase_p)
//   g_q = amp_q * exp(i * phase_q)
//
// Residual (diagonal RIME, pp pol):
//   r_re = Re(V_obs_pp) - Re(g_i_p * V_M_pp * conj(g_j_p))
//   r_im = Im(V_obs_pp) - Im(g_i_p * V_M_pp * conj(g_j_p))
//
// For the Jacobian we need ∂r/∂(amp_i), ∂r/∂(phase_i), ∂r/∂(amp_j), ∂r/∂(phase_j)
// separately for pp and qq polarizations.

struct GainParam {
    static constexpr int params_per_ant = 4;
    static constexpr int residuals_per_bl = 4;
    static constexpr bool freq_dependent = false;
    static constexpr bool per_antenna = true;

    // Convert parameters to diagonal Jones elements.
    KOKKOS_INLINE_FUNCTION
    static void params_to_diagonal(
        const real_type* params,  // pointer to this antenna's 4 params
        complex_type& g_p,
        complex_type& g_q)
    {
        const real_type amp_p   = params[0];
        const real_type phase_p = params[1];
        const real_type amp_q   = params[2];
        const real_type phase_q = params[3];
        g_p = complex_type(amp_p * Kokkos::cos(phase_p), amp_p * Kokkos::sin(phase_p));
        g_q = complex_type(amp_q * Kokkos::cos(phase_q), amp_q * Kokkos::sin(phase_q));
    }

    // Fill Jacobian values for one baseline, one polarization (pp or qq).
    //
    // For pp polarization:
    //   pred = g_i * M * conj(g_j)
    //   r_re = Re(obs - pred), r_im = Im(obs - pred)
    //
    // The Jacobian rows are [r_re, r_im] and columns are the params of ant_i and ant_j.
    // Since r = obs - pred, dr/dparam = -d(pred)/dparam.
    //
    // vals_i: output array of length params_per_ant (Jacobian entries for ant_i's params)
    // vals_j: output array of length params_per_ant (Jacobian entries for ant_j's params)
    // is_real: true for Re(r) row, false for Im(r) row
    // pol: 0 for pp, 1 for qq
    KOKKOS_INLINE_FUNCTION
    static void jacobian_one_row(
        real_type* vals_i,       // output: ppa entries for ant_i block
        real_type* vals_j,       // output: ppa entries for ant_j block
        const real_type* params_i,
        const real_type* params_j,
        complex_type model_val,  // V_M for this pol (pp or qq)
        int pol,                 // 0=pp, 1=qq
        bool is_real)            // true = Re(r) row, false = Im(r) row
    {
        // Which param indices within the 4-element block correspond to this pol?
        // pol=0 (pp): amp=params[0], phase=params[1]
        // pol=1 (qq): amp=params[2], phase=params[3]
        const int amp_idx   = pol * 2;
        const int phase_idx = pol * 2 + 1;

        const real_type ai = params_i[amp_idx];
        const real_type pi = params_i[phase_idx];
        const real_type aj = params_j[amp_idx];
        const real_type pj = params_j[phase_idx];

        const real_type ci = Kokkos::cos(pi), si = Kokkos::sin(pi);
        const real_type cj = Kokkos::cos(pj), sj = Kokkos::sin(pj);

        // gi = ai*(ci + i*si), conj(gj) = aj*(cj - i*sj)
        // M_cgj = model * conj(gj)
        const real_type mr = model_val.real(), mi = model_val.imag();
        const real_type mcr = mr * aj * cj + mi * aj * sj;
        const real_type mci = mi * aj * cj - mr * aj * sj;

        // pred = gi * M_cgj
        const real_type pred_re = ai * (ci * mcr - si * mci);
        const real_type pred_im = ai * (si * mcr + ci * mci);

        // Zero all entries first (non-active pol params get 0).
        for (int k = 0; k < params_per_ant; ++k) { vals_i[k] = 0.0; vals_j[k] = 0.0; }

        // d(pred)/d(amp_i) = pred / amp_i (when amp_i != 0)
        const real_type dp_amp_i_re = ci * mcr - si * mci;
        const real_type dp_amp_i_im = si * mcr + ci * mci;

        // d(pred)/d(phase_i) = i * pred (rotation)
        const real_type dp_phi_i_re = -pred_im;
        const real_type dp_phi_i_im =  pred_re;

        // d(pred)/d(amp_j): need d(M*conj(gj))/d(amp_j)
        // conj(gj)/aj = cj - i*sj, so d(M*conj(gj))/d(aj) = M*(cj - i*sj)
        const real_type t_re = mr * cj + mi * sj;
        const real_type t_im = mi * cj - mr * sj;
        const real_type dp_amp_j_re = ai * (ci * t_re - si * t_im);
        const real_type dp_amp_j_im = ai * (si * t_re + ci * t_im);

        // d(pred)/d(phase_j): d(conj(gj))/d(pj) = -i*conj(gj)
        // So d(pred)/d(pj) = gi * M * (-i*conj(gj)) = -i * pred
        const real_type dp_phi_j_re =  pred_im;
        const real_type dp_phi_j_im = -pred_re;

        // r = obs - pred, so dr/dparam = -d(pred)/dparam.
        // Select Re or Im component.
        if (is_real) {
            vals_i[amp_idx]   = -dp_amp_i_re;
            vals_i[phase_idx] = -dp_phi_i_re;
            vals_j[amp_idx]   = -dp_amp_j_re;
            vals_j[phase_idx] = -dp_phi_j_re;
        } else {
            vals_i[amp_idx]   = -dp_amp_i_im;
            vals_i[phase_idx] = -dp_phi_i_im;
            vals_j[amp_idx]   = -dp_amp_j_im;
            vals_j[phase_idx] = -dp_phi_j_im;
        }
    }
};

// ---------------------------------------------------------------------------
// DelayParam — parallel delay: J = diag(exp(-2πi τ_p ν), exp(-2πi τ_q ν))
// ---------------------------------------------------------------------------
// Parameters per antenna: [tau_p, tau_q] in nanoseconds.
// Freq-dependent: each freq channel produces residuals.

struct DelayParam {
    static constexpr int params_per_ant = 2;
    static constexpr int residuals_per_bl_per_freq = 4;
    static constexpr bool freq_dependent = true;
    static constexpr bool per_antenna = true;

    // Convert delay to diagonal Jones at one frequency.
    KOKKOS_INLINE_FUNCTION
    static void params_to_diagonal(
        const real_type* params,  // [tau_p, tau_q] ns
        real_type freq_hz,
        complex_type& g_p,
        complex_type& g_q)
    {
        // phase = -2π * τ * ν, with τ in ns → τ*1e-9 in seconds.
        const real_type phase_p = -2.0 * Kokkos::numbers::pi_v<real_type> * params[0] * 1.0e-9 * freq_hz;
        const real_type phase_q = -2.0 * Kokkos::numbers::pi_v<real_type> * params[1] * 1.0e-9 * freq_hz;
        g_p = complex_type(Kokkos::cos(phase_p), Kokkos::sin(phase_p));
        g_q = complex_type(Kokkos::cos(phase_q), Kokkos::sin(phase_q));
    }

    // Jacobian for one row: d(residual)/d(tau_i) and d(residual)/d(tau_j).
    // pred_pol = g_i * model * conj(g_j)
    // g_i = exp(-2πi τ_i ν * 1e-9), dg_i/dτ_i = -2πi ν * 1e-9 * g_i
    KOKKOS_INLINE_FUNCTION
    static void jacobian_one_row(
        real_type* vals_i,
        real_type* vals_j,
        const real_type* params_i,
        const real_type* params_j,
        complex_type model_val,
        real_type freq_hz,
        int pol,        // 0=pp, 1=qq
        bool is_real)
    {
        vals_i[0] = 0.0; vals_i[1] = 0.0;
        vals_j[0] = 0.0; vals_j[1] = 0.0;

        const real_type w = -2.0 * Kokkos::numbers::pi_v<real_type> * 1.0e-9 * freq_hz;
        const real_type tau_i = params_i[pol];
        const real_type tau_j = params_j[pol];
        const real_type phase_i = w * tau_i;
        const real_type phase_j = w * tau_j;

        // g_i = exp(i*phase_i), conj(g_j) = exp(-i*phase_j)
        const complex_type gi(Kokkos::cos(phase_i), Kokkos::sin(phase_i));
        const complex_type cgj(Kokkos::cos(phase_j), -Kokkos::sin(phase_j));

        // pred = gi * model * cgj
        const complex_type pred = gi * model_val * cgj;

        // d(pred)/d(tau_i) = (i*w) * gi * model * cgj = (i*w) * pred
        const complex_type dp_i = complex_type(0.0, w) * pred;

        // d(pred)/d(tau_j): d(cgj)/d(tau_j) = -i*w * cgj (since cgj = exp(-i*w*tau_j))
        // Wait: cgj = exp(-i * phase_j) = exp(-i * w * tau_j)
        // d(cgj)/d(tau_j) = -i*w * cgj
        // So d(pred)/d(tau_j) = gi * model * (-i*w * cgj) = -i*w * pred
        const complex_type dp_j = complex_type(0.0, -w) * pred;

        // r = obs - pred → dr/dparam = -d(pred)/dparam
        if (is_real) {
            vals_i[pol] = -dp_i.real();
            vals_j[pol] = -dp_j.real();
        } else {
            vals_i[pol] = -dp_i.imag();
            vals_j[pol] = -dp_j.imag();
        }
    }
};

// ---------------------------------------------------------------------------
// LeakageParam — D-Jones: J = [[1, d_pq], [d_qp, 1]]
// ---------------------------------------------------------------------------
// Parameters per antenna: [Re(d_pq), Im(d_pq), Re(d_qp), Im(d_qp)]
// Full 2x2 RIME: R = V_obs - J_i * V_M * J_j†

struct LeakageParam {
    static constexpr int params_per_ant = 4;
    static constexpr int residuals_per_bl = 8;
    static constexpr bool freq_dependent = false;
    static constexpr bool per_antenna = true;

    // Convert params to Jones matrix.
    KOKKOS_INLINE_FUNCTION
    static Mat2 params_to_jones(const real_type* params) {
        Mat2 J;
        J(0, 0) = complex_type(1.0, 0.0);
        J(0, 1) = complex_type(params[0], params[1]);  // d_pq
        J(1, 0) = complex_type(params[2], params[3]);  // d_qp
        J(1, 1) = complex_type(1.0, 0.0);
        return J;
    }

    // Jacobian for one row of the full 2x2 residual.
    //
    // pred = J_i * V_M * J_j†
    // R = V_obs - pred
    //
    // The 8 residual rows per baseline map to (matrix_element, re/im):
    //   row 0: Re(R(0,0)), row 1: Im(R(0,0))
    //   row 2: Re(R(0,1)), row 3: Im(R(0,1))
    //   row 4: Re(R(1,0)), row 5: Im(R(1,0))
    //   row 6: Re(R(1,1)), row 7: Im(R(1,1))
    //
    // res_idx: 0..7 within this baseline's residual block.
    KOKKOS_INLINE_FUNCTION
    static void jacobian_one_row(
        real_type* vals_i,
        real_type* vals_j,
        const real_type* params_i,
        const real_type* params_j,
        const Mat2& VM,
        int res_idx)
    {
        // Which element of the 2x2 and re/im?
        const int mat_idx = res_idx / 2;  // 0..3 → (0,0),(0,1),(1,0),(1,1)
        const bool is_real = (res_idx % 2 == 0);
        const int rm = mat_idx / 2;  // matrix row: 0 or 1
        const int cm = mat_idx % 2;  // matrix col: 0 or 1

        // Build Ji, Jj from params.
        const complex_type dpi(params_i[0], params_i[1]);
        const complex_type dqi(params_i[2], params_i[3]);
        const complex_type dpj(params_j[0], params_j[1]);
        const complex_type dqj(params_j[2], params_j[3]);

        // A = V_M * J_j† (intermediate for Ji derivatives)
        // J_j† = [[1, conj(dqj)], [conj(dpj), 1]]
        const complex_type cdpj = Kokkos::conj(dpj);
        const complex_type cdqj = Kokkos::conj(dqj);

        // MJjH[r][c] = sum_k VM[r][k] * JjH[k][c]
        // JjH = [[1, cdqj], [cdpj, 1]]
        complex_type MJjH[2][2];
        MJjH[0][0] = VM(0, 0) + VM(0, 1) * cdpj;
        MJjH[0][1] = VM(0, 0) * cdqj + VM(0, 1);
        MJjH[1][0] = VM(1, 0) + VM(1, 1) * cdpj;
        MJjH[1][1] = VM(1, 0) * cdqj + VM(1, 1);

        // JiM[r][c] = sum_k Ji[r][k] * VM[k][c]
        // Ji = [[1, dpi], [dqi, 1]]
        complex_type JiM[2][2];
        JiM[0][0] = VM(0, 0) + dpi * VM(1, 0);
        JiM[0][1] = VM(0, 1) + dpi * VM(1, 1);
        JiM[1][0] = dqi * VM(0, 0) + VM(1, 0);
        JiM[1][1] = dqi * VM(0, 1) + VM(1, 1);

        // pred[rm][cm] = Ji[rm,:] * MJjH[:,cm]
        // = (rm==0 ? [1, dpi] : [dqi, 1]) . MJjH[:,cm]

        // --- Derivatives wrt Ji parameters ---
        // dpi appears in Ji row 0: Ji[0,:] = [1, dpi]
        // dqi appears in Ji row 1: Ji[1,:] = [dqi, 1]
        //
        // d(pred[rm][cm])/d(Re dpi):
        //   rm==0: MJjH[1][cm] (since Ji[0,:] = [1, dpi], d(dpi)/d(Re dpi) = 1)
        //   rm==1: 0
        // d(pred[rm][cm])/d(Im dpi):
        //   rm==0: i * MJjH[1][cm]
        //   rm==1: 0
        //
        // d(pred[rm][cm])/d(Re dqi):
        //   rm==0: 0
        //   rm==1: MJjH[0][cm]
        // d(pred[rm][cm])/d(Im dqi):
        //   rm==0: 0
        //   rm==1: i * MJjH[0][cm]

        complex_type dp_dpi_re, dp_dpi_im, dp_dqi_re, dp_dqi_im;
        if (rm == 0) {
            dp_dpi_re = MJjH[1][cm];
            dp_dpi_im = complex_type(0.0, 1.0) * MJjH[1][cm];
            dp_dqi_re = complex_type(0.0, 0.0);
            dp_dqi_im = complex_type(0.0, 0.0);
        } else {
            dp_dpi_re = complex_type(0.0, 0.0);
            dp_dpi_im = complex_type(0.0, 0.0);
            dp_dqi_re = MJjH[0][cm];
            dp_dqi_im = complex_type(0.0, 1.0) * MJjH[0][cm];
        }

        // --- Derivatives wrt Jj parameters ---
        // JjH = [[1, cdqj], [cdpj, 1]]
        // d(cdpj)/d(Re dpj) = 1, d(cdpj)/d(Im dpj) = -i
        // d(cdqj)/d(Re dqj) = 1, d(cdqj)/d(Im dqj) = -i
        //
        // d(pred[rm][cm])/d(Re dpj):
        //   pred = Ji * M * JjH, d(JjH)/d(Re dpj) = [[0,0],[1,0]]
        //   d(pred)/d(Re dpj) = Ji * M * [[0,0],[1,0]] col cm
        //   col 0: JiM[:,1] * 0 + ... → only JiM[:,1]*delta(row=1,col=0)
        //   Actually: d(MJjH)/d(Re dpj) col cm → only affects MJjH[r][0] via cdpj term.
        //   M * d(JjH)/d(Re dpj) = M * [[0,0],[1,0]] → column: [M[0,1], M[1,1]] for col 0, [0,0] for col 1
        //   Ji times that: Ji * [M[:,1], 0] → pred derivative
        //   But simpler: JiM[rm][1] for cm==0, 0 for cm==1 (dpj is in JjH[1][0] position)
        // Wait, let me redo more carefully.
        // d(JjH)/d(Re dpj): JjH[1][0] = cdpj, d(cdpj)/d(Re dpj) = 1
        // So dJjH = [[0,0],[1,0]]
        // d(pred) = Ji * VM * dJjH
        // (Ji*VM*dJjH)[rm][cm] = sum_k JiM[rm][k] * dJjH[k][cm]
        //   = JiM[rm][1] * dJjH[1][cm] (k=1 only nonzero)
        //   = JiM[rm][1] * delta(cm, 0)  → nonzero only for cm==0

        complex_type dp_dpj_re, dp_dpj_im, dp_dqj_re, dp_dqj_im;

        // dpj: d(JjH)/d(Re dpj) = [[0,0],[1,0]]
        dp_dpj_re = (cm == 0) ? JiM[rm][1] : complex_type(0.0, 0.0);
        // d(JjH)/d(Im dpj): JjH[1][0] = cdpj, d(cdpj)/d(Im dpj) = -i
        dp_dpj_im = (cm == 0) ? complex_type(0.0, -1.0) * JiM[rm][1] : complex_type(0.0, 0.0);

        // dqj: d(JjH)/d(Re dqj) = [[0,1],[0,0]]  (JjH[0][1] = cdqj)
        dp_dqj_re = (cm == 1) ? JiM[rm][0] : complex_type(0.0, 0.0);
        dp_dqj_im = (cm == 1) ? complex_type(0.0, -1.0) * JiM[rm][0] : complex_type(0.0, 0.0);

        // r = obs - pred → dr/dparam = -d(pred)/dparam
        // Extract Re or Im component.
        if (is_real) {
            vals_i[0] = -dp_dpi_re.real();
            vals_i[1] = -dp_dpi_im.real();
            vals_i[2] = -dp_dqi_re.real();
            vals_i[3] = -dp_dqi_im.real();
            vals_j[0] = -dp_dpj_re.real();
            vals_j[1] = -dp_dpj_im.real();
            vals_j[2] = -dp_dqj_re.real();
            vals_j[3] = -dp_dqj_im.real();
        } else {
            vals_i[0] = -dp_dpi_re.imag();
            vals_i[1] = -dp_dpi_im.imag();
            vals_i[2] = -dp_dqi_re.imag();
            vals_i[3] = -dp_dqi_im.imag();
            vals_j[0] = -dp_dpj_re.imag();
            vals_j[1] = -dp_dpj_im.imag();
            vals_j[2] = -dp_dqj_re.imag();
            vals_j[3] = -dp_dqj_im.imag();
        }
    }
};

// ---------------------------------------------------------------------------
// CrossDelayParam — single global delay: J = diag(exp(-2πi τ ν), 1)
// ---------------------------------------------------------------------------
// 1 global parameter: tau (nanoseconds).
// Freq-dependent. Affects only the pp cross-hand term.
// residual uses cross-hand: V_pq and V_qp.

struct CrossDelayParam {
    static constexpr int n_global_params = 1;
    static constexpr bool freq_dependent = true;
    static constexpr bool per_antenna = false;

    // g_p = exp(-2πi τ ν), g_q = 1
    KOKKOS_INLINE_FUNCTION
    static void param_to_diagonal(
        real_type tau_ns,
        real_type freq_hz,
        complex_type& g_p,
        complex_type& g_q)
    {
        const real_type phase = -2.0 * Kokkos::numbers::pi_v<real_type> * tau_ns * 1.0e-9 * freq_hz;
        g_p = complex_type(Kokkos::cos(phase), Kokkos::sin(phase));
        g_q = complex_type(1.0, 0.0);
    }

    // Jacobian: d(residual)/d(tau).
    // Cross-hand pq: pred_pq = g_p * V_M_pq * conj(g_q) = g_p * V_M_pq (since g_q=1)
    //   d(pred_pq)/d(tau) = (-2πi ν * 1e-9) * g_p * V_M_pq
    // Cross-hand qp: pred_qp = g_q * V_M_qp * conj(g_p) = V_M_qp * conj(g_p)
    //   d(pred_qp)/d(tau) = V_M_qp * d(conj(g_p))/d(tau) = V_M_qp * (2πi ν * 1e-9) * conj(g_p)
    KOKKOS_INLINE_FUNCTION
    static void jacobian_cross(
        real_type& val,
        real_type tau_ns,
        real_type freq_hz,
        complex_type model_val,
        bool is_pq,     // true for pq, false for qp
        bool is_real)
    {
        const real_type w = -2.0 * Kokkos::numbers::pi_v<real_type> * 1.0e-9 * freq_hz;
        const real_type phase = w * tau_ns;
        const complex_type gp(Kokkos::cos(phase), Kokkos::sin(phase));

        complex_type dpred;
        if (is_pq) {
            // pred = gp * model, d(pred)/d(tau) = i*w * gp * model
            dpred = complex_type(0.0, w) * gp * model_val;
        } else {
            // pred = model * conj(gp), d(pred)/d(tau) = model * d(conj(gp))/d(tau)
            // d(conj(gp))/d(tau) = conj(i*w*gp) = -i*w * conj(gp)
            dpred = model_val * complex_type(0.0, -w) * Kokkos::conj(gp);
        }

        // r = obs - pred → dr/dtau = -dpred/dtau
        val = is_real ? -dpred.real() : -dpred.imag();
    }
};

// ---------------------------------------------------------------------------
// CrossPhaseParam — single global phase: J = diag(1, exp(i φ))
// ---------------------------------------------------------------------------
// 1 global parameter: phi (radians).
// Freq-independent. Affects cross-hand correlations.

struct CrossPhaseParam {
    static constexpr int n_global_params = 1;
    static constexpr bool freq_dependent = false;
    static constexpr bool per_antenna = false;

    // g_p = 1, g_q = exp(i*phi)
    KOKKOS_INLINE_FUNCTION
    static void param_to_diagonal(
        real_type phi,
        complex_type& g_p,
        complex_type& g_q)
    {
        g_p = complex_type(1.0, 0.0);
        g_q = complex_type(Kokkos::cos(phi), Kokkos::sin(phi));
    }

    // Cross-hand pq: pred_pq = g_p * V_M_pq * conj(g_q) = V_M_pq * conj(exp(i*phi))
    //   d(pred_pq)/d(phi) = V_M_pq * (-i) * conj(exp(i*phi))
    // Cross-hand qp: pred_qp = g_q * V_M_qp * conj(g_p) = exp(i*phi) * V_M_qp
    //   d(pred_qp)/d(phi) = i * exp(i*phi) * V_M_qp
    KOKKOS_INLINE_FUNCTION
    static void jacobian_cross(
        real_type& val,
        real_type phi,
        complex_type model_val,
        bool is_pq,
        bool is_real)
    {
        const complex_type gq(Kokkos::cos(phi), Kokkos::sin(phi));

        complex_type dpred;
        if (is_pq) {
            dpred = model_val * complex_type(0.0, -1.0) * Kokkos::conj(gq);
        } else {
            dpred = complex_type(0.0, 1.0) * gq * model_val;
        }

        val = is_real ? -dpred.real() : -dpred.imag();
    }
};

}  // namespace boa

#endif  // BOA_PARAMETERIZATION_HPP
