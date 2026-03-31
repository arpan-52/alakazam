/*
 * ALAKAZAM C++ Ceres Solvers — pybind11 module.
 *
 * Pure C++ cost functions + solver wrappers for all 5 Jones types:
 *   solve_gains      — G solver (diagonal, amp+phase)
 *   solve_leakage    — D solver (full 2x2)
 *   solve_delay      — K solver (parallel delay)
 *   solve_cross_delay — KC solver (global cross-hand delay)
 *   solve_cross_phase — CP solver (global cross-hand phase)
 *
 * No Python callbacks in the hot loop — Ceres runs entirely in C++.
 *
 * Developed by Arpan Pal 2026, NRAO / NCRA
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <ceres/ceres.h>
#include <cmath>
#include <complex>
#include <vector>

namespace py = pybind11;
using cd = std::complex<double>;

// ======================================================================
// Shared Ceres options
// ======================================================================

static ceres::Solver::Options make_opts(int max_iter, double tol) {
    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    opts.max_num_iterations = max_iter;
    opts.function_tolerance = tol;
    opts.gradient_tolerance = tol;
    opts.parameter_tolerance = tol;
    opts.minimizer_progress_to_stdout = false;
    opts.num_threads = 1;  // parallelism at cell level
    return opts;
}


// ======================================================================
// G solver — diagonal gain, per baseline per pol
// ======================================================================

struct GainCost : public ceres::SizedCostFunction<2, 1, 1, 1, 1> {
    double obs_re, obs_im, mod_re, mod_im;

    GainCost(cd obs, cd mod)
        : obs_re(obs.real()), obs_im(obs.imag()),
          mod_re(mod.real()), mod_im(mod.imag()) {}

    bool Evaluate(double const* const* p,
                  double* res, double** jac) const override {
        double ai = p[0][0], pi_ = p[1][0];
        double aj = p[2][0], pj = p[3][0];

        double ci = cos(pi_), si = sin(pi_);
        double cj = cos(pj), sj = sin(pj);

        // gi = ai*(ci + i*si), gj = aj*(cj + i*sj)
        // pred = gi * model * conj(gj)
        // model * conj(gj) = (mr + i*mi)*(aj*cj - i*aj*sj)
        double mcr = mod_re * aj * cj + mod_im * aj * sj;
        double mci = mod_im * aj * cj - mod_re * aj * sj;

        double pred_re = ai * (ci * mcr - si * mci);
        double pred_im = ai * (si * mcr + ci * mci);

        res[0] = obs_re - pred_re;
        res[1] = obs_im - pred_im;

        if (jac) {
            // d/d(ai): pred/ai
            if (jac[0]) {
                jac[0][0] = -(ci * mcr - si * mci);
                jac[0][1] = -(si * mcr + ci * mci);
            }
            // d/d(pi): i * gi * M*conj(gj) = i * pred
            if (jac[1]) {
                jac[1][0] = pred_im;   // -(-pred_im)
                jac[1][1] = -pred_re;  // -(pred_re)
            }
            // d/d(aj): gi * model * d(conj(gj))/d(aj)
            // d(conj(gj))/d(aj) = cj - i*sj
            if (jac[2]) {
                double t_re = mod_re * cj + mod_im * sj;
                double t_im = mod_im * cj - mod_re * sj;
                double r_re = ai * (ci * t_re - si * t_im);
                double r_im = ai * (si * t_re + ci * t_im);
                jac[2][0] = -r_re;
                jac[2][1] = -r_im;
            }
            // d/d(pj): gi * model * (-i) * conj(gj)
            if (jac[3]) {
                jac[3][0] = pred_im;
                jac[3][1] = -pred_re;
                // Actually: d(conj(gj))/d(pj) = aj*(-sj - i*cj) = -i*conj(gj)
                // dpred/dpj = gi * M * (-i*conj(gj)) = -i * pred
                // So: d(pred_re)/dpj = pred_im, d(pred_im)/dpj = -pred_re
                // res = obs - pred, so jac = -dpred/dpj
                jac[3][0] = -pred_im;
                jac[3][1] = pred_re;
            }
        }
        return true;
    }
};


py::tuple solve_gains(
    py::array_t<cd> obs_arr,     // (n_bl, 2, 2)
    py::array_t<cd> model_arr,   // (n_bl, 2, 2)
    py::array_t<int> ant1_arr,   // (n_bl,)
    py::array_t<int> ant2_arr,   // (n_bl,)
    int n_ant, int ref_ant, int max_iter, double tol,
    py::array_t<double> amp_init_arr,   // (n_ant, 2)
    py::array_t<double> phase_init_arr, // (n_ant, 2)
    bool phase_only)
{
    auto obs = obs_arr.unchecked<3>();
    auto mod = model_arr.unchecked<3>();
    auto a1 = ant1_arr.unchecked<1>();
    auto a2 = ant2_arr.unchecked<1>();
    auto amp_i = amp_init_arr.unchecked<2>();
    auto ph_i = phase_init_arr.unchecked<2>();
    int n_bl = obs.shape(0);

    // Parameter arrays
    std::vector<double> amp_p(n_ant), phi_p(n_ant);
    std::vector<double> amp_q(n_ant), phi_q(n_ant);
    for (int a = 0; a < n_ant; a++) {
        amp_p[a] = amp_i(a, 0); phi_p[a] = ph_i(a, 0);
        amp_q[a] = amp_i(a, 1); phi_q[a] = ph_i(a, 1);
    }

    ceres::Problem problem;
    for (int k = 0; k < n_bl; k++) {
        int i = a1(k), j = a2(k);
        cd o_pp(obs(k, 0, 0)), m_pp(mod(k, 0, 0));
        cd o_qq(obs(k, 1, 1)), m_qq(mod(k, 1, 1));

        problem.AddResidualBlock(new GainCost(o_pp, m_pp), nullptr,
            &amp_p[i], &phi_p[i], &amp_p[j], &phi_p[j]);
        problem.AddResidualBlock(new GainCost(o_qq, m_qq), nullptr,
            &amp_q[i], &phi_q[i], &amp_q[j], &phi_q[j]);
    }

    problem.SetParameterBlockConstant(&phi_p[ref_ant]);
    problem.SetParameterBlockConstant(&phi_q[ref_ant]);

    auto opts = make_opts(max_iter, tol);
    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    // Pack result: (n_ant, 2, 2) Jones
    auto jones = py::array_t<cd>({n_ant, 2, 2});
    auto j = jones.mutable_unchecked<3>();
    for (int a = 0; a < n_ant; a++) {
        double ap = phase_only ? 1.0 : amp_p[a];
        double aq = phase_only ? 1.0 : amp_q[a];
        phi_p[ref_ant] = 0.0; phi_q[ref_ant] = 0.0;
        j(a, 0, 0) = cd(ap * cos(phi_p[a]), ap * sin(phi_p[a]));
        j(a, 0, 1) = cd(0.0, 0.0);
        j(a, 1, 0) = cd(0.0, 0.0);
        j(a, 1, 1) = cd(aq * cos(phi_q[a]), aq * sin(phi_q[a]));
    }

    return py::make_tuple(
        jones, summary.final_cost,
        summary.num_successful_steps,
        summary.termination_type == ceres::CONVERGENCE);
}


// ======================================================================
// D solver — full 2x2 leakage
// ======================================================================

struct LeakageCost : public ceres::SizedCostFunction<8, 2, 2, 2, 2> {
    double obs_re[4], obs_im[4];
    double mod_re[4], mod_im[4];

    LeakageCost(const cd* obs, const cd* mod) {
        for (int i = 0; i < 4; i++) {
            obs_re[i] = obs[i].real(); obs_im[i] = obs[i].imag();
            mod_re[i] = mod[i].real(); mod_im[i] = mod[i].imag();
        }
    }

    bool Evaluate(double const* const* p,
                  double* res, double** jac) const override {
        // Ji = [[1, dp_i], [dq_i, 1]]
        // Jj = [[1, dp_j], [dq_j, 1]]
        cd dp_i(p[0][0], p[0][1]);
        cd dq_i(p[1][0], p[1][1]);
        cd dp_j(p[2][0], p[2][1]);
        cd dq_j(p[3][0], p[3][1]);

        // M as 2x2
        cd M[4] = {cd(mod_re[0], mod_im[0]), cd(mod_re[1], mod_im[1]),
                    cd(mod_re[2], mod_im[2]), cd(mod_re[3], mod_im[3])};

        // JjH = conj(Jj)^T = [[1, conj(dq_j)], [conj(dp_j), 1]]
        cd cjp = std::conj(dp_j), cjq = std::conj(dq_j);

        // MJjH = M @ JjH
        cd MJH[4];
        MJH[0] = M[0] + M[1] * cjp;
        MJH[1] = M[0] * cjq + M[1];
        MJH[2] = M[2] + M[3] * cjp;
        MJH[3] = M[2] * cjq + M[3];

        // pred = Ji @ MJjH
        cd pred[4];
        pred[0] = MJH[0] + dp_i * MJH[2];
        pred[1] = MJH[1] + dp_i * MJH[3];
        pred[2] = dq_i * MJH[0] + MJH[2];
        pred[3] = dq_i * MJH[1] + MJH[3];

        for (int i = 0; i < 4; i++) {
            res[2*i]   = obs_re[i] - pred[i].real();
            res[2*i+1] = obs_im[i] - pred[i].imag();
        }

        if (jac) {
            // JiM = Ji @ M
            cd JiM[4];
            JiM[0] = M[0] + dp_i * M[2];
            JiM[1] = M[1] + dp_i * M[3];
            JiM[2] = dq_i * M[0] + M[2];
            JiM[3] = dq_i * M[1] + M[3];

            // d/d dp_i (Re, Im): dpred/d(Re dp_i) = [[MJH[2,0], MJH[2,1]],[0,0]]
            if (jac[0]) {
                // d(dp_i)/d(Re dp_i) = 1, d(dp_i)/d(Im dp_i) = i
                cd dp_dre[4] = {MJH[2], MJH[3], cd(0), cd(0)};
                cd dp_dim[4] = {cd(0,1)*MJH[2], cd(0,1)*MJH[3], cd(0), cd(0)};
                for (int i = 0; i < 4; i++) {
                    jac[0][2*i*2+0] = -dp_dre[i].real();
                    jac[0][2*i*2+1] = -dp_dim[i].real();
                    jac[0][(2*i+1)*2+0] = -dp_dre[i].imag();
                    jac[0][(2*i+1)*2+1] = -dp_dim[i].imag();
                }
            }
            // d/d dq_i
            if (jac[1]) {
                cd dq_dre[4] = {cd(0), cd(0), MJH[0], MJH[1]};
                cd dq_dim[4] = {cd(0), cd(0), cd(0,1)*MJH[0], cd(0,1)*MJH[1]};
                for (int i = 0; i < 4; i++) {
                    jac[1][2*i*2+0] = -dq_dre[i].real();
                    jac[1][2*i*2+1] = -dq_dim[i].real();
                    jac[1][(2*i+1)*2+0] = -dq_dre[i].imag();
                    jac[1][(2*i+1)*2+1] = -dq_dim[i].imag();
                }
            }
            // d/d dp_j: dJjH/d(Re dp_j) has conj derivative
            // JjH = [[1, conj(dq_j)],[conj(dp_j), 1]]
            // d(conj(dp_j))/d(Re dp_j) = 1, d(conj(dp_j))/d(Im dp_j) = -i
            if (jac[2]) {
                // dpred/d(Re dp_j) = Ji @ M @ d(JjH)/d(Re dp_j)
                // dJjH/d(Re dp_j) = [[0,0],[1,0]]
                // M @ dJjH = [[M01, 0],[M11, 0]] ... no
                // pred = Ji @ M @ JjH, dpred/d = Ji @ M @ (dJjH/d)
                // dJjH/d(Re dp_j) = [[0,0],[1,0]]
                // M @ [[0,0],[1,0]] = [[M[0,1], 0],[M[1,1], 0]]
                // Ji @ that = [[ M01 + dp_i*M11, 0], [dq_i*M01 + M11, 0]]
                cd col0_re[4] = {JiM[1], cd(0), JiM[3], cd(0)};
                // dJjH/d(Im dp_j) = [[0,0],[-i,0]]
                cd col0_im[4] = {cd(0,-1)*JiM[1], cd(0), cd(0,-1)*JiM[3], cd(0)};
                for (int i = 0; i < 4; i++) {
                    jac[2][2*i*2+0] = -col0_re[i].real();
                    jac[2][2*i*2+1] = -col0_im[i].real();
                    jac[2][(2*i+1)*2+0] = -col0_re[i].imag();
                    jac[2][(2*i+1)*2+1] = -col0_im[i].imag();
                }
            }
            // d/d dq_j
            if (jac[3]) {
                // dJjH/d(Re dq_j) = [[0,1],[0,0]]
                // M @ [[0,1],[0,0]] = [[0, M00],[0, M10]]
                // Ji @ that = [[0, JiM00], [0, JiM10]]
                cd col1_re[4] = {cd(0), JiM[0], cd(0), JiM[2]};
                cd col1_im[4] = {cd(0), cd(0,-1)*JiM[0], cd(0), cd(0,-1)*JiM[2]};
                for (int i = 0; i < 4; i++) {
                    jac[3][2*i*2+0] = -col1_re[i].real();
                    jac[3][2*i*2+1] = -col1_im[i].real();
                    jac[3][(2*i+1)*2+0] = -col1_re[i].imag();
                    jac[3][(2*i+1)*2+1] = -col1_im[i].imag();
                }
            }
        }
        return true;
    }
};


py::tuple solve_leakage(
    py::array_t<cd> obs_arr,     // (n_bl, 2, 2)
    py::array_t<cd> model_arr,
    py::array_t<int> ant1_arr,
    py::array_t<int> ant2_arr,
    int n_ant, int ref_ant, int max_iter, double tol,
    py::array_t<cd> dpq_init_arr,   // (n_ant,)
    py::array_t<cd> dqp_init_arr)   // (n_ant,)
{
    auto obs = obs_arr.unchecked<3>();
    auto mod = model_arr.unchecked<3>();
    auto a1 = ant1_arr.unchecked<1>();
    auto a2 = ant2_arr.unchecked<1>();
    auto dpq_i = dpq_init_arr.unchecked<1>();
    auto dqp_i = dqp_init_arr.unchecked<1>();
    int n_bl = obs.shape(0);

    // Parameters: [Re(dpq), Im(dpq)] and [Re(dqp), Im(dqp)] per antenna
    std::vector<std::array<double, 2>> params_dpq(n_ant), params_dqp(n_ant);
    for (int a = 0; a < n_ant; a++) {
        params_dpq[a] = {dpq_i(a).real(), dpq_i(a).imag()};
        params_dqp[a] = {dqp_i(a).real(), dqp_i(a).imag()};
    }

    ceres::Problem problem;
    for (int k = 0; k < n_bl; k++) {
        int i = a1(k), j = a2(k);
        cd o[4] = {obs(k,0,0), obs(k,0,1), obs(k,1,0), obs(k,1,1)};
        cd m[4] = {mod(k,0,0), mod(k,0,1), mod(k,1,0), mod(k,1,1)};
        problem.AddResidualBlock(new LeakageCost(o, m), nullptr,
            params_dpq[i].data(), params_dqp[i].data(),
            params_dpq[j].data(), params_dqp[j].data());
    }

    problem.SetParameterBlockConstant(params_dpq[ref_ant].data());

    auto opts = make_opts(max_iter, tol);
    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    // Build Jones: [[1, dpq], [dqp, 1]]
    auto jones = py::array_t<cd>({n_ant, 2, 2});
    auto j = jones.mutable_unchecked<3>();
    for (int a = 0; a < n_ant; a++) {
        cd dpq(params_dpq[a][0], params_dpq[a][1]);
        cd dqp(params_dqp[a][0], params_dqp[a][1]);
        if (a == ref_ant) dpq = cd(0, 0);
        j(a, 0, 0) = cd(1, 0);
        j(a, 0, 1) = dpq;
        j(a, 1, 0) = dqp;
        j(a, 1, 1) = cd(1, 0);
    }

    return py::make_tuple(
        jones, summary.final_cost,
        summary.num_successful_steps,
        summary.termination_type == ceres::CONVERGENCE);
}


// ======================================================================
// K solver — parallel delay (frequency-dependent)
// ======================================================================

struct DelayCost : public ceres::CostFunction {
    std::vector<double> obs_re, obs_im;
    std::vector<double> mod_re, mod_im;
    std::vector<double> twopi_nu;
    int n_freq;

    DelayCost(const cd* obs_data, const cd* mod_data,
              const double* freqs, int nf)
        : n_freq(nf) {
        obs_re.resize(nf); obs_im.resize(nf);
        mod_re.resize(nf); mod_im.resize(nf);
        twopi_nu.resize(nf);
        for (int i = 0; i < nf; i++) {
            obs_re[i] = obs_data[i].real();
            obs_im[i] = obs_data[i].imag();
            mod_re[i] = mod_data[i].real();
            mod_im[i] = mod_data[i].imag();
            twopi_nu[i] = 2.0 * M_PI * freqs[i] * 1e-9;
        }
        set_num_residuals(nf * 2);
        mutable_parameter_block_sizes()->push_back(1);
        mutable_parameter_block_sizes()->push_back(1);
    }

    bool Evaluate(double const* const* p,
                  double* res, double** jac) const override {
        double tau_i = p[0][0], tau_j = p[1][0];
        double dtau = tau_i - tau_j;

        // pred = model * gi * conj(gj) = model * exp(-2πi (τi-τj) ν)
        for (int f = 0; f < n_freq; f++) {
            double phase = -twopi_nu[f] * dtau;
            double cp = cos(phase), sp = sin(phase);
            // exp(i*phase) * model = (cp + i*sp) * (mr + i*mi)
            double pred_re = cp * mod_re[f] - sp * mod_im[f];
            double pred_im = sp * mod_re[f] + cp * mod_im[f];
            res[2*f]   = obs_re[f] - pred_re;
            res[2*f+1] = obs_im[f] - pred_im;

            if (jac) {
                // d(pred)/d(dtau) = model * d(exp(i*phase))/d(dtau)
                //                 = model * (-i * twopi_nu) * exp(i*phase)
                // d(pred)/d(tau_i) = model * (-i * twopi_nu) * exp(i*phase)
                // dp_re = twopi_nu * (sp*mr + cp*mi) = twopi_nu * pred_im
                // dp_im = twopi_nu * (-cp*mr + sp*mi) = -twopi_nu * pred_re
                // Wait, let me be precise:
                // exp' = (-i*w)*exp = (-i*w)*(cp+i*sp) = w*sp - i*w*cp
                // (model * exp')_re = mr*w*sp - mi*(-w*cp) = w*(mr*sp + mi*cp)
                // (model * exp')_im = mi*w*sp + mr*(-w*cp) = w*(mi*sp - mr*cp)
                double w = -twopi_nu[f];  // d(phase)/d(tau_i)
                double exp_d_re = w * sp;   // d(exp_re)/d(dtau) = w * sp
                double exp_d_im = -w * cp;  // d(exp_im)/d(dtau) = -w * cp (actually w*cp)

                // Correct: d(exp(i*phase))/d(dtau) = i*w*exp(i*phase)
                //        = i*w*(cp + i*sp) = -w*sp + i*w*cp
                double de_re = -w * sp;
                double de_im =  w * cp;
                // d(pred)/d(dtau) = model * d(exp)/d(dtau)
                double dp_re = mod_re[f] * de_re - mod_im[f] * de_im;
                double dp_im = mod_re[f] * de_im + mod_im[f] * de_re;

                if (jac[0]) {
                    // d(res)/d(tau_i) = -d(pred)/d(tau_i) = -dp (dtau = tau_i - tau_j)
                    jac[0][2*f]   = -dp_re;
                    jac[0][2*f+1] = -dp_im;
                }
                if (jac[1]) {
                    // d(res)/d(tau_j) = -d(pred)/d(tau_j) = +dp (since d(dtau)/d(tau_j) = -1)
                    jac[1][2*f]   = dp_re;
                    jac[1][2*f+1] = dp_im;
                }
            }
        }
        return true;
    }
};


py::tuple solve_delay(
    py::array_t<cd> obs_arr,      // (n_bl, n_freq, 2, 2)
    py::array_t<cd> mod_arr,      // (n_bl, n_freq, 2, 2)
    py::array_t<double> freqs_arr, // (n_freq,)
    py::array_t<int> ant1_arr,
    py::array_t<int> ant2_arr,
    int n_ant, int ref_ant, int max_iter, double tol,
    py::array_t<double> delay_init_arr)  // (n_ant, 2)
{
    auto obs = obs_arr.unchecked<4>();
    auto mod = mod_arr.unchecked<4>();
    auto freqs = freqs_arr.unchecked<1>();
    auto a1 = ant1_arr.unchecked<1>();
    auto a2 = ant2_arr.unchecked<1>();
    auto di = delay_init_arr.unchecked<2>();
    int n_bl = obs.shape(0);
    int n_freq = obs.shape(1);

    std::vector<double> delays_p(n_ant), delays_q(n_ant);
    for (int a = 0; a < n_ant; a++) {
        delays_p[a] = di(a, 0);
        delays_q[a] = di(a, 1);
    }

    // Frequency array for cost functions
    std::vector<double> freq_vec(n_freq);
    for (int f = 0; f < n_freq; f++) freq_vec[f] = freqs(f);

    ceres::Problem problem;
    for (int k = 0; k < n_bl; k++) {
        int i = a1(k), j = a2(k);

        // pol P and Q: obs and model per baseline
        std::vector<cd> obs_pp(n_freq), obs_qq(n_freq);
        std::vector<cd> mod_pp(n_freq), mod_qq(n_freq);
        for (int f = 0; f < n_freq; f++) {
            obs_pp[f] = obs(k, f, 0, 0);
            obs_qq[f] = obs(k, f, 1, 1);
            mod_pp[f] = mod(k, f, 0, 0);
            mod_qq[f] = mod(k, f, 1, 1);
        }
        problem.AddResidualBlock(
            new DelayCost(obs_pp.data(), mod_pp.data(), freq_vec.data(), n_freq),
            nullptr, &delays_p[i], &delays_p[j]);
        problem.AddResidualBlock(
            new DelayCost(obs_qq.data(), mod_qq.data(), freq_vec.data(), n_freq),
            nullptr, &delays_q[i], &delays_q[j]);
    }

    problem.SetParameterBlockConstant(&delays_p[ref_ant]);
    problem.SetParameterBlockConstant(&delays_q[ref_ant]);

    auto opts = make_opts(max_iter, tol);
    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    delays_p[ref_ant] = 0.0;
    delays_q[ref_ant] = 0.0;

    auto delay_out = py::array_t<double>({n_ant, 2});
    auto d = delay_out.mutable_unchecked<2>();
    for (int a = 0; a < n_ant; a++) {
        d(a, 0) = delays_p[a];
        d(a, 1) = delays_q[a];
    }

    // Jones at mid frequency
    double freq_mid = 0.0;
    for (int f = 0; f < n_freq; f++) freq_mid += freq_vec[f];
    freq_mid /= n_freq;

    auto jones = py::array_t<cd>({n_ant, 2, 2});
    auto j = jones.mutable_unchecked<3>();
    for (int a = 0; a < n_ant; a++) {
        double ph_p = -2.0 * M_PI * delays_p[a] * 1e-9 * freq_mid;
        double ph_q = -2.0 * M_PI * delays_q[a] * 1e-9 * freq_mid;
        j(a, 0, 0) = cd(cos(ph_p), sin(ph_p));
        j(a, 0, 1) = cd(0, 0);
        j(a, 1, 0) = cd(0, 0);
        j(a, 1, 1) = cd(cos(ph_q), sin(ph_q));
    }

    return py::make_tuple(
        jones, delay_out, summary.final_cost,
        summary.num_successful_steps,
        summary.termination_type == ceres::CONVERGENCE);
}


// ======================================================================
// KC solver — cross-hand delay (single global parameter)
// ======================================================================

struct CrossDelayCost : public ceres::CostFunction {
    std::vector<double> obs_re, obs_im, mod_re, mod_im;
    std::vector<double> twopi_nu;
    int n_total;  // n_bl * n_freq
    bool is_pq;

    CrossDelayCost(const cd* obs_data, const cd* mod_data,
                   const double* freqs, int n_bl, int n_freq, bool pq)
        : n_total(n_bl * n_freq), is_pq(pq) {
        obs_re.resize(n_total); obs_im.resize(n_total);
        mod_re.resize(n_total); mod_im.resize(n_total);
        twopi_nu.resize(n_freq);
        for (int f = 0; f < n_freq; f++)
            twopi_nu[f] = 2.0 * M_PI * freqs[f] * 1e-9;
        for (int k = 0; k < n_bl; k++) {
            for (int f = 0; f < n_freq; f++) {
                int idx = k * n_freq + f;
                obs_re[idx] = obs_data[idx].real();
                obs_im[idx] = obs_data[idx].imag();
                mod_re[idx] = mod_data[idx].real();
                mod_im[idx] = mod_data[idx].imag();
            }
        }
        set_num_residuals(n_total * 2);
        mutable_parameter_block_sizes()->push_back(1);
    }

    bool Evaluate(double const* const* p,
                  double* res, double** jac) const override {
        double tau = p[0][0];
        int n_freq = (int)twopi_nu.size();
        int n_bl = n_total / n_freq;

        for (int k = 0; k < n_bl; k++) {
            for (int f = 0; f < n_freq; f++) {
                int idx = k * n_freq + f;
                double phase = -twopi_nu[f] * tau;
                double cp = cos(phase), sp = sin(phase);
                double jr, ji;
                if (is_pq) { jr = cp; ji = sp; }
                else { jr = cp; ji = -sp; }

                double pr = jr * mod_re[idx] - ji * mod_im[idx];
                double pi_ = jr * mod_im[idx] + ji * mod_re[idx];
                res[2*idx]   = obs_re[idx] - pr;
                res[2*idx+1] = obs_im[idx] - pi_;

                if (jac && jac[0]) {
                    double dph = -twopi_nu[f];
                    double djr, dji;
                    if (is_pq) { djr = sp * dph; dji = -cp * dph; }
                    else { djr = sp * dph; dji = cp * dph; }
                    // Wait: d(cos(phase))/dtau = -sin(phase) * dphase/dtau
                    // dphase/dtau = -twopi_nu[f]
                    if (is_pq) {
                        djr = -(-sp) * twopi_nu[f];  // sp * twopi_nu
                        dji = -(cp) * (-twopi_nu[f]); // cp * twopi_nu
                    } else {
                        djr = -(-sp) * twopi_nu[f];
                        dji = -(-(- cp)) * (-twopi_nu[f]);
                    }
                    // Simpler: let me redo this cleanly
                    // phase = -twopi_nu * tau
                    // if is_pq: J = exp(i*phase) = cos(ph) + i*sin(ph)
                    //   dJ/dtau = i * (-twopi_nu) * J
                    //   dJ_re/dtau = -(-twopi_nu)*sin(ph) = twopi_nu*sin(ph)
                    //   dJ_im/dtau = (-twopi_nu)*cos(ph)
                    // if !is_pq: J = exp(-i*phase) = cos(ph) - i*sin(ph)
                    //   dJ/dtau = -i * (-twopi_nu) * J = i*twopi_nu*J
                    //   dJ_re/dtau = -twopi_nu*(-sin(ph)) = twopi_nu*sin(ph)  same?
                    //   Actually exp(-i*phase) = cos(-phase) + i*sin(-phase) = cos(phase) - i*sin(phase)
                    //   d/dtau = (-i)(-twopi_nu) exp(-i*phase) = i*twopi_nu * (cos(ph) - i*sin(ph))
                    //   = twopi_nu * (-sin(ph) + i*cos(ph))... hmm
                    // Let me just do it numerically-clean:
                    double dph_dtau = -twopi_nu[f];
                    double d_cp = -sp * dph_dtau;   // d(cos(phase))/dtau
                    double d_sp = cp * dph_dtau;     // d(sin(phase))/dtau

                    double djr_dt, dji_dt;
                    if (is_pq) {
                        djr_dt = d_cp;   // d(jr)/dtau where jr = cos(phase)
                        dji_dt = d_sp;   // d(ji)/dtau where ji = sin(phase)
                    } else {
                        djr_dt = d_cp;
                        dji_dt = -d_sp;  // ji = -sin(phase)
                    }
                    double dpr = djr_dt * mod_re[idx] - dji_dt * mod_im[idx];
                    double dpi = djr_dt * mod_im[idx] + dji_dt * mod_re[idx];
                    jac[0][2*idx]   = -dpr;
                    jac[0][2*idx+1] = -dpi;
                }
            }
        }
        return true;
    }
};


py::tuple solve_cross_delay(
    py::array_t<cd> obs_arr,       // (n_bl, n_freq, 2, 2)
    py::array_t<cd> model_arr,
    py::array_t<double> freqs_arr,
    int n_ant, int max_iter, double tol,
    double tau_init)
{
    auto obs = obs_arr.unchecked<4>();
    auto mod = model_arr.unchecked<4>();
    auto freqs = freqs_arr.unchecked<1>();
    int n_bl = obs.shape(0);
    int n_freq = obs.shape(1);

    std::vector<double> freq_vec(n_freq);
    for (int f = 0; f < n_freq; f++) freq_vec[f] = freqs(f);

    // Flatten cross-hand data
    std::vector<cd> obs_pq(n_bl * n_freq), mod_pq(n_bl * n_freq);
    std::vector<cd> obs_qp(n_bl * n_freq), mod_qp(n_bl * n_freq);
    for (int k = 0; k < n_bl; k++) {
        for (int f = 0; f < n_freq; f++) {
            obs_pq[k*n_freq+f] = obs(k, f, 0, 1);
            mod_pq[k*n_freq+f] = mod(k, f, 0, 1);
            obs_qp[k*n_freq+f] = obs(k, f, 1, 0);
            mod_qp[k*n_freq+f] = mod(k, f, 1, 0);
        }
    }

    double tau = tau_init;
    ceres::Problem problem;
    problem.AddResidualBlock(
        new CrossDelayCost(obs_pq.data(), mod_pq.data(),
                           freq_vec.data(), n_bl, n_freq, true),
        nullptr, &tau);
    problem.AddResidualBlock(
        new CrossDelayCost(obs_qp.data(), mod_qp.data(),
                           freq_vec.data(), n_bl, n_freq, false),
        nullptr, &tau);

    auto opts = make_opts(max_iter, tol);
    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    // Jones: diag(exp(-2pi*i*tau*nu_mid), 1) for all antennas
    double freq_mid = 0;
    for (int f = 0; f < n_freq; f++) freq_mid += freq_vec[f];
    freq_mid /= n_freq;

    auto jones = py::array_t<cd>({n_ant, 2, 2});
    auto j = jones.mutable_unchecked<3>();
    double ph = -2.0 * M_PI * tau * 1e-9 * freq_mid;
    for (int a = 0; a < n_ant; a++) {
        j(a, 0, 0) = cd(cos(ph), sin(ph));
        j(a, 0, 1) = cd(0, 0);
        j(a, 1, 0) = cd(0, 0);
        j(a, 1, 1) = cd(1, 0);
    }

    auto delay_out = py::array_t<double>({n_ant, 2});
    auto d = delay_out.mutable_unchecked<2>();
    for (int a = 0; a < n_ant; a++) {
        d(a, 0) = tau;
        d(a, 1) = 0.0;
    }

    return py::make_tuple(
        jones, delay_out, summary.final_cost,
        summary.num_successful_steps,
        summary.termination_type == ceres::CONVERGENCE);
}


// ======================================================================
// CP solver — cross-hand phase (single global parameter)
// ======================================================================

struct CrossPhaseCost : public ceres::SizedCostFunction<-1, 1> {
    std::vector<double> obs_re, obs_im, mod_re, mod_im;
    int n_bl;
    bool is_pq;

    CrossPhaseCost(const cd* obs_data, const cd* mod_data, int nb, bool pq)
        : n_bl(nb), is_pq(pq) {
        obs_re.resize(nb); obs_im.resize(nb);
        mod_re.resize(nb); mod_im.resize(nb);
        for (int k = 0; k < nb; k++) {
            obs_re[k] = obs_data[k].real(); obs_im[k] = obs_data[k].imag();
            mod_re[k] = mod_data[k].real(); mod_im[k] = mod_data[k].imag();
        }
        set_num_residuals(nb * 2);
    }

    bool Evaluate(double const* const* p,
                  double* res, double** jac) const override {
        double phi = p[0][0];
        double cp = cos(phi), sp = sin(phi);

        for (int k = 0; k < n_bl; k++) {
            double jr, ji;
            if (is_pq) {
                // pred = model * conj(exp(i*phi)) = model * (cp - i*sp)
                jr = cp; ji = -sp;
            } else {
                // pred = exp(i*phi) * model = (cp + i*sp) * model
                jr = cp; ji = sp;
            }
            double pr = jr * mod_re[k] - ji * mod_im[k];
            double pi_ = jr * mod_im[k] + ji * mod_re[k];
            res[2*k]   = obs_re[k] - pr;
            res[2*k+1] = obs_im[k] - pi_;

            if (jac && jac[0]) {
                double djr, dji;
                if (is_pq) {
                    djr = -sp; dji = -cp;  // d(conj(e^iphi))/dphi
                } else {
                    djr = -sp; dji = cp;   // d(e^iphi)/dphi
                }
                double dpr = djr * mod_re[k] - dji * mod_im[k];
                double dpi = djr * mod_im[k] + dji * mod_re[k];
                jac[0][2*k]   = -dpr;
                jac[0][2*k+1] = -dpi;
            }
        }
        return true;
    }
};


py::tuple solve_cross_phase(
    py::array_t<cd> obs_arr,     // (n_bl, 2, 2)
    py::array_t<cd> model_arr,
    py::array_t<int> ant1_arr,
    py::array_t<int> ant2_arr,
    int n_ant, int max_iter, double tol,
    double phi_init)
{
    auto obs = obs_arr.unchecked<3>();
    auto mod = model_arr.unchecked<3>();
    int n_bl = obs.shape(0);

    std::vector<cd> obs_pq(n_bl), mod_pq(n_bl);
    std::vector<cd> obs_qp(n_bl), mod_qp(n_bl);
    for (int k = 0; k < n_bl; k++) {
        obs_pq[k] = obs(k, 0, 1); mod_pq[k] = mod(k, 0, 1);
        obs_qp[k] = obs(k, 1, 0); mod_qp[k] = mod(k, 1, 0);
    }

    double phi = phi_init;
    ceres::Problem problem;
    problem.AddResidualBlock(
        new CrossPhaseCost(obs_pq.data(), mod_pq.data(), n_bl, true),
        nullptr, &phi);
    problem.AddResidualBlock(
        new CrossPhaseCost(obs_qp.data(), mod_qp.data(), n_bl, false),
        nullptr, &phi);

    auto opts = make_opts(max_iter, tol);
    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    auto jones = py::array_t<cd>({n_ant, 2, 2});
    auto j = jones.mutable_unchecked<3>();
    for (int a = 0; a < n_ant; a++) {
        j(a, 0, 0) = cd(1, 0);
        j(a, 0, 1) = cd(0, 0);
        j(a, 1, 0) = cd(0, 0);
        j(a, 1, 1) = cd(cos(phi), sin(phi));
    }

    return py::make_tuple(
        jones, summary.final_cost,
        summary.num_successful_steps,
        summary.termination_type == ceres::CONVERGENCE);
}


// ======================================================================
// Module definition
// ======================================================================

PYBIND11_MODULE(_cpp_solvers, m) {
    m.doc() = "ALAKAZAM C++ Ceres solvers — no Python callbacks";

    m.def("solve_gains", &solve_gains,
          "G solver: diagonal complex gains",
          py::arg("obs"), py::arg("model"),
          py::arg("ant1"), py::arg("ant2"),
          py::arg("n_ant"), py::arg("ref_ant"),
          py::arg("max_iter"), py::arg("tol"),
          py::arg("amp_init"), py::arg("phase_init"),
          py::arg("phase_only"));

    m.def("solve_leakage", &solve_leakage,
          "D solver: full 2x2 leakage",
          py::arg("obs"), py::arg("model"),
          py::arg("ant1"), py::arg("ant2"),
          py::arg("n_ant"), py::arg("ref_ant"),
          py::arg("max_iter"), py::arg("tol"),
          py::arg("dpq_init"), py::arg("dqp_init"));

    m.def("solve_delay", &solve_delay,
          "K solver: parallel delay",
          py::arg("obs"), py::arg("model"), py::arg("freqs"),
          py::arg("ant1"), py::arg("ant2"),
          py::arg("n_ant"), py::arg("ref_ant"),
          py::arg("max_iter"), py::arg("tol"),
          py::arg("delay_init"));

    m.def("solve_cross_delay", &solve_cross_delay,
          "KC solver: cross-hand delay",
          py::arg("obs"), py::arg("model"), py::arg("freqs"),
          py::arg("n_ant"), py::arg("max_iter"), py::arg("tol"),
          py::arg("tau_init"));

    m.def("solve_cross_phase", &solve_cross_phase,
          "CP solver: cross-hand phase",
          py::arg("obs"), py::arg("model"),
          py::arg("ant1"), py::arg("ant2"),
          py::arg("n_ant"), py::arg("max_iter"), py::arg("tol"),
          py::arg("phi_init"));
}
